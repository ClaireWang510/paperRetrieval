from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scientific_resource_release.models import SemanticUnitConfig
from scientific_resource_release.utils.paths import resolve_project_paths

from .clustering import cluster_facts
from .data_loader import load_paper, walk_sections
from .fact_extraction import extract_fact_units_from_section, should_skip_section
from .label_merging import get_merge_map_for_facts, group_facts_by_merged_labels, groups_to_ordered_label_cluster_pairs
from .llm_client import SemanticUnitLLMClient
from .schemas import FactUnit, PaperSemanticUnits
from .semantic_unit_synthesis import synthesize_semantic_unit


logger = logging.getLogger(__name__)


class SemanticUnitReleasePipeline:
    MAX_UNITS_PER_PAPER = 10

    def __init__(self, config: SemanticUnitConfig):
        self.config = config
        self.paths = resolve_project_paths(config.project_root)
        self.extracted_dir = config.extracted_dir or self.paths.extracted_root
        self.output_dir = config.output_dir or self.extracted_dir
        self.llm_client = SemanticUnitLLMClient(config)

    def run_one(self, arxiv_id: str) -> Path:
        clean_id = arxiv_id.split("v")[0]
        out_path = self.output_dir / clean_id / "semantic_units.json"
        if self.config.skip_existing and out_path.exists():
            logger.info("Skip %s: semantic_units.json already exists", clean_id)
            return out_path

        result, stats = self._process_one(clean_id)
        self._write_result(out_path, result)
        logger.info(
            "Semantic units done for %s: facts=%s clusters=%s units=%s",
            clean_id,
            stats["n_facts"],
            stats["n_clusters"],
            stats["n_units"],
        )
        return out_path

    def run_batch(self, arxiv_ids: List[str]) -> Dict[str, Path]:
        outputs = {}
        for arxiv_id in arxiv_ids:
            clean_id = arxiv_id.split("v")[0]
            try:
                outputs[clean_id] = self.run_one(clean_id)
            except Exception as exc:
                logger.exception("Semantic unit generation failed for %s: %s", clean_id, exc)
        return outputs

    def run_batch_from_extracted(self) -> Tuple[Dict[str, Path], List[str]]:
        arxiv_ids = []
        for subdir in sorted(self.extracted_dir.iterdir()):
            if subdir.is_dir() and (subdir / "metadata.json").exists():
                arxiv_ids.append(subdir.name)
        return self.run_batch(arxiv_ids), arxiv_ids

    def _process_one(self, arxiv_id: str) -> Tuple[PaperSemanticUnits, Dict[str, Any]]:
        stats = {"n_facts": 0, "n_clusters": 0, "n_units": 0}
        metadata, annotation, figure_table_semantics = load_paper(arxiv_id, self.extracted_dir)
        facts = self._collect_facts(metadata, annotation, figure_table_semantics)
        stats["n_facts"] = len(facts)

        if not facts:
            return PaperSemanticUnits(arxiv_id=arxiv_id, paper_title=metadata.get("title"), semantic_units=[]), stats

        ordered_labels = []
        if self.config.cluster_method == "label_based":
            merge_map = get_merge_map_for_facts(
                facts,
                use_llm_merge=self.config.use_llm_label_merge,
                llm_client=self.llm_client,
            )
            groups = group_facts_by_merged_labels(facts, merge_map)
            if len(groups) > self.MAX_UNITS_PER_PAPER:
                selected_labels = {
                    label
                    for label, _ in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[: self.MAX_UNITS_PER_PAPER]
                }
                groups = {label: cluster for label, cluster in groups.items() if label in selected_labels}
            label_cluster_pairs = groups_to_ordered_label_cluster_pairs(groups)
            clusters = [cluster for _, cluster in label_cluster_pairs]
            ordered_labels = [label for label, _ in label_cluster_pairs]
        else:
            clusters = cluster_facts(
                facts,
                method=self.config.cluster_method,
                n_clusters=self.config.n_clusters,
                llm_client=self.llm_client,
            )
            if len(clusters) > self.MAX_UNITS_PER_PAPER:
                clusters = sorted(clusters, key=lambda cluster: len(cluster), reverse=True)[: self.MAX_UNITS_PER_PAPER]

        stats["n_clusters"] = len(clusters)

        units = []
        for index, cluster in enumerate(clusters):
            if not cluster:
                continue
            units.append(
                synthesize_semantic_unit(
                    cluster,
                    arxiv_id=arxiv_id,
                    cluster_index=index,
                    unit_id="%s_%s" % (arxiv_id, index),
                    suggested_semantic_role=ordered_labels[index] if index < len(ordered_labels) else None,
                    llm_client=self.llm_client,
                    paper_title=metadata.get("title"),
                )
            )
        stats["n_units"] = len(units)
        return PaperSemanticUnits(arxiv_id=arxiv_id, paper_title=metadata.get("title"), semantic_units=units), stats

    def _collect_facts(
        self,
        metadata: Dict[str, Any],
        annotation: Dict[str, Any],
        figure_table_semantics: Dict[str, Dict[str, Any]],
    ) -> List[FactUnit]:
        sections = metadata.get("sections") or []
        section_annotations = annotation.get("sections") or []
        if len(section_annotations) != len(sections):
            section_annotations = _align_annotations(sections, section_annotations)

        all_units = []
        for section, section_annotation, full_title in walk_sections(sections, section_annotations):
            content = (section.get("content") or "").strip()
            if not content:
                continue
            if should_skip_section(full_title):
                continue
            figure_ids = section_annotation.get("figure_ids") or []
            table_ids = section_annotation.get("table_ids") or []
            units, _, _ = extract_fact_units_from_section(
                section_title=full_title,
                section_content=content,
                figure_ids=figure_ids,
                table_ids=table_ids,
                figure_table_semantics=figure_table_semantics,
                llm_client=self.llm_client,
                use_figure_semantics=self.config.use_figure_semantics,
            )
            all_units.extend(units)
        return all_units

    @staticmethod
    def _write_result(out_path: Path, result: PaperSemanticUnits) -> None:
        payload = {
            "arxiv_id": result.arxiv_id,
            "paper_title": result.paper_title,
            "semantic_units": [unit.model_dump() for unit in result.semantic_units],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def _align_annotations(sections: List[Dict[str, Any]], section_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for index, section in enumerate(sections):
        annotation = section_annotations[index] if index < len(section_annotations) else {}
        result.append(
            {
                "title": annotation.get("title", section.get("title", "")),
                "figure_ids": annotation.get("figure_ids", []),
                "table_ids": annotation.get("table_ids", []),
                "children": _align_annotations(section.get("children", []), annotation.get("children", [])),
            }
        )
    return result
