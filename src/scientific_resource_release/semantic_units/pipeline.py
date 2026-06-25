from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scientific_resource_release.models import SemanticUnitConfig
from scientific_resource_release.utils.io import read_json
from scientific_resource_release.utils.paths import resolve_project_paths


logger = logging.getLogger(__name__)


class SemanticUnitReleasePipeline:
    """Standalone semantic unit generation pipeline."""

    def __init__(self, config: SemanticUnitConfig):
        self.config = config
        self.paths = resolve_project_paths(config.project_root)
        self.extracted_dir = config.extracted_dir or self.paths.extracted_root
        self.output_dir = config.output_dir or self.extracted_dir

    def run_one(self, arxiv_id: str) -> Path:
        clean_id = arxiv_id.split("v")[0]
        out_path = self.output_dir / clean_id / "semantic_units.json"
        if self.config.skip_existing and out_path.exists():
            logger.info("Skip %s: semantic_units.json already exists", clean_id)
            return out_path

        paper_dir = self.extracted_dir / clean_id
        metadata_path = paper_dir / "metadata.json"
        annotation_path = paper_dir / "annotation.json"
        semantics_path = paper_dir / "figure_table_semantics.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

        metadata = read_json(metadata_path)
        annotation = read_json(annotation_path) if annotation_path.exists() else {"sections": []}
        figure_sem = read_json(semantics_path) if semantics_path.exists() else {"items": []}

        facts = self._collect_facts(metadata, annotation, figure_sem)
        units = self._build_semantic_units(clean_id, metadata.get("title"), facts)
        result = {
            "arxiv_id": clean_id,
            "paper_title": metadata.get("title"),
            "semantic_units": units,
        }

        self._write_result(out_path, result)
        logger.info(
            "Semantic units done for %s: facts=%s clusters=%s units=%s",
            clean_id,
            len(facts),
            len({x.get('semantic_label') for x in facts if x.get('semantic_label')}),
            len(units),
        )
        return out_path

    def run_batch(self, arxiv_ids: List[str]) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}
        for arxiv_id in arxiv_ids:
            clean_id = arxiv_id.split("v")[0]
            try:
                outputs[clean_id] = self.run_one(clean_id)
            except Exception as exc:
                logger.exception("Semantic unit generation failed for %s: %s", clean_id, exc)
        return outputs

    def run_batch_from_extracted(self) -> Tuple[Dict[str, Path], List[str]]:
        ids: List[str] = []
        for subdir in sorted(self.extracted_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / "metadata.json").exists():
                ids.append(subdir.name)
        outputs = self.run_batch(ids)
        return outputs, ids

    def _collect_facts(
        self,
        metadata: Dict[str, Any],
        annotation: Dict[str, Any],
        figure_semantics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []

        ann_lookup = self._flatten_annotations(annotation.get("sections") or [])
        for sec in self._flatten_sections(metadata.get("sections") or []):
            title = sec.get("title", "")
            content = sec.get("content", "")
            if not content.strip():
                continue

            fig_ids, tab_ids = ann_lookup.get(title, ([], []))
            sec_facts = self._extract_facts_from_text(
                section_title=title,
                section_content=content,
                figure_ids=fig_ids,
                table_ids=tab_ids,
            )
            facts.extend(sec_facts)

        if self.config.use_figure_semantics:
            for item in figure_semantics.get("items", []):
                summary = item.get("semantic_summary") or ""
                if not summary:
                    continue
                facts.append(
                    {
                        "statement": summary,
                        "semantic_label": "evidence",
                        "source_section": "figure_table_semantics",
                        "source_type": item.get("item_type", "figure"),
                        "source_id": item.get("item_id"),
                    }
                )
        return facts

    def _extract_facts_from_text(
        self,
        section_title: str,
        section_content: str,
        figure_ids: List[str],
        table_ids: List[str],
    ) -> List[Dict[str, Any]]:
        if self.config.llm_api_key:
            llm_facts = self._extract_facts_with_llm(section_title, section_content, figure_ids, table_ids)
            if llm_facts:
                return llm_facts

        sentences = re.split(r"(?<=[.!?])\s+", section_content)
        out = []
        for s in sentences:
            s = s.strip()
            if len(s) < 40:
                continue
            label = self._guess_label(s)
            out.append(
                {
                    "statement": s,
                    "semantic_label": label,
                    "source_section": section_title,
                    "source_type": "text",
                    "source_id": None,
                }
            )
        return out[:20]

    def _build_semantic_units(
        self,
        arxiv_id: str,
        paper_title: Optional[str],
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not facts:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for fact in facts:
            label = fact.get("semantic_label") or "other"
            grouped.setdefault(label, []).append(fact)

        # Keep most informative groups only.
        groups = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        units: List[Dict[str, Any]] = []
        for i, (label, cluster) in enumerate(groups):
            if self.config.llm_api_key:
                llm_unit = self._synthesize_unit_with_llm(arxiv_id, i, label, cluster)
                if llm_unit:
                    units.append(llm_unit)
                    continue

            top_sentences = [x.get("statement", "") for x in cluster[:5] if x.get("statement")]
            content = " ".join(top_sentences)
            keywords = self._extract_keywords(content)
            sections = sorted({x.get("source_section") for x in cluster if x.get("source_section")})
            units.append(
                {
                    "id": f"{arxiv_id}_{i}",
                    "arxiv_id": arxiv_id,
                    "semantic_role": label,
                    "title": f"{label.title()} summary",
                    "content": content,
                    "keywords": keywords,
                    "source_section_hints": sections,
                    "cluster_index": i,
                    "fact_count": len(cluster),
                    "extra": {"paper_title": paper_title},
                }
            )
        return units

    @staticmethod
    def _flatten_sections(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in nodes:
            out.append(node)
            out.extend(SemanticUnitReleasePipeline._flatten_sections(node.get("children") or []))
        return out

    @staticmethod
    def _flatten_annotations(nodes: List[Dict[str, Any]]) -> Dict[str, Tuple[List[str], List[str]]]:
        out: Dict[str, Tuple[List[str], List[str]]] = {}
        for node in nodes:
            title = node.get("title", "")
            out[title] = (node.get("figure_ids") or [], node.get("table_ids") or [])
            out.update(SemanticUnitReleasePipeline._flatten_annotations(node.get("children") or []))
        return out

    @staticmethod
    def _guess_label(sentence: str) -> str:
        s = sentence.lower()
        if any(x in s for x in ["we propose", "our method", "framework", "architecture"]):
            return "method"
        if any(x in s for x in ["experiment", "dataset", "benchmark", "evaluation"]):
            return "experiment"
        if any(x in s for x in ["result", "improve", "outperform", "achieve"]):
            return "result"
        if any(x in s for x in ["contribution", "main contribution"]):
            return "contribution"
        return "other"

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
        stop = {"the", "and", "for", "with", "from", "this", "that", "are", "was", "were"}
        out = []
        seen = set()
        for t in tokens:
            if t in stop or t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= 10:
                break
        return out

    def _extract_facts_with_llm(
        self,
        section_title: str,
        section_content: str,
        figure_ids: List[str],
        table_ids: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        payload = {
            "section_title": section_title,
            "section_content": section_content[:6000],
            "figure_ids": figure_ids,
            "table_ids": table_ids,
        }
        prompt = (
            "Extract atomic facts from this section. "
            "Return STRICT JSON array. Each item must contain: "
            "statement, semantic_label, source_section, source_type, source_id.\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        content = self._call_llm(prompt)
        if not content:
            return None
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return None
        return None

    def _synthesize_unit_with_llm(
        self,
        arxiv_id: str,
        idx: int,
        label: str,
        cluster: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        prompt = (
            "You are synthesizing a semantic unit for scientific retrieval. "
            "Return STRICT JSON object with keys: title, content, keywords (array), source_section_hints (array).\n"
            f"semantic_role={label}\n"
            f"facts={json.dumps(cluster[:12], ensure_ascii=False)}"
        )
        content = self._call_llm(prompt)
        if not content:
            return None
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        return {
            "id": f"{arxiv_id}_{idx}",
            "arxiv_id": arxiv_id,
            "semantic_role": label,
            "title": obj.get("title") or f"{label.title()} summary",
            "content": obj.get("content") or "",
            "keywords": obj.get("keywords") or [],
            "source_section_hints": obj.get("source_section_hints") or [],
            "cluster_index": idx,
            "fact_count": len(cluster),
            "extra": {},
        }

    def _call_llm(self, prompt: str) -> Optional[str]:
        api_key = (self.config.llm_api_key or "").strip()
        if not api_key:
            return None
        base_url = (self.config.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        model = self.config.llm_model or "gpt-4o-mini"
        body = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.llm_timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_result(out_path: Path, result: Dict[str, Any]) -> None:
        data = result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
