from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scientific_resource_release.utils.io import read_json


def load_paper(
    arxiv_id: str,
    data_root,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    paper_dir = data_root / arxiv_id
    metadata_path = paper_dir / "metadata.json"
    annotation_path = paper_dir / "annotation.json"
    semantics_path = paper_dir / "figure_table_semantics.json"

    if not metadata_path.exists():
        raise FileNotFoundError("metadata.json not found: %s" % metadata_path)

    metadata = read_json(metadata_path)
    annotation = read_json(annotation_path) if annotation_path.exists() else {"sections": []}

    figure_table_semantics = {}
    if semantics_path.exists():
        semantics = read_json(semantics_path)
        for item in semantics.get("items", []):
            if not isinstance(item, dict):
                continue
            item_id = item.get("item_id")
            if item_id:
                figure_table_semantics[str(item_id)] = item

    return metadata, annotation, figure_table_semantics


def walk_sections(
    sections: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    prefix_title: str = "",
) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    result = []
    for section, annotation in zip(sections, annotations):
        title = section.get("title", "")
        full_title = "%s > %s" % (prefix_title, title) if prefix_title else title
        result.append((section, annotation, full_title))
        child_sections = section.get("children") or []
        child_annotations = annotation.get("children") or []
        if child_sections and child_annotations:
            result.extend(walk_sections(child_sections, child_annotations, full_title))
    return result