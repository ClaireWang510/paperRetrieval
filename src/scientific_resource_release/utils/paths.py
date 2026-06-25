from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    extracted_root: Path
    latex_root: Path
    pdf_root: Path
    default_jsonl: Path


def resolve_project_paths(project_root: Path) -> ProjectPaths:
    root = project_root.resolve()
    data_root = root / "data"
    return ProjectPaths(
        project_root=root,
        data_root=data_root,
        extracted_root=data_root / "extracted",
        latex_root=data_root / "raw" / "latex",
        pdf_root=data_root / "raw" / "pdfs",
        default_jsonl=data_root / "raw" / "arxiv_ai_2024_2025_augmented.jsonl",
    )
