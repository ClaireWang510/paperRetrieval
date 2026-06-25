from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional


@dataclass
class BaseConfig:
    project_root: Path


@dataclass
class PreprocessConfig(BaseConfig):
    jsonl_path: Optional[Path] = None
    latex_dir: Optional[Path] = None
    pdf_dir: Optional[Path] = None
    extracted_dir: Optional[Path] = None
    run_annotation: bool = True
    run_figure_table_semantics: bool = True
    refresh_semantic_scholar: bool = True
    skip_done: bool = True
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if self.llm_api_key is None:
            self.llm_api_key = os.getenv("LLM_API_KEY")
        if self.llm_base_url is None:
            self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        if self.llm_model is None:
            self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")


@dataclass
class SemanticUnitConfig(BaseConfig):
    extracted_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    use_figure_semantics: bool = True
    cluster_method: str = "label_based"
    n_clusters: Optional[int] = None
    use_llm_label_merge: bool = True
    skip_existing: bool = True
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if self.llm_api_key is None:
            self.llm_api_key = os.getenv("LLM_API_KEY")
        if self.llm_base_url is None:
            self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        if self.llm_model is None:
            self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
