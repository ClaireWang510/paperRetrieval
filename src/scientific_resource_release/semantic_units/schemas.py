from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FactUnit(BaseModel):
    statement: str = Field(..., description="Atomic fact statement in English")
    semantic_label: Optional[str] = Field(default=None, description="Extraction-stage semantic label")
    source_section: Optional[str] = Field(default=None, description="Source section title")
    source_type: str = Field(default="text", description="text | figure | table")
    source_id: Optional[str] = Field(default=None, description="Source figure/table item_id if available")
    core_entities: Optional[List[str]] = Field(default=None, description="Reserved for future entity-aware clustering")


class SemanticUnit(BaseModel):
    id: Optional[str] = Field(default=None)
    arxiv_id: str = Field(...)
    semantic_role: Optional[str] = Field(default=None)
    title: str = Field(...)
    content: str = Field(...)
    keywords: List[str] = Field(default_factory=list)
    source_section_hints: List[str] = Field(default_factory=list)
    cluster_index: int = Field(default=0)
    fact_count: int = Field(default=0)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_storage_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "arxiv_id": self.arxiv_id,
            "semantic_role": self.semantic_role,
            "title": self.title,
            "content": self.content,
            "keywords": self.keywords,
            "source_section_hints": self.source_section_hints,
            "cluster_index": self.cluster_index,
            "fact_count": self.fact_count,
            "extra": self.extra,
        }


class PaperSemanticUnits(BaseModel):
    arxiv_id: str = Field(...)
    paper_title: Optional[str] = Field(default=None)
    semantic_units: List[SemanticUnit] = Field(default_factory=list)