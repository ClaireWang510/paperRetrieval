from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    published_date_start: Optional[date] = None
    published_date_end: Optional[date] = None
    roles: Optional[List[str]] = None
    venues: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    sparse_query: Optional[str] = None
    dense_query: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[SearchFilters] = None

    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    doc_level_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    semantic_top_n: int = Field(default=2, ge=1, le=10)
    unit_source: str = "semantic_units"
    unit_pooling: str = "top_n_mean"
    intent_channels: Optional[List[str]] = None
    intent_decomposer: str = "llm"
    rewritten_query: Optional[str] = None
    exclude_unit_roles: Optional[List[str]] = None


class SearchResultItem(BaseModel):
    arxiv_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    score: float


class SearchResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    intent_queries: Optional[List[str]] = None
    total: int
    results: List[SearchResultItem]
