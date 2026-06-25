from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from scientific_resource_release.retrieval.schemas import SearchFilters

PaperRow = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[int]]


def _tokenize(text_value: str) -> List[str]:
    if not text_value:
        return []
    return [tok for tok in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text_value.lower()) if tok]


def load_papers(db: Session, filters: Optional[SearchFilters]) -> Tuple[List[PaperRow], List[str]]:
    where_clauses = ["1=1"]
    params: Dict[str, Any] = {}
    if filters:
        if filters.published_date_start:
            where_clauses.append("p.published_date >= :date_start")
            params["date_start"] = filters.published_date_start
        if filters.published_date_end:
            where_clauses.append("p.published_date <= :date_end")
            params["date_end"] = filters.published_date_end
        if filters.venues:
            clauses = []
            for idx, venue in enumerate(filters.venues):
                value = (venue or "").strip()
                if not value:
                    continue
                key = f"venue_{idx}"
                clauses.append(f"p.venue ILIKE :{key}")
                params[key] = f"%{value}%"
            if clauses:
                where_clauses.append("(" + " OR ".join(clauses) + ")")

    sql = f"""
        SELECT p.arxiv_id, p.title, p.abstract, p.venue, p.citation_count
        FROM papers_metadata p
        WHERE {' AND '.join(where_clauses)}
    """
    rows = [tuple(r) for r in db.execute(text(sql), params).fetchall()]
    docs = [f"{r[1] or ''} {r[2] or ''}".strip() for r in rows]
    return rows, docs


def bm25_search(
    db: Session,
    query: str,
    filters: Optional[SearchFilters],
    limit: int,
) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[int], float]]:
    from rank_bm25 import BM25Okapi

    if not (query or "").strip():
        return []

    rows, docs = load_papers(db, filters)
    if not rows:
        return []

    tokenized_corpus = [_tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    indexed = [(idx, float(score)) for idx, score in enumerate(scores)]
    indexed.sort(key=lambda pair: pair[1], reverse=True)

    out = []
    for idx, score in indexed[:limit]:
        row = rows[idx]
        out.append((row[0], row[1], row[2], row[3], row[4], score))
    return out
