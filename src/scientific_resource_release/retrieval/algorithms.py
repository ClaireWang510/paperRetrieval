from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from scientific_resource_release.retrieval.embedding import get_embedding_model
from scientific_resource_release.retrieval.intent import build_intent_retrieval_payload
from scientific_resource_release.retrieval.keyword import bm25_search, load_papers
from scientific_resource_release.retrieval.schemas import SearchFilters, SearchRequest, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

_MIN_CANDIDATE_LIMIT = 50
_MAX_CANDIDATE_LIMIT = 200
_RRF_K = 60
_INTENT_CHANNEL_CANDIDATE_LIMIT = 150
_ORIGINAL_REWRITE_CHANNEL_WEIGHT = 1.5
_SPARSE_SPLIT_RE = re.compile(r"[,，;；|、\n]+")

_DOC_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}


def _embedding_to_float_array(raw_embedding: Any) -> np.ndarray:
    if raw_embedding is None:
        return np.asarray([], dtype=np.float32)

    if isinstance(raw_embedding, np.ndarray):
        return np.asarray(raw_embedding, dtype=np.float32).reshape(-1)

    if isinstance(raw_embedding, (list, tuple)):
        return np.asarray(raw_embedding, dtype=np.float32).reshape(-1)

    # Some datasets persist vectors as text like "[0.1,0.2,...]" in PostgreSQL text columns.
    if isinstance(raw_embedding, str):
        text_value = raw_embedding.strip()
        if not text_value:
            return np.asarray([], dtype=np.float32)
        try:
            parsed = json.loads(text_value)
            return np.asarray(parsed, dtype=np.float32).reshape(-1)
        except Exception:
            cleaned = text_value.strip("[](){}")
            if not cleaned:
                return np.asarray([], dtype=np.float32)
            return np.fromstring(cleaned, sep=",", dtype=np.float32).reshape(-1)

    return np.asarray(raw_embedding, dtype=np.float32).reshape(-1)


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.0] * len(scores)
    return [(score - lo) / (hi - lo) for score in scores]


def _limit_sparse_query_keywords(sparse_query_text: str, max_keywords: int = 3) -> str:
    text_value = (sparse_query_text or "").strip()
    if not text_value:
        return ""
    parts = [part.strip() for part in _SPARSE_SPLIT_RE.split(text_value) if part and part.strip()]
    if len(parts) <= 1:
        return text_value

    out: List[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
        if len(out) >= max_keywords:
            break
    return " ".join(out) if out else text_value


def _dense_search_title_abstract(
    db: Session,
    dense_query: str,
    filters: Optional[SearchFilters],
    limit: int,
) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[int], float]]:
    if not (dense_query or "").strip():
        return []

    rows, doc_texts = load_papers(db, filters)
    if not rows:
        return []

    model = get_embedding_model()
    query_embedding = model.encode(dense_query, normalize_embeddings=True)
    query_vec = np.asarray(
        query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding,
        dtype=np.float32,
    )

    ids_need_lookup: List[str] = [row[0] for row in rows if row[0] not in _DOC_EMBEDDING_CACHE]
    if ids_need_lookup:
        sql = text(
            """
            SELECT arxiv_id, embedding FROM paper_dense_embeddings
            WHERE arxiv_id = ANY(:arxiv_ids)
            """
        )
        for arxiv_id, embedding in db.execute(sql, {"arxiv_ids": ids_need_lookup}).fetchall():
            if embedding is None:
                continue
            emb_arr = _embedding_to_float_array(embedding)
            if emb_arr.size > 0:
                _DOC_EMBEDDING_CACHE[str(arxiv_id)] = emb_arr

    missing_idx: List[int] = []
    missing_texts: List[str] = []
    for idx, row in enumerate(rows):
        if row[0] not in _DOC_EMBEDDING_CACHE:
            missing_idx.append(idx)
            missing_texts.append(doc_texts[idx])

    if missing_texts:
        encoded = model.encode(missing_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        encoded_arr = np.asarray(encoded, dtype=np.float32)
        for idx, emb in zip(missing_idx, encoded_arr):
            _DOC_EMBEDDING_CACHE[rows[idx][0]] = emb

    available_rows = [row for row in rows if row[0] in _DOC_EMBEDDING_CACHE]
    if not available_rows:
        return []

    matrix = np.vstack([_DOC_EMBEDDING_CACHE[row[0]] for row in available_rows])
    scores = matrix @ query_vec
    if limit >= len(available_rows):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -limit)[-limit:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    out: List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[int], float]] = []
    for idx in top_indices.tolist():
        row = available_rows[idx]
        out.append((row[0], row[1], row[2], row[3], row[4], float(scores[idx])))
    return out


def _dense_search_units(
    db: Session,
    vec_str: str,
    filters: Optional[SearchFilters],
    limit: int,
    unit_source: str,
    unit_pooling: str,
    top_n: int,
    exclude_unit_roles: Optional[List[str]] = None,
) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[int], float]]:
    chunk_table = "paper_fulltext_chunks" if unit_source.strip().lower() in {"chunks", "fulltext_chunks"} else "semantic_chunks"
    use_max = unit_pooling.strip().lower() in {"max", "top1"}

    where_clauses = ["c.embedding IS NOT NULL"]
    params: Dict[str, Any] = {
        "vec": vec_str,
        "chunk_limit": max(limit * 40, 4000),
        "top_n": max(1, int(top_n)),
        "paper_limit": limit,
    }

    if filters:
        if filters.published_date_start:
            where_clauses.append("p.published_date >= :date_start")
            params["date_start"] = filters.published_date_start
        if filters.published_date_end:
            where_clauses.append("p.published_date <= :date_end")
            params["date_end"] = filters.published_date_end
        if filters.roles and chunk_table == "semantic_chunks":
            where_clauses.append("c.role = ANY(:roles)")
            params["roles"] = list(filters.roles)
        if filters.venues:
            venue_clauses = []
            for idx, venue in enumerate(filters.venues):
                value = (venue or "").strip()
                if not value:
                    continue
                key = f"venue_{idx}"
                venue_clauses.append(f"p.venue ILIKE :{key}")
                params[key] = f"%{value}%"
            if venue_clauses:
                where_clauses.append("(" + " OR ".join(venue_clauses) + ")")

    if exclude_unit_roles and chunk_table == "semantic_chunks":
        normalized = [str(role).strip().lower() for role in exclude_unit_roles if str(role).strip()]
        if normalized:
            where_clauses.append("COALESCE(LOWER(c.role), '') != ALL(:exclude_unit_roles)")
            params["exclude_unit_roles"] = normalized

    agg_expr = "MAX(r.score)::float" if use_max else "AVG(r.score)::float"
    sql = f"""
        WITH top_chunks AS (
            SELECT
                c.arxiv_id,
                p.title,
                p.abstract,
                p.venue,
                p.citation_count,
                (1 - (c.embedding <=> CAST(:vec AS vector)))::float AS score
            FROM {chunk_table} c
            JOIN papers_metadata p ON p.arxiv_id = c.arxiv_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY c.embedding <=> CAST(:vec AS vector)
            LIMIT :chunk_limit
        ),
        ranked AS (
            SELECT
                t.arxiv_id,
                t.title,
                t.abstract,
                t.venue,
                t.citation_count,
                t.score,
                ROW_NUMBER() OVER (PARTITION BY t.arxiv_id ORDER BY t.score DESC) AS rn
            FROM top_chunks t
        )
        SELECT
            r.arxiv_id,
            MAX(r.title) AS title,
            MAX(r.abstract) AS abstract,
            MAX(r.venue) AS venue,
            MAX(r.citation_count) AS citation_count,
            {agg_expr} AS score
        FROM ranked r
        WHERE r.rn <= :top_n
        GROUP BY r.arxiv_id
        ORDER BY score DESC
        LIMIT :paper_limit
    """
    return [tuple(row) for row in db.execute(text(sql), params).fetchall()]


def _build_doc_hybrid_by_id(req: SearchRequest, db: Session, candidate_limit: int):
    sparse_query_text = _limit_sparse_query_keywords((req.sparse_query or req.query or "").strip(), max_keywords=3)
    dense_query_text = (req.dense_query or req.query or "").strip()

    sparse_rows = bm25_search(db, sparse_query_text, req.filters, candidate_limit)
    sparse_norm = _normalize_scores([float(row[5]) for row in sparse_rows])
    sparse_by_id: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], Optional[int], float]] = {}
    for idx, row in enumerate(sparse_rows):
        sparse_by_id[row[0]] = (row[1], row[2], row[3], row[4], sparse_norm[idx] if idx < len(sparse_norm) else 0.0)

    dense_rows: List[Tuple[Any, ...]] = []
    if req.sparse_weight < 1.0 and dense_query_text:
        dense_rows = _dense_search_title_abstract(db, dense_query_text, req.filters, candidate_limit)
    dense_norm = _normalize_scores([float(row[5]) for row in dense_rows])
    dense_by_id: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], Optional[int], float]] = {}
    for idx, row in enumerate(dense_rows):
        dense_by_id[row[0]] = (row[1], row[2], row[3], row[4], dense_norm[idx] if idx < len(dense_norm) else 0.0)

    all_ids = list({*sparse_by_id.keys(), *dense_by_id.keys()})
    out: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], Optional[int], float]] = {}
    for arxiv_id in all_ids:
        title, abstract, venue, citation_count = None, None, None, None
        s_score, d_score = 0.0, 0.0
        if arxiv_id in sparse_by_id:
            title, abstract, venue, citation_count, s_score = sparse_by_id[arxiv_id]
        if arxiv_id in dense_by_id:
            t2, a2, v2, c2, d_score = dense_by_id[arxiv_id]
            if title is None:
                title, abstract, venue, citation_count = t2, a2, v2, c2
        score = req.sparse_weight * s_score + (1.0 - req.sparse_weight) * d_score
        out[arxiv_id] = (title, abstract, venue, citation_count, score)
    return out


def run_search_hybrid_doc_semantic_units(req: SearchRequest, db: Session) -> SearchResponse:
    candidate_limit = min(max(req.top_k * 8, _MIN_CANDIDATE_LIMIT), _MAX_CANDIDATE_LIMIT)
    dense_query_text = (req.dense_query or req.query or "").strip()
    doc_level_weight = float(req.doc_level_weight)

    doc_by_id = _build_doc_hybrid_by_id(req, db, candidate_limit)

    unit_by_id: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], Optional[int], float]] = {}
    if dense_query_text and doc_level_weight < 1.0:
        try:
            model = get_embedding_model()
            query_embedding = model.encode(dense_query_text, normalize_embeddings=True)
            query_vec = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
            vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"
            unit_rows = _dense_search_units(
                db=db,
                vec_str=vec_str,
                filters=req.filters,
                limit=candidate_limit,
                unit_source=req.unit_source,
                unit_pooling=req.unit_pooling,
                top_n=req.semantic_top_n,
                exclude_unit_roles=req.exclude_unit_roles,
            )
            unit_norm = _normalize_scores([float(row[5]) for row in unit_rows])
            for idx, row in enumerate(unit_rows):
                unit_by_id[row[0]] = (row[1], row[2], row[3], row[4], unit_norm[idx] if idx < len(unit_norm) else 0.0)
        except Exception as exc:
            logger.warning("Unit-level dense retrieval failed, fallback to doc-level only: %s", exc)

    all_ids = list({*doc_by_id.keys(), *unit_by_id.keys()})
    combined = []
    for arxiv_id in all_ids:
        title, abstract, venue, citation_count = None, None, None, None
        doc_score, unit_score = 0.0, 0.0
        if arxiv_id in doc_by_id:
            title, abstract, venue, citation_count, doc_score = doc_by_id[arxiv_id]
        if arxiv_id in unit_by_id:
            t2, a2, v2, c2, unit_score = unit_by_id[arxiv_id]
            if title is None:
                title, abstract, venue, citation_count = t2, a2, v2, c2
        final_score = doc_level_weight * doc_score + (1.0 - doc_level_weight) * unit_score
        combined.append((arxiv_id, title, abstract, venue, citation_count, final_score))

    combined.sort(key=lambda row: row[5], reverse=True)
    items = [
        SearchResultItem(
            arxiv_id=row[0],
            title=row[1],
            abstract=row[2],
            venue=row[3],
            citation_count=row[4],
            score=round(float(row[5]), 4),
        )
        for row in combined[: req.top_k]
    ]
    return SearchResponse(query=req.query, total=len(items), results=items)


def _build_intent_channel_specs(req: SearchRequest, base_query: str) -> Dict[str, Any]:
    requested_decomposer = (req.intent_decomposer or "").strip().lower()
    effective_decomposer = requested_decomposer if requested_decomposer in {"llm", "relational", "lexical", "none"} else "llm"
    payload = build_intent_retrieval_payload(
        base_query=base_query,
        intent_decomposer=effective_decomposer,
        intent_channels=req.intent_channels,
        base_filters=req.filters,
    )
    rewritten_query = str(payload.get("rewritten_query") or "").strip() or base_query
    task_specs = list(payload.get("task_specs") or [])
    if not task_specs:
        task_specs = [{"sparse_query": rewritten_query, "dense_query": rewritten_query, "filters": req.filters}]

    channel_specs = [
        {
            "channel": "base_rewrite",
            "sparse_query": rewritten_query,
            "dense_query": rewritten_query,
            "filters": req.filters,
            "weight": _ORIGINAL_REWRITE_CHANNEL_WEIGHT,
        }
    ]

    for idx, spec in enumerate(task_specs, start=1):
        sparse_query = str(spec.get("sparse_query") or "").strip() or rewritten_query
        dense_query = str(spec.get("dense_query") or "").strip() or rewritten_query
        filters = spec.get("filters", req.filters)
        if sparse_query.lower() == rewritten_query.lower() and dense_query.lower() == rewritten_query.lower() and filters == req.filters:
            continue
        channel_specs.append(
            {
                "channel": f"task_{idx}",
                "sparse_query": sparse_query,
                "dense_query": dense_query,
                "filters": filters,
                "weight": 1.0,
            }
        )

    intent_queries = []
    for value in [rewritten_query, *list(payload.get("intent_queries") or [])]:
        text_value = str(value or "").strip()
        if text_value and text_value.lower() not in {q.lower() for q in intent_queries}:
            intent_queries.append(text_value)

    return {"rewritten_query": rewritten_query, "channel_specs": channel_specs, "intent_queries": intent_queries}


def run_search_hybrid_doc_semantic_units_intent(req: SearchRequest, db: Session) -> SearchResponse:
    base_query = (req.query or req.dense_query or req.sparse_query or "").strip()
    if not base_query:
        return SearchResponse(query=req.query, rewritten_query=None, intent_queries=None, total=0, results=[])

    payload = _build_intent_channel_specs(req, base_query)
    channel_specs = payload.get("channel_specs") or []
    rewritten_query = str(payload.get("rewritten_query") or "").strip() or base_query
    intent_queries = list(payload.get("intent_queries") or [])

    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for spec in channel_specs:
        weight = float(spec.get("weight", 0.0) or 0.0)
        if weight <= 0:
            continue

        channel_req = req.model_copy(
            update={
                "query": rewritten_query,
                "sparse_query": str(spec.get("sparse_query") or rewritten_query),
                "dense_query": str(spec.get("dense_query") or rewritten_query),
                "filters": spec.get("filters", req.filters),
                "top_k": _INTENT_CHANNEL_CANDIDATE_LIMIT,
            }
        )
        channel_resp = run_search_hybrid_doc_semantic_units(channel_req, db)
        for rank, item in enumerate(channel_resp.results, start=1):
            arxiv_id = str(item.arxiv_id)
            if not arxiv_id:
                continue
            if arxiv_id not in merged_by_id:
                merged_by_id[arxiv_id] = {
                    "arxiv_id": item.arxiv_id,
                    "title": item.title,
                    "abstract": item.abstract,
                    "venue": item.venue,
                    "citation_count": item.citation_count,
                    "_rrf": 0.0,
                }
            merged_by_id[arxiv_id]["_rrf"] += weight / float(_RRF_K + rank)

    ranked = sorted(merged_by_id.values(), key=lambda row: float(row.get("_rrf", 0.0)), reverse=True)
    items = [
        SearchResultItem(
            arxiv_id=row["arxiv_id"],
            title=row.get("title"),
            abstract=row.get("abstract"),
            venue=row.get("venue"),
            citation_count=row.get("citation_count"),
            score=round(float(row.get("_rrf", 0.0)), 4),
        )
        for row in ranked[: req.top_k]
    ]
    return SearchResponse(
        query=req.query,
        rewritten_query=rewritten_query,
        intent_queries=intent_queries,
        total=len(items),
        results=items,
    )
