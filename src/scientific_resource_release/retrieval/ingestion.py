from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import delete

from scientific_resource_release.retrieval.database import (
    PaperDenseEmbedding,
    PaperFulltextChunk,
    PaperMetadata,
    SemanticChunk,
    get_engine,
    get_session_factory,
    init_db,
)
from scientific_resource_release.retrieval.embedding import get_embedding_model

logger = logging.getLogger(__name__)

EXCLUDE_META_KEYS = {"sections", "figures", "tables"}
DEFAULT_FULLTEXT_CHUNK_SIZE = 256


def _parse_metadata(meta_path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", meta_path, exc)
        return None

    clean = {k: v for k, v in payload.items() if k not in EXCLUDE_META_KEYS}
    if "fieldsOfStudy" in clean and "fields_of_study" not in clean:
        clean["fields_of_study"] = clean.pop("fieldsOfStudy") or []
    if "citationCount" in clean and "citation_count" not in clean:
        clean["citation_count"] = clean.get("citationCount")
    if not clean.get("arxiv_id"):
        clean["arxiv_id"] = meta_path.parent.name
    return clean


def _metadata_row(meta: Dict[str, Any]) -> Dict[str, Any]:
    published = meta.get("published_date")
    if isinstance(published, str) and published:
        try:
            published = datetime.strptime(published[:10], "%Y-%m-%d").date()
        except ValueError:
            published = None
    else:
        published = None

    fields = meta.get("fields_of_study") or []
    if isinstance(fields, str):
        fields = [fields]

    return {
        "arxiv_id": meta.get("arxiv_id", ""),
        "title": meta.get("title") or "",
        "abstract": meta.get("abstract") or "",
        "authors": meta.get("authors") or [],
        "published_date": published,
        "fields_of_study": fields,
        "venue": meta.get("venue"),
        "citation_count": meta.get("citation_count"),
    }


def _parse_semantic_units(units_path: Path) -> List[Dict[str, Any]]:
    if not units_path.exists():
        return []
    try:
        payload = json.loads(units_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    values = payload.get("semantic_units") or []
    return [v for v in values if isinstance(v, dict)]


def _iter_section_nodes(sections: List[Dict[str, Any]], parent: Optional[List[str]] = None) -> List[Dict[str, str]]:
    parent = parent or []
    blocks: List[Dict[str, str]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "").strip()
        content = str(section.get("content") or "").strip()
        path = list(parent)
        if title:
            path.append(title)
        if title or content:
            text_value = "\n".join([part for part in [title, content] if part]).strip()
            if text_value:
                blocks.append({
                    "section_path": " > ".join(path) if path else "",
                    "section_title": title,
                    "text": text_value,
                })
        children = section.get("children") or []
        if isinstance(children, list) and children:
            blocks.extend(_iter_section_nodes(children, path))
    return blocks


def _build_fulltext_chunks(meta_path: Path, arxiv_id: str, chunk_size: int) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    sections = payload.get("sections") or []
    if not isinstance(sections, list) or not sections:
        return []

    blocks = _iter_section_nodes(sections)
    out: List[Dict[str, Any]] = []
    chunk_index = 0
    for block in blocks:
        tokens = str(block.get("text") or "").split()
        if not tokens:
            continue
        for offset in range(0, len(tokens), chunk_size):
            piece = " ".join(tokens[offset : offset + chunk_size]).strip()
            if not piece:
                continue
            out.append(
                {
                    "chunk_id": f"{arxiv_id}_fulltext_{chunk_index:06d}",
                    "arxiv_id": arxiv_id,
                    "chunk_index": chunk_index,
                    "section_path": block.get("section_path") or None,
                    "section_title": block.get("section_title") or None,
                    "content": piece,
                }
            )
            chunk_index += 1
    return out


def ingest_one(
    data_root: Path,
    arxiv_id: str,
    session,
    skip_existing_semantic_units: bool = False,
    skip_existing_title_abstract_embedding: bool = False,
    skip_existing_fulltext_chunks: bool = False,
    fulltext_chunk_size: int = DEFAULT_FULLTEXT_CHUNK_SIZE,
) -> Tuple[int, int, int, int]:
    paper_dir = data_root / arxiv_id
    meta_path = paper_dir / "metadata.json"
    units_path = paper_dir / "semantic_units.json"
    if not meta_path.exists():
        logger.warning("[%s] metadata.json missing", arxiv_id)
        return 0, 0, 0, 0

    meta = _parse_metadata(meta_path)
    if not meta:
        return 0, 0, 0, 0

    aid = str(meta.get("arxiv_id") or arxiv_id)
    row = _metadata_row(meta)

    existing = session.get(PaperMetadata, row["arxiv_id"])
    if existing:
        for key, value in row.items():
            setattr(existing, key, value)
        meta_inserted = 0
    else:
        session.add(PaperMetadata(**row))
        meta_inserted = 1
    session.flush()

    model = get_embedding_model()
    doc_text = f"{row.get('title') or ''} {row.get('abstract') or ''}".strip()

    doc_upserted = 0
    has_doc = session.get(PaperDenseEmbedding, aid) is not None
    if not (skip_existing_title_abstract_embedding and has_doc):
        doc_emb = model.encode(doc_text, normalize_embeddings=True)
        emb_value = doc_emb.tolist() if hasattr(doc_emb, "tolist") else list(doc_emb)
        existing_emb = session.get(PaperDenseEmbedding, aid)
        if existing_emb:
            existing_emb.embedding = emb_value
        else:
            session.add(PaperDenseEmbedding(arxiv_id=aid, embedding=emb_value))
        doc_upserted = 1

    fulltext_inserted = 0
    has_fulltext = (
        session.query(PaperFulltextChunk.chunk_id).filter(PaperFulltextChunk.arxiv_id == aid).first() is not None
    )
    if not (skip_existing_fulltext_chunks and has_fulltext):
        session.execute(delete(PaperFulltextChunk).where(PaperFulltextChunk.arxiv_id == aid))
        fulltext_payload = _build_fulltext_chunks(meta_path, aid, fulltext_chunk_size)
        if fulltext_payload:
            fulltext_emb = model.encode([x["content"] for x in fulltext_payload], normalize_embeddings=True)
            for idx, record in enumerate(fulltext_payload):
                emb_value = fulltext_emb[idx].tolist() if hasattr(fulltext_emb[idx], "tolist") else list(fulltext_emb[idx])
                record["embedding"] = emb_value
                session.add(PaperFulltextChunk(**record))
                fulltext_inserted += 1

    chunks_inserted = 0
    has_units = session.query(SemanticChunk.chunk_id).filter(SemanticChunk.arxiv_id == aid).first() is not None
    if skip_existing_semantic_units and has_units:
        session.commit()
        return meta_inserted, doc_upserted, 0, fulltext_inserted

    units = _parse_semantic_units(units_path)
    if not units:
        session.commit()
        return meta_inserted, doc_upserted, 0, fulltext_inserted

    session.execute(delete(SemanticChunk).where(SemanticChunk.arxiv_id == aid))
    contents = [str(unit.get("content") or "") for unit in units]
    embeddings = model.encode(contents, normalize_embeddings=True)

    for idx, unit in enumerate(units):
        chunk_id = unit.get("id") or f"{aid}_{idx:04d}"
        emb_value = embeddings[idx].tolist() if hasattr(embeddings[idx], "tolist") else list(embeddings[idx])
        keywords = unit.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        session.add(
            SemanticChunk(
                chunk_id=str(chunk_id),
                arxiv_id=aid,
                role=(unit.get("semantic_role") or unit.get("role") or "other"),
                content=str(unit.get("content") or ""),
                keywords=keywords,
                embedding=emb_value,
            )
        )
        chunks_inserted += 1

    session.commit()
    return meta_inserted, doc_upserted, chunks_inserted, fulltext_inserted


def run_ingestion(
    data_root: Path,
    arxiv_ids: Optional[List[str]] = None,
    skip_existing_semantic_units: bool = False,
    skip_existing_title_abstract_embedding: bool = False,
    skip_existing_fulltext_chunks: bool = False,
    fulltext_chunk_size: int = DEFAULT_FULLTEXT_CHUNK_SIZE,
) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    if arxiv_ids is None:
        arxiv_ids = [
            path.name
            for path in sorted(data_root.iterdir())
            if path.is_dir() and (path / "metadata.json").exists()
        ]

    engine = get_engine()
    init_db(engine)
    factory = get_session_factory(engine)

    total_meta = 0
    total_doc = 0
    total_chunks = 0
    total_fulltext = 0
    failed: List[str] = []

    for idx, arxiv_id in enumerate(arxiv_ids):
        try:
            with factory() as session:
                m, d, c, f = ingest_one(
                    data_root=data_root,
                    arxiv_id=arxiv_id,
                    session=session,
                    skip_existing_semantic_units=skip_existing_semantic_units,
                    skip_existing_title_abstract_embedding=skip_existing_title_abstract_embedding,
                    skip_existing_fulltext_chunks=skip_existing_fulltext_chunks,
                    fulltext_chunk_size=fulltext_chunk_size,
                )
            total_meta += m
            total_doc += d
            total_chunks += c
            total_fulltext += f
            logger.info(
                "[%d/%d] %s metadata=%d doc=%d semantic_chunks=%d fulltext_chunks=%d",
                idx + 1,
                len(arxiv_ids),
                arxiv_id,
                m,
                d,
                c,
                f,
            )
        except Exception as exc:
            logger.exception("[%s] ingestion failed: %s", arxiv_id, exc)
            failed.append(arxiv_id)

    logger.info(
        "Done. papers=%d metadata=%d doc_embeddings=%d semantic_chunks=%d fulltext_chunks=%d",
        len(arxiv_ids),
        total_meta,
        total_doc,
        total_chunks,
        total_fulltext,
    )
    if failed:
        logger.warning("Failed ids: %s", failed)
