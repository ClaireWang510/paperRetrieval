from __future__ import annotations

import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from scientific_resource_release.retrieval.algorithms import run_search_hybrid_doc_semantic_units_intent
from scientific_resource_release.retrieval.database import get_engine, get_session_factory, init_db
from scientific_resource_release.retrieval.intent import build_intent_retrieval_payload
from scientific_resource_release.retrieval.schemas import SearchRequest

logger = logging.getLogger(__name__)


_RERANKER_MODEL_CACHE = {}
_RERANKER_MODEL_LOCK = threading.Lock()
_LATIN_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ']+")


def _resolve_local_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _resolve_cache_dir() -> Optional[str]:
    raw = os.environ.get("BGE_MODEL_CACHE_DIR", "").strip()
    if not raw:
        return None
    path = _resolve_local_path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _resolve_reranker_source(default_model: str) -> str:
    local_raw = os.environ.get("BGE_RERANKER_MODEL_PATH", "").strip()
    if local_raw:
        local_path = _resolve_local_path(local_raw)
        if local_path.exists():
            return str(local_path)
        logger.warning("BGE_RERANKER_MODEL_PATH not found: %s", local_path)

    name = os.environ.get("BGE_RERANKER_MODEL_NAME", default_model).strip()
    return name or default_model


def _resolve_backend_preference() -> str:
    value = os.environ.get("BGE_RERANKER_BACKEND", "hf_transformers").strip().lower()
    if value in {"hf", "transformers", "hf_transformers"}:
        return "hf_transformers"
    if value in {"flag", "flagembedding"}:
        return "flagembedding"
    if value == "auto":
        return "auto"
    return "hf_transformers"


def _resolve_reranker_device() -> str:
    value = os.environ.get("BGE_RERANKER_DEVICE", "auto").strip().lower()
    if value in {"cpu", "cuda", "auto"}:
        return value
    return "auto"


def _sigmoid(x_value: float) -> float:
    if x_value >= 0:
        z_value = math.exp(-x_value)
        return 1.0 / (1.0 + z_value)
    z_value = math.exp(x_value)
    return z_value / (1.0 + z_value)


def _normalize_minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [_sigmoid(v) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


class _HFPairReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 32, cache_dir: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        self.batch_size = batch_size
        self._torch = torch
        self.device_preference = _resolve_reranker_device()
        local_files_only = Path(model_name).is_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.device = self._select_device()
        self._move_model_to_device(self.device)
        self.model.eval()

    def _select_device(self) -> str:
        torch = self._torch
        if self.device_preference == "cpu":
            return "cpu"
        if self.device_preference == "cuda":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _move_model_to_device(self, device: str) -> None:
        try:
            self.model.to(device)
            self.device = device
        except RuntimeError as exc:
            if device == "cuda" and "out of memory" in str(exc).lower():
                logger.warning("Reranker CUDA OOM on load, falling back to CPU: %s", exc)
                self._fallback_to_cpu()
                return
            raise

    def _fallback_to_cpu(self) -> None:
        torch = self._torch
        self.model.to("cpu")
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def predict(self, pairs: List[List[str]]) -> List[float]:
        if not pairs:
            return []

        scores: List[float] = []
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                queries = [item[0] for item in batch]
                docs = [item[1] for item in batch]
                encoded = self.tokenizer(
                    queries,
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                try:
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    logits = self.model(**encoded).logits
                except RuntimeError as exc:
                    if self.device == "cuda" and "out of memory" in str(exc).lower():
                        logger.warning("Reranker CUDA OOM on predict, retrying on CPU: %s", exc)
                        self._fallback_to_cpu()
                        encoded = {key: value.to(self.device) for key, value in encoded.items()}
                        logits = self.model(**encoded).logits
                    else:
                        raise
                if logits.ndim == 2 and logits.size(-1) == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.ndim == 2:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.reshape(-1)
                scores.extend(float(v) for v in batch_scores.detach().cpu().tolist())
        return scores


def _lexical_overlap_score(query: str, text_value: str) -> float:
    q_set = {token for token in query.lower().split() if token}
    d_set = {token for token in text_value.lower().split() if token}
    if not q_set or not d_set:
        return 0.0
    return len(q_set.intersection(d_set)) / max(len(q_set), 1)


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model_source = _resolve_reranker_source(model_name)
        self.cache_dir = _resolve_cache_dir()
        self.backend_preference = _resolve_backend_preference()
        self._mode = "lexical"
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        cache_key = self.model_source
        with _RERANKER_MODEL_LOCK:
            cached = _RERANKER_MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._mode, self._model = cached
            return

        def _try_hf() -> bool:
            try:
                self._model = _HFPairReranker(self.model_source, cache_dir=self.cache_dir)
                self._mode = "hf_transformers"
                return True
            except Exception as exc:
                logger.warning("HF reranker unavailable: %s", exc)
                return False

        def _try_flag() -> bool:
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_source, use_fp16=False)
                self._mode = "flagembedding"
                return True
            except Exception as exc:
                logger.warning("FlagEmbedding reranker unavailable: %s", exc)
                return False

        if self.backend_preference == "hf_transformers":
            loaded = _try_hf() or _try_flag()
        elif self.backend_preference == "flagembedding":
            loaded = _try_flag() or _try_hf()
        else:
            loaded = _try_flag() or _try_hf()

        if not loaded:
            self._mode = "lexical"
            self._model = None

        with _RERANKER_MODEL_LOCK:
            _RERANKER_MODEL_CACHE[cache_key] = (self._mode, self._model)

    def score(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        if self._mode == "flagembedding" and self._model is not None:
            raw = self._model.compute_score(pairs)
            raw_scores = [float(raw)] if isinstance(raw, (float, int)) else [float(v) for v in raw]
            return _normalize_minmax(raw_scores)

        if self._mode == "hf_transformers" and self._model is not None:
            raw_scores = [float(v) for v in self._model.predict(pairs)]
            return _normalize_minmax(raw_scores)

        return _normalize_minmax([_lexical_overlap_score(query, doc) for doc in documents])


def _sanitize_text(text_value: str) -> str:
    return (text_value or "").encode("utf-8", errors="backslashreplace").decode("utf-8")


def _infer_language(query: str) -> str:
    text_value = (query or "").strip()
    if not text_value:
        return "zh"
    if re.search(r"[\u4e00-\u9fff]", text_value):
        return "zh"
    tokens = [t.lower() for t in _LATIN_TOKEN_RE.findall(text_value)]
    if not tokens:
        return "en"
    english_hits = len(set(tokens).intersection({"the", "and", "for", "with", "in", "on", "of", "to"}))
    return "en" if english_hits >= 0 else "en"


def _language_name(lang: str) -> str:
    return {"zh": "中文", "en": "English"}.get(lang, "English")


def _ensure_nltk_tokenizers() -> None:
    import nltk

    for resource, locator in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(locator)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                logger.warning("Failed to download NLTK resource %s: %s", resource, exc)


def _split_answer_sentences(answer_text: str, lang: str) -> List[str]:
    text_value = str(answer_text or "").strip()
    if not text_value:
        return []

    try:
        if lang == "en":
            import nltk

            _ensure_nltk_tokenizers()
            parts = nltk.tokenize.sent_tokenize(text_value)
        else:
            from nltk.tokenize import RegexpTokenizer

            tokenizer = RegexpTokenizer(r"[^。！？!?]+[。！？!?]?")
            parts = tokenizer.tokenize(text_value)
    except Exception as exc:
        logger.warning("Sentence tokenization fallback used: %s", exc)
        if lang == "en":
            parts = re.split(r"(?<=[.!?])\s+", text_value)
        else:
            parts = re.findall(r"[^。！？!?]+[。！？!?]?", text_value)

    return [part.strip() for part in parts if str(part).strip()]


def _call_llm(messages: List[Dict[str, str]], use_json_mode: bool = False) -> str:
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return _sanitize_text(resp.choices[0].message.content or "")


@dataclass(frozen=True)
class ScoreWeights:
    rerank_weight: float = 0.65
    retrieval_weight: float = 0.30
    venue_prior_weight: float = 0.05


def _venue_prior_score(venue: Optional[str]) -> float:
    text_value = (venue or "").strip().lower()
    if not text_value:
        return 0.0
    if "arxiv" in text_value:
        return 0.0
    return 0.2


def _build_rerank_document(item: Dict[str, Any]) -> str:
    return f"Title: {item.get('title') or ''}\nAbstract: {item.get('abstract') or ''}".strip()


def _extract_candidate_semantic_units(item: Dict[str, Any], max_units: int) -> List[str]:
    values = item.get("semantic_units")
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        if isinstance(value, dict):
            content = str(value.get("content") or "").strip()
        else:
            content = str(value or "").strip()
        if not content:
            continue
        out.append(content)
        if len(out) >= max_units:
            break
    return out


def _fetch_semantic_units_for_candidates(db, candidates: List[Dict[str, Any]], shortlist_idx: List[int], max_units: int) -> Dict[str, List[str]]:
    arxiv_ids: List[str] = []
    seen = set()
    for idx in shortlist_idx:
        aid = str(candidates[idx].get("arxiv_id") or "").strip()
        if aid and aid not in seen:
            seen.add(aid)
            arxiv_ids.append(aid)
    if not arxiv_ids:
        return {}

    sql = text(
        """
        SELECT arxiv_id, content
        FROM (
            SELECT arxiv_id, content,
                   ROW_NUMBER() OVER (PARTITION BY arxiv_id ORDER BY length(content) DESC) AS rn
            FROM semantic_chunks
            WHERE arxiv_id = ANY(:arxiv_ids)
              AND content IS NOT NULL
              AND length(content) > 0
        ) t
        WHERE rn <= :max_units
        ORDER BY arxiv_id, rn
        """
    )
    rows = db.execute(sql, {"arxiv_ids": arxiv_ids, "max_units": max_units}).fetchall()

    out: Dict[str, List[str]] = {}
    for arxiv_id, content in rows:
        aid = str(arxiv_id or "").strip()
        txt = str(content or "").strip()
        if not aid or not txt:
            continue
        out.setdefault(aid, []).append(txt)
    return out


def _build_intent_queries(base_query: str, intent_decomposer: str) -> List[str]:
    payload = build_intent_retrieval_payload(
        base_query=base_query,
        intent_decomposer=intent_decomposer,
        intent_channels=["task"],
        base_filters=None,
    )
    out = []
    for query in [payload.get("rewritten_query"), *list(payload.get("intent_queries") or [])]:
        value = str(query or "").strip()
        if value and value.lower() not in {x.lower() for x in out}:
            out.append(value)
    return out or [base_query]


def _fetch_top_semantic_units(db, arxiv_id: str, query: str, reranker: BGEReranker, top_n: int = 3) -> List[Dict[str, str]]:
    sql = text(
        """
        SELECT role, content
        FROM semantic_chunks
        WHERE arxiv_id = :arxiv_id
          AND content IS NOT NULL
          AND length(content) > 0
        ORDER BY length(content) DESC
        LIMIT 12
        """
    )
    rows = db.execute(sql, {"arxiv_id": arxiv_id}).fetchall()
    units = [{"role": str(row[0] or "other"), "content": str(row[1] or "")} for row in rows if str(row[1] or "").strip()]
    if not units:
        return []

    scores = reranker.score(query=query, documents=[u["content"] for u in units])
    for idx, unit in enumerate(units):
        unit["_score"] = scores[idx] if idx < len(scores) else 0.0
    units.sort(key=lambda item: float(item.get("_score", 0.0)), reverse=True)
    out = []
    for item in units[:top_n]:
        out.append({"role": item["role"], "content": item["content"]})
    return out


def _generate_reason(query: str, title: Optional[str], abstract: Optional[str], semantic_units: List[Dict[str, str]]) -> str:
    if not semantic_units:
        return ""

    evidence = "\n".join(f"- [{u.get('role') or 'other'}] {u.get('content') or ''}" for u in semantic_units)
    lang_name = _language_name(_infer_language(query))
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic search assistant. Write 2-3 sentences recommendation reason grounded in evidence. "
                f"Output language must be {lang_name}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{query}\n\n"
                f"Title:\n{title or ''}\n\n"
                f"Abstract:\n{abstract or ''}\n\n"
                f"Evidence:\n{evidence}\n\n"
                "Output recommendation reason only."
            ),
        },
    ]
    try:
        reason = _call_llm(messages)
        if reason.strip():
            return reason.strip()
    except Exception as exc:
        logger.warning("Reason generation fallback used: %s", exc)

    snippet = semantic_units[0].get("content", "")[:180]
    if _infer_language(query) == "en":
        return f"This paper is relevant to \"{query}\" and provides direct evidence: {snippet}"
    return f"该论文与“{query}”直接相关，证据要点包括：{snippet}"


def _synthesize_answer(query: str, rewritten_query: str, results: List[Dict[str, Any]]) -> Optional[str]:
    evidence_lines = []
    for idx, item in enumerate(results[:10], start=1):
        for unit in (item.get("semantic_units") or [])[:3]:
            content = str(unit.get("content") or "").strip()
            role = str(unit.get("role") or "other").strip()
            if content:
                evidence_lines.append(f"[{idx}] {item.get('title') or 'Untitled'} | {role}: {content}")

    if not evidence_lines:
        return None

    lang_name = _language_name(_infer_language(query))
    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic retrieval QA assistant. Answer briefly in 3-5 sentences, grounded only in evidence. "
                f"Output language must be {lang_name}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original query: {query}\n"
                f"Rewritten query: {rewritten_query or query}\n\n"
                "Evidence:\n"
                + "\n".join(evidence_lines)
            ),
        },
    ]
    try:
        answer = _call_llm(messages)
        if answer.strip():
            return answer.strip()
    except Exception as exc:
        logger.warning("Answer synthesis fallback used: %s", exc)

    top_snippets = "; ".join(line.split(":", 1)[-1].strip()[:120] for line in evidence_lines[:3])
    if _infer_language(query) == "en":
        return f"Based on retrieved evidence, the key answer for \"{query}\" is: {top_snippets}"
    return f"基于检索证据，“{query}”的核心结论是：{top_snippets}"


def _attach_answer_references(answer_text: Optional[str], results: List[Dict[str, Any]], reranker: BGEReranker) -> Dict[str, Any]:
    clean_answer = str(answer_text or "").strip()
    if not clean_answer:
        return {"answer_markdown": "", "references": []}

    evidence_entries: List[Dict[str, Any]] = []
    for paper_index, item in enumerate(results[:10], start=1):
        for unit in (item.get("semantic_units") or [])[:3]:
            content = str(unit.get("content") or "").strip()
            if not content:
                continue
            evidence_entries.append(
                {
                    "paper_index": paper_index,
                    "title": item.get("title") or "无标题",
                    "arxiv_id": item.get("arxiv_id") or "-",
                    "role": unit.get("role") or "other",
                    "content": content,
                }
            )

    if not evidence_entries:
        return {"answer_markdown": clean_answer, "references": []}

    lang = _infer_language(clean_answer)
    sentences = _split_answer_sentences(clean_answer, lang)
    if not sentences:
        return {"answer_markdown": clean_answer, "references": []}

    documents = [entry["content"] for entry in evidence_entries]
    seen_papers = set()
    references: List[Dict[str, Any]] = []
    annotated_sentences: List[str] = []

    for sentence in sentences:
        scores = reranker.score(query=sentence, documents=documents)
        if not scores:
            annotated_sentences.append(sentence)
            continue

        best_idx = max(range(len(scores)), key=lambda idx: float(scores[idx]))
        best_entry = evidence_entries[best_idx]
        paper_index = int(best_entry["paper_index"])
        annotated_sentences.append(f"{sentence} [{paper_index}]")

        if paper_index not in seen_papers:
            seen_papers.add(paper_index)
            references.append(
                {
                    "paper_index": paper_index,
                    "title": best_entry["title"],
                    "arxiv_id": best_entry["arxiv_id"],
                    "role": best_entry["role"],
                    "semantic_unit": best_entry["content"],
                }
            )

    answer_markdown = " ".join(annotated_sentences) if lang == "en" else "".join(annotated_sentences)
    return {"answer_markdown": answer_markdown, "references": references}


class RetrievalKnowledgeService:
    def __init__(self, database_url: Optional[str] = None):
        self.engine = get_engine(database_url)
        init_db(self.engine)
        self.session_factory = get_session_factory(self.engine)
        self.reranker = BGEReranker()

    def search(self, request: SearchRequest) -> Dict[str, Any]:
        with self.session_factory() as db:
            response = run_search_hybrid_doc_semantic_units_intent(request, db)
        return response.model_dump()

    def search_with_knowledge(
        self,
        request: SearchRequest,
        top_k: Optional[int] = None,
        reason_top_k: int = 10,
        rerank_pool_size: Optional[int] = None,
        rerank_mode: str = "semantic_units_intent_max",
        rerank_intent_decomposer: str = "llm",
    ) -> Dict[str, Any]:
        del rerank_mode
        search_payload = self.search(request)
        candidates = [dict(item) for item in search_payload.get("results", [])]
        if not candidates:
            search_payload["answer"] = None
            return search_payload

        retrieval_scores = [float(item.get("score", 0.0) or 0.0) for item in candidates]
        retrieval_norm = _normalize_minmax(retrieval_scores)
        for idx, item in enumerate(candidates):
            item["retrieval_score_norm"] = retrieval_norm[idx] if idx < len(retrieval_norm) else 0.0

        limit = top_k or request.top_k
        pool_size = rerank_pool_size if rerank_pool_size is not None else max(limit * 4, 40)
        pool_size = min(max(pool_size, limit), len(candidates))

        ranked_idx = sorted(range(len(candidates)), key=lambda i: retrieval_scores[i], reverse=True)
        shortlist_idx = ranked_idx[:pool_size]

        intent_queries = [str(v).strip() for v in (search_payload.get("intent_queries") or []) if str(v).strip()]
        if not intent_queries:
            intent_queries = _build_intent_queries(request.query, rerank_intent_decomposer)

        with self.session_factory() as db:
            units_map = _fetch_semantic_units_for_candidates(db, candidates, shortlist_idx, max_units=24)

        rerank_scores = [0.0 for _ in candidates]
        for idx in shortlist_idx:
            candidate = candidates[idx]
            aid = str(candidate.get("arxiv_id") or "")
            units = _extract_candidate_semantic_units(candidate, max_units=24) or units_map.get(aid, [])
            if not units:
                units = [_build_rerank_document(candidate)]

            per_intent = []
            for intent_query in intent_queries:
                scores = self.reranker.score(intent_query, units)
                per_intent.append(max(scores) if scores else 0.0)
            rerank_scores[idx] = sum(per_intent) / len(per_intent) if per_intent else 0.0

        weights = ScoreWeights()
        enriched = []
        for idx, item in enumerate(candidates):
            rerank_score = rerank_scores[idx]
            retrieval_score = item.get("retrieval_score_norm", 0.0)
            final_score = (
                weights.rerank_weight * rerank_score
                + weights.retrieval_weight * retrieval_score
                + weights.venue_prior_weight * _venue_prior_score(item.get("venue"))
            )
            out = dict(item)
            out["rerank_score"] = round(rerank_score, 4)
            out["final_score"] = round(final_score, 4)
            enriched.append(out)

        enriched.sort(key=lambda row: float(row.get("final_score", 0.0)), reverse=True)
        final_results = enriched[:limit]

        with self.session_factory() as db:
            for idx, item in enumerate(final_results):
                if idx >= reason_top_k:
                    item["recommendation_reason"] = ""
                    continue
                semantic_units = _fetch_top_semantic_units(
                    db,
                    arxiv_id=str(item.get("arxiv_id") or ""),
                    query=request.query,
                    reranker=self.reranker,
                    top_n=3,
                )
                item["semantic_units"] = semantic_units
                item["recommendation_reason"] = _generate_reason(
                    query=request.query,
                    title=item.get("title"),
                    abstract=item.get("abstract"),
                    semantic_units=semantic_units,
                )

        answer = _synthesize_answer(
            query=request.query,
            rewritten_query=str(search_payload.get("rewritten_query") or request.query),
            results=final_results,
        )
        answer_bundle = _attach_answer_references(answer, final_results, self.reranker)

        search_payload["results"] = final_results
        search_payload["total"] = len(final_results)
        search_payload["answer"] = answer_bundle.get("answer_markdown") or answer
        search_payload["answer_references"] = answer_bundle.get("references") or []
        return search_payload
