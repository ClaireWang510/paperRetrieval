from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from scientific_resource_release.retrieval.schemas import SearchFilters

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


def _tokenize(text_value: str) -> List[str]:
    if not text_value:
        return []
    return [tok for tok in _TOKEN_RE.split(text_value.lower()) if tok]


def _clip_keywords(values: List[str], max_keywords: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text_value = (value or "").strip()
        key = text_value.lower()
        if not text_value or key in seen:
            continue
        seen.add(key)
        out.append(text_value)
        if len(out) >= max_keywords:
            break
    return out


def _as_filters(payload: Any) -> Optional[SearchFilters]:
    if not payload:
        return None
    try:
        if isinstance(payload, SearchFilters):
            return payload
        if isinstance(payload, dict):
            return SearchFilters(**payload)
    except Exception:
        return None
    return None


def _fallback_task_specs(base_query: str, base_filters: Optional[SearchFilters]) -> List[Dict[str, Any]]:
    return [
        {
            "sparse_query": " ".join(_clip_keywords(_tokenize(base_query))) or base_query,
            "dense_query": base_query,
            "filters": base_filters,
        }
    ]


def _decompose_with_llm(base_query: str) -> Dict[str, Any]:
    api_key = os.environ.get("INTENT_LLM_API_KEY") or os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("INTENT_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1"
    model = os.environ.get("INTENT_LLM_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    if not api_key:
        raise RuntimeError("Missing INTENT_LLM_API_KEY or LLM_API_KEY")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = {
        "rewritten_query": "english rewritten query",
        "tasks": [
            {"dense_query": "...", "sparse_keywords": ["...", "..."], "filters": {"venues": ["..."], "roles": ["..."]}}
        ],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite and decompose academic search queries. Return JSON only with keys rewritten_query and tasks. "
                "tasks is a list with dense_query, sparse_keywords, optional filters. Keep tasks concise."
            ),
        },
        {
            "role": "user",
            "content": f"Query: {base_query}\nOutput schema example: {json.dumps(prompt)}",
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def build_intent_retrieval_payload(
    *,
    base_query: str,
    intent_decomposer: Optional[str],
    intent_channels: Optional[List[str]],
    base_filters: Optional[SearchFilters],
) -> Dict[str, Any]:
    del intent_channels

    decomposer = (intent_decomposer or "llm").strip().lower()
    rewritten_query = base_query
    task_specs: List[Dict[str, Any]] = []

    try:
        if decomposer in {"llm", "relational"}:
            llm_payload = _decompose_with_llm(base_query)
            rewritten_query = str(llm_payload.get("rewritten_query") or "").strip() or base_query
            tasks = list(llm_payload.get("tasks") or [])
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                dense_query = str(task.get("dense_query") or task.get("sub_intention") or rewritten_query).strip()
                sparse_keywords = task.get("sparse_keywords")
                if isinstance(sparse_keywords, str):
                    sparse_values = [sparse_keywords]
                else:
                    sparse_values = [str(v) for v in (sparse_keywords or [])]
                sparse_query = " ".join(_clip_keywords(sparse_values))
                task_specs.append(
                    {
                        "sparse_query": sparse_query or rewritten_query,
                        "dense_query": dense_query or rewritten_query,
                        "filters": _as_filters(task.get("filters")) or base_filters,
                    }
                )
        elif decomposer == "none":
            task_specs = _fallback_task_specs(base_query, base_filters)
        else:
            tokens = _clip_keywords(_tokenize(base_query), 3)
            task_specs = [
                {
                    "sparse_query": " ".join(tokens) or base_query,
                    "dense_query": base_query,
                    "filters": base_filters,
                }
            ]
    except Exception as exc:
        logger.warning("Intent decomposition failed, fallback to lexical: %s", exc)
        task_specs = _fallback_task_specs(base_query, base_filters)

    if not task_specs:
        task_specs = _fallback_task_specs(rewritten_query, base_filters)

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for spec in task_specs:
        key = (
            str(spec.get("sparse_query") or "").strip().lower(),
            str(spec.get("dense_query") or "").strip().lower(),
            str(spec.get("filters") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)

    intent_queries = []
    for spec in deduped:
        dense = str(spec.get("dense_query") or "").strip()
        if dense and dense.lower() not in {q.lower() for q in intent_queries}:
            intent_queries.append(dense)

    return {
        "rewritten_query": rewritten_query,
        "task_specs": deduped,
        "intent_queries": intent_queries,
    }
