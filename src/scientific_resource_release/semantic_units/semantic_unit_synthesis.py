from __future__ import annotations

import logging
from typing import List, Optional

from .llm_client import SemanticUnitLLMClient, parse_json_from_response
from .prompts import SYSTEM_PROMPT_SEMANTIC_UNIT, build_semantic_unit_user_prompt
from .schemas import FactUnit, SemanticUnit


logger = logging.getLogger(__name__)


def synthesize_semantic_unit(
    facts: List[FactUnit],
    arxiv_id: str,
    cluster_index: int,
    unit_id: Optional[str] = None,
    suggested_semantic_role: Optional[str] = None,
    llm_client: Optional[SemanticUnitLLMClient] = None,
    paper_title: Optional[str] = None,
) -> SemanticUnit:
    if not facts:
        return SemanticUnit(
            id=unit_id,
            arxiv_id=arxiv_id,
            semantic_role=suggested_semantic_role,
            title="(No content)",
            content="",
            keywords=[],
            source_section_hints=[],
            cluster_index=cluster_index,
            fact_count=0,
            extra={"paper_title": paper_title} if paper_title else {},
        )

    statements = [fact.statement for fact in facts]
    sections = sorted({fact.source_section for fact in facts if fact.source_section})

    if llm_client and llm_client.enabled:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SEMANTIC_UNIT},
            {
                "role": "user",
                "content": build_semantic_unit_user_prompt(
                    statements,
                    suggested_semantic_role=suggested_semantic_role,
                ),
            },
        ]
        try:
            raw_response, _, _ = llm_client.call_llm(
                messages,
                use_json_mode=True,
                call_name="semantic_unit_synthesis",
            )
            data = parse_json_from_response(raw_response)
        except Exception as exc:
            logger.warning("Semantic unit synthesis failed for cluster %s: %s", cluster_index, exc)
            data = None

        if isinstance(data, dict):
            keywords = data.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = []
            return SemanticUnit(
                id=unit_id,
                arxiv_id=arxiv_id,
                semantic_role=(data.get("semantic_role") or suggested_semantic_role or "").strip() or None,
                title=(data.get("title") or "Summary").strip(),
                content=(data.get("content") or "").strip(),
                keywords=[str(keyword).strip() for keyword in keywords if str(keyword).strip()],
                source_section_hints=sections,
                cluster_index=cluster_index,
                fact_count=len(facts),
                extra={"paper_title": paper_title} if paper_title else {},
            )

    content = " ".join(statements[:5])
    if len(statements) > 5:
        content += " ..."
    return SemanticUnit(
        id=unit_id,
        arxiv_id=arxiv_id,
        semantic_role=suggested_semantic_role,
        title="Merged facts",
        content=content,
        keywords=[],
        source_section_hints=sections,
        cluster_index=cluster_index,
        fact_count=len(facts),
        extra={"paper_title": paper_title} if paper_title else {},
    )