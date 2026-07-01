from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import SemanticUnitLLMClient, parse_json_from_response
from .prompts import SYSTEM_PROMPT_FACT_UNITS, build_fact_units_user_prompt
from .schemas import FactUnit


logger = logging.getLogger(__name__)

_SKIP_SECTION_KEYWORDS = (
    "related work",
    "acknowledg",
    "references",
    "bibliograph",
)

_MAX_SECTION_CONTENT_CHARS = 12000


def _parse_fact_units_response(
    raw_response: str,
    section_title: str,
    source_type: str = "text",
    source_id: Optional[str] = None,
) -> List[FactUnit]:
    data = parse_json_from_response(raw_response)
    if not data:
        return []

    facts_data = data.get("facts", data) if isinstance(data, dict) else data
    if not isinstance(facts_data, list):
        return []

    units = []
    for item in facts_data:
        if isinstance(item, dict):
            statement = (item.get("statement") or "").strip()
            label = (item.get("semantic_label") or "").strip() or None
        elif isinstance(item, str):
            statement = item.strip()
            label = None
        else:
            continue
        if not statement:
            continue
        units.append(
            FactUnit(
                statement=statement,
                semantic_label=label,
                source_section=section_title or None,
                source_type=source_type,
                source_id=source_id,
            )
        )
    return units


def should_skip_section(section_title: str) -> bool:
    lowered = (section_title or "").strip().lower()
    return any(keyword in lowered for keyword in _SKIP_SECTION_KEYWORDS)


def _guess_label(sentence: str, section_title: str = "") -> str:
    lowered = sentence.lower()
    title_lowered = (section_title or "").lower()

    if any(token in title_lowered for token in ["method", "approach", "model", "architecture", "framework"]):
        return "method"
    if any(token in title_lowered for token in ["experiment", "evaluation", "benchmark"]):
        return "experiment"
    if any(token in title_lowered for token in ["result", "analysis", "ablation"]):
        return "result"
    if any(token in title_lowered for token in ["introduction", "motivation", "problem"]):
        return "problem"
    if any(token in title_lowered for token in ["conclusion", "discussion", "contribution"]):
        return "contribution"

    if any(token in lowered for token in ["we propose", "our method", "framework", "architecture"]):
        return "method"
    if any(token in lowered for token in ["experiment", "dataset", "benchmark", "evaluation"]):
        return "experiment"
    if any(token in lowered for token in ["result", "improve", "outperform", "achieve"]):
        return "result"
    if any(token in lowered for token in ["contribution", "main contribution"]):
        return "contribution"
    return "other"


def _heuristic_fact_units(section_title: str, section_content: str) -> List[FactUnit]:
    units = []
    for sentence in re.split(r"(?<=[.!?])\s+", section_content):
        sentence = sentence.strip()
        if len(sentence) < 40:
            continue
        units.append(
            FactUnit(
                statement=sentence,
                semantic_label=_guess_label(sentence, section_title),
                source_section=section_title or None,
                source_type="text",
                source_id=None,
            )
        )
        if len(units) >= 20:
            break
    return units


def _clip_section_content(section_title: str, section_content: str) -> str:
    content = (section_content or "").strip()
    if len(content) <= _MAX_SECTION_CONTENT_CHARS:
        return content

    clipped = content[:_MAX_SECTION_CONTENT_CHARS].rstrip()
    last_break = max(clipped.rfind("\n\n"), clipped.rfind(". "), clipped.rfind("\n"))
    if last_break >= _MAX_SECTION_CONTENT_CHARS // 2:
        clipped = clipped[:last_break].rstrip()

    logger.info(
        "Clip oversized section for fact extraction: %s (%s -> %s chars)",
        section_title,
        len(content),
        len(clipped),
    )
    return clipped


def extract_fact_units_from_section(
    section_title: str,
    section_content: str,
    figure_ids: List[str],
    table_ids: List[str],
    figure_table_semantics: Dict[str, Dict[str, Any]],
    llm_client: Optional[SemanticUnitLLMClient] = None,
    use_figure_semantics: bool = True,
) -> Tuple[List[FactUnit], int, int]:
    figure_semantics = []
    table_semantics = []

    if use_figure_semantics:
        for figure_id in figure_ids:
            item = figure_table_semantics.get(figure_id)
            if not item:
                continue
            summary_parts = []
            if item.get("semantic_summary"):
                summary_parts.append(item["semantic_summary"])
            if item.get("key_elements"):
                summary_parts.append("Key elements: %s" % "; ".join(item["key_elements"]))
            if item.get("role_in_paper"):
                summary_parts.append("Role in paper: %s" % item["role_in_paper"])
            if summary_parts:
                figure_semantics.append("Figure %s: %s" % (figure_id, ". ".join(summary_parts)))

        for table_id in table_ids:
            item = figure_table_semantics.get(table_id)
            if not item:
                continue
            summary_parts = []
            if item.get("semantic_summary"):
                summary_parts.append(item["semantic_summary"])
            if item.get("key_elements"):
                summary_parts.append("Key elements: %s" % "; ".join(item["key_elements"]))
            if item.get("role_in_paper"):
                summary_parts.append("Role in paper: %s" % item["role_in_paper"])
            if summary_parts:
                table_semantics.append("Table %s: %s" % (table_id, ". ".join(summary_parts)))

    if llm_client and llm_client.enabled:
        clipped_section_content = _clip_section_content(section_title, section_content)
        user_prompt = build_fact_units_user_prompt(
            section_title=section_title,
            section_content=clipped_section_content,
            figure_semantics=figure_semantics,
            table_semantics=table_semantics,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FACT_UNITS},
            {"role": "user", "content": user_prompt},
        ]
        try:
            raw_response, input_tokens, output_tokens = llm_client.call_llm(
                messages,
                use_json_mode=True,
                call_name="fact_extraction",
            )
            units = _parse_fact_units_response(raw_response, section_title, "text", None)
            if units:
                return units, input_tokens, output_tokens
            logger.warning("Fact extraction returned no valid facts for section %s; falling back", section_title)
        except Exception as exc:
            logger.warning("Fact extraction failed for section %s: %s", section_title, exc)

    fallback_units = _heuristic_fact_units(section_title, section_content)
    return fallback_units, 0, 0