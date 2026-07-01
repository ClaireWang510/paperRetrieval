from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .llm_client import SemanticUnitLLMClient, parse_json_from_response
from .prompts import SYSTEM_PROMPT_LABEL_MERGE, build_label_merge_user_prompt
from .schemas import FactUnit


logger = logging.getLogger(__name__)

CANONICAL_LABELS = frozenset([
    "problem",
    "definition",
    "method",
    "evaluation",
    "experiment",
    "result",
    "contribution",
])

DEFAULT_LABEL_MERGE_MAP = {
    "methodology": "method",
    "approach": "method",
    "finding": "result",
    "findings": "result",
    "results": "result",
    "claim": "contribution",
    "novelty": "contribution",
    "motivation": "problem",
    "experiments": "experiment",
    "background": "definition",
}

CANONICAL_ORDER = [
    "problem",
    "definition",
    "method",
    "evaluation",
    "experiment",
    "result",
    "contribution",
    "other",
]


def _normalize_label(value: Optional[str]) -> str:
    if not value or not str(value).strip():
        return "other"
    return str(value).strip().lower()


def merge_labels_rule_based(unique_labels: List[str]) -> Dict[str, str]:
    mapping = {}
    for raw in unique_labels:
        normalized = _normalize_label(raw)
        if normalized in CANONICAL_LABELS:
            mapping[raw] = normalized
        elif normalized in DEFAULT_LABEL_MERGE_MAP:
            mapping[raw] = DEFAULT_LABEL_MERGE_MAP[normalized]
        else:
            mapping[raw] = normalized
    return mapping


def merge_labels_llm(
    unique_labels: List[str],
    llm_client: Optional[SemanticUnitLLMClient] = None,
) -> Dict[str, str]:
    labels = [str(label).strip() for label in unique_labels if str(label).strip()]
    if not labels:
        return {}
    labels = list(dict.fromkeys(labels))
    rule_based_map = merge_labels_rule_based(labels)

    if not llm_client or not llm_client.enabled:
        return rule_based_map

    canonical_like = []
    custom_labels = []
    for raw in labels:
        mapped = _normalize_label(rule_based_map.get(raw))
        if mapped in CANONICAL_LABELS:
            canonical_like.append(raw)
        else:
            custom_labels.append(raw)

    mapping = {}
    for raw in canonical_like:
        mapping[raw] = _normalize_label(rule_based_map[raw])

    if custom_labels:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LABEL_MERGE},
            {"role": "user", "content": build_label_merge_user_prompt(custom_labels)},
        ]
        try:
            raw_response, _, _ = llm_client.call_llm(
                messages,
                use_json_mode=True,
                call_name="label_merging",
            )
            data = parse_json_from_response(raw_response)
        except Exception as exc:
            logger.warning("Label merging failed: %s", exc)
            data = None

        raw_mapping = data.get("mapping") if isinstance(data, dict) else None
        if not isinstance(raw_mapping, dict):
            logger.warning("Label merging returned invalid payload; falling back to rule-based map")
            for raw in custom_labels:
                mapping[raw] = _normalize_label(rule_based_map.get(raw, raw))
        else:
            for key, value in raw_mapping.items():
                normalized_value = _normalize_label(str(value) if value else "other")
                if normalized_value in DEFAULT_LABEL_MERGE_MAP:
                    normalized_value = DEFAULT_LABEL_MERGE_MAP[normalized_value]
                mapping[str(key).strip()] = normalized_value

    for label in labels:
        if label not in mapping:
            normalized = _normalize_label(label)
            mapping[label] = DEFAULT_LABEL_MERGE_MAP.get(normalized, normalized)
    return mapping


def get_merge_map_for_facts(
    facts: List[FactUnit],
    use_llm_merge: bool = True,
    llm_client: Optional[SemanticUnitLLMClient] = None,
) -> Dict[str, str]:
    labels = list({fact.semantic_label for fact in facts if fact.semantic_label})
    if not labels:
        return {}
    if use_llm_merge:
        return merge_labels_llm(labels, llm_client=llm_client)
    return merge_labels_rule_based(labels)


def group_facts_by_merged_labels(
    facts: List[FactUnit],
    merge_map: Dict[str, str],
    unknown_label: str = "other",
) -> Dict[str, List[FactUnit]]:
    groups = {}
    for fact in facts:
        raw = (fact.semantic_label or "").strip()
        if raw and raw in merge_map:
            canonical = merge_map[raw]
        else:
            canonical = merge_map.get(_normalize_label(raw), unknown_label) if raw else unknown_label
        groups.setdefault(canonical, []).append(fact)
    return groups


def groups_to_ordered_label_cluster_pairs(
    groups: Dict[str, List[FactUnit]],
    order: Optional[List[str]] = None,
) -> List[Tuple[str, List[FactUnit]]]:
    ordered = []
    seen = set()
    for category in order or CANONICAL_ORDER:
        if category in groups and groups[category]:
            ordered.append((category, groups[category]))
            seen.add(category)
    for category in sorted(groups.keys()):
        if category not in seen and groups[category]:
            ordered.append((category, groups[category]))
    return ordered