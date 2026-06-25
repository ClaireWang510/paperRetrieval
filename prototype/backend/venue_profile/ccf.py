from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_VALID_TIERS = ("A", "B", "C")
_TIER_RANK = {"A": 3, "B": 2, "C": 1}


def _normalize_text(text: str) -> str:
    value = (text or "").strip().lower()
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=1)
def _load_profile() -> Dict[str, Any]:
    profile_path = Path(__file__).resolve().parent / "data" / "ccf_venues.json"
    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_alias_index() -> List[Tuple[str, Dict[str, Any]]]:
    profile = _load_profile()
    entries = profile.get("venues") or []
    index: List[Tuple[str, Dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        aliases = list(entry.get("aliases") or [])
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            aliases.append(name)
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            norm_alias = _normalize_text(alias)
            if norm_alias:
                index.append((norm_alias, entry))
    index.sort(key=lambda x: len(x[0]), reverse=True)
    return index


def _find_best_entry(venue_text: str) -> Optional[Dict[str, Any]]:
    norm = _normalize_text(venue_text)
    if not norm:
        return None

    best: Optional[Dict[str, Any]] = None
    best_len = -1
    for alias, entry in _build_alias_index():
        if alias == norm or alias in norm or norm in alias:
            if len(alias) > best_len:
                best = entry
                best_len = len(alias)
    return best


def resolve_venue_tier(venue_text: Optional[str]) -> Optional[str]:
    if not venue_text:
        return None
    entry = _find_best_entry(venue_text)
    if not entry:
        return None
    tier = str(entry.get("tier") or "").upper().strip()
    return tier if tier in _VALID_TIERS else None


def expand_venue_aliases(venue_expr: Optional[str]) -> List[str]:
    if not venue_expr:
        return []
    tokens = [
        t.strip()
        for t in re.split(r"[|,;/，；、]", venue_expr)
        if isinstance(t, str) and t.strip()
    ]
    if not tokens:
        return []

    merged: List[str] = []
    for token in tokens:
        entry = _find_best_entry(token)
        if entry:
            values = [entry.get("name")] + list(entry.get("aliases") or [])
            for v in values:
                if isinstance(v, str) and v.strip() and v.strip() not in merged:
                    merged.append(v.strip())
        elif token not in merged:
            merged.append(token)
    return merged


def _normalize_tier(tier: Optional[str]) -> Optional[str]:
    value = (tier or "").strip().upper()
    if value in _VALID_TIERS:
        return value
    return None


def _tier_meets(min_tier: str, tier: str) -> bool:
    return _TIER_RANK.get(tier, 0) >= _TIER_RANK.get(min_tier, 0)


def select_venues_by_tier(
    min_tier: str,
    venue_type: Optional[str] = None,
    limit: int = 120,
) -> List[str]:
    normalized_tier = _normalize_tier(min_tier)
    if not normalized_tier:
        return []

    kind = (venue_type or "").strip().lower()
    profile = _load_profile()
    entries = profile.get("venues") or []
    output: List[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tier = _normalize_tier(str(entry.get("tier") or ""))
        if not tier or not _tier_meets(normalized_tier, tier):
            continue
        entry_type = str(entry.get("type") or "").lower().strip()
        if kind in ("conference", "journal") and entry_type != kind:
            continue

        values = [entry.get("name")]
        values.extend(entry.get("aliases") or [])
        for v in values:
            if isinstance(v, str) and v.strip() and v.strip() not in output:
                output.append(v.strip())
                if len(output) >= limit:
                    return output
    return output


def infer_venue_requirement(text: Optional[str]) -> Dict[str, Optional[str]]:
    content = _normalize_text(text or "")
    if not content:
        return {"min_tier": None, "venue_type": None}

    venue_type: Optional[str] = None
    if re.search(r"\b(journal|periodical)\b|期刊|顶刊", content):
        venue_type = "journal"
    if re.search(r"\b(conference|conf)\b|会议|顶会", content):
        venue_type = "conference"

    min_tier: Optional[str] = None
    if re.search(r"ccf\s*a\b|a类|甲类", content):
        min_tier = "A"
    elif re.search(r"ccf\s*b\b|b类|乙类", content):
        min_tier = "B"
    elif re.search(r"ccf\s*c\b|c类|丙类", content):
        min_tier = "C"

    if re.search(r"(及以上|or above|and above|at least)", content):
        min_tier = min_tier or "B"

    if min_tier is None and re.search(r"顶会|顶刊|top tier|top conference|top journal|first tier", content):
        min_tier = "B"

    if min_tier is None and re.search(r"高水平|high quality|well[- ]?regarded", content):
        min_tier = "C"

    return {"min_tier": min_tier, "venue_type": venue_type}
