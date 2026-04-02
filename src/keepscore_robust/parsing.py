from __future__ import annotations

import re

from keepscore_robust.models import ParsedTurn


CATEGORY_MAP = {
    "running": "running",
    "runner": "running",
    "walking": "walking",
    "walk": "walking",
    "trail": "trail",
    "hiking": "trail",
    "lifestyle": "lifestyle",
    "casual": "lifestyle",
}
COLORS = {"black", "white", "grey", "gray", "blue", "red", "green", "beige", "navy", "purple", "orange"}
SHELVES = {
    "recommended": "Recommended Matches",
    "trending": "Trending Shoes",
    "new launch": "New Launch",
    "new launches": "New Launch",
    "high keepscore": "High KeepScore",
}
GENDER_REGEX = [
    ("women", re.compile(r"(?:\bwomen\b|\bwoman\b|\bwomen['’]s\b|\bwomens\b|\bladies\b|\bfemale\b|for her)", re.I)),
    ("men", re.compile(r"(?:\bmen\b|\bman\b|\bmen['’]s\b|\bmens\b|\bmale\b|for him)", re.I)),
    ("unisex", re.compile(r"(?:\bunisex\b|all gender|all-gender)", re.I)),
]


def parse_turn(text: str) -> ParsedTurn:
    raw = text.strip()
    lower = raw.lower()
    result = ParsedTurn(raw_text=raw)

    budget_match = re.search(r"(?:under|below|less than|budget)\s*\$?\s*(\d+(?:\.\d+)?)", lower)
    if budget_match:
        result.budget_max = float(budget_match.group(1))

    for token, canonical in CATEGORY_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", lower):
            result.category = canonical
            break

    for gender, pattern in GENDER_REGEX:
        if pattern.search(raw):
            result.gender = gender
            break

    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", lower):
            result.color = "grey" if color == "gray" else color
            break

    if "wide" in lower:
        result.width = "wide"

    if any(term in lower for term in ["soft", "softer", "plush", "cushion", "cushioned"]):
        result.softness = 1.0
    if any(term in lower for term in ["premium", "nicer", "high end", "better material"]):
        result.premium = 1.0
    if any(term in lower for term in ["light", "lighter", "lightweight"]):
        result.lightweight = 1.0
    if any(term in lower for term in ["support", "stable", "stability", "structured"]):
        result.support = 1.0
    if "waterproof" in lower or "water resistant" in lower:
        result.waterproof = True

    if any(term in lower for term in ["replace this", "not this one", "different one", "other one", "another one"]):
        result.reject_current = True
    if "similar" in lower:
        result.wants_similar = True

    for phrase, shelf_name in SHELVES.items():
        if phrase in lower:
            result.preferred_shelves.append(shelf_name)

    return result
