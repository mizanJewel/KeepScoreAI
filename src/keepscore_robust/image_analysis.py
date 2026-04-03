from __future__ import annotations

import base64
import json
import re
from io import BytesIO

from PIL import Image

from keepscore_robust.llm import ollama_chat


COLOR_SWATCHES = {
    "black": (30, 30, 30),
    "white": (235, 235, 235),
    "grey": (140, 140, 140),
    "blue": (60, 90, 180),
    "red": (180, 60, 60),
    "green": (70, 140, 80),
    "beige": (198, 178, 134),
    "navy": (35, 52, 102),
    "purple": (125, 90, 170),
    "orange": (220, 130, 50),
}
CATEGORY_HINTS = {
    "running": {"run", "runner", "running", "trainer"},
    "walking": {"walk", "walking"},
    "trail": {"trail", "hiking", "outdoor"},
    "lifestyle": {"casual", "lifestyle", "retro", "fashion", "street"},
}


def _closest_color(rgb: tuple[int, int, int]) -> str:
    def distance(swatch: tuple[int, int, int]) -> int:
        return sum((a - b) ** 2 for a, b in zip(rgb, swatch))

    return min(COLOR_SWATCHES.items(), key=lambda item: distance(item[1]))[0]


def _extract_image_features(image_bytes: bytes, filename: str) -> dict:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        small = rgb.resize((64, 64))
        pixels = list(small.getdata())
        avg_rgb = tuple(int(sum(channel) / len(pixels)) for channel in zip(*pixels))
        dominant = _closest_color(avg_rgb)
        width, height = rgb.size
    filename_tokens = set(re.findall(r"[a-z0-9]+", filename.lower()))
    category = None
    for label, hints in CATEGORY_HINTS.items():
        if filename_tokens.intersection(hints):
            category = label
            break
    return {
        "filename": filename,
        "dominant_color": dominant,
        "average_rgb": list(avg_rgb),
        "aspect_ratio": round(width / max(height, 1), 2),
        "filename_tokens": sorted(filename_tokens)[:12],
        "guessed_category": category or "running",
    }


def _fallback_analysis(features: dict) -> dict:
    category = features["guessed_category"]
    color = features["dominant_color"]
    description = (
        f"The uploaded shoe appears to be a {color} {category} shoe. "
        f"The visual analysis is heuristic, so the category is inferred mainly from color balance and filename hints."
    )
    related_terms = [category, color]
    if category == "running":
        related_terms.extend(["daily trainer", "soft cushioning"])
    elif category == "walking":
        related_terms.extend(["all-day comfort", "support"])
    elif category == "trail":
        related_terms.extend(["grip", "rugged"])
    else:
        related_terms.extend(["casual", "premium"])
    return {
        "description": description,
        "category": category,
        "color": color,
        "style_tags": related_terms[:4],
        "search_query": f"{color} {category} shoe",
        "related_suggestions": related_terms[:4],
        "analysis_mode": "heuristic",
    }


def analyze_uploaded_shoe_image(image_bytes: bytes, filename: str) -> dict:
    features = _extract_image_features(image_bytes, filename)
    prompt = "\n".join(
        [
            "Analyze this uploaded shoe image and respond only as compact JSON.",
            'Required keys: description, category, color, style_tags, search_query, related_suggestions.',
            'Use category from: running, walking, trail, lifestyle.',
            'Use color from: black, white, grey, blue, red, green, beige, navy, purple, orange.',
            "Keep description to one or two sentences. Keep search_query short and shopping-oriented.",
        ]
    )
    result = ollama_chat(
        [
            {"role": "system", "content": "You are a visual shoe analyst. Return strict JSON only."},
            {"role": "user", "content": prompt, "images": [base64.b64encode(image_bytes).decode("utf-8")]},
        ],
        timeout=30.0,
    )
    if result.get("ok"):
        content = result.get("content", "")
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                parsed["analysis_mode"] = "ollama_vision"
                parsed["features"] = features
                parsed["model"] = result.get("model")
                return parsed
        except json.JSONDecodeError:
            pass
    fallback = _fallback_analysis(features)
    fallback["features"] = features
    fallback["model"] = None
    fallback["error"] = result.get("error")
    return fallback
