from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from keepscore_robust.data import DATA_ROOT
from keepscore_robust.models import ShopperProfile


MEMORY_ROOT = DATA_ROOT / "users"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _empty_profile() -> ShopperProfile:
    return ShopperProfile()


def _empty_record(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "profile": asdict(_empty_profile()),
        "chat_messages": [],
        "turn_summaries": [],
        "updated_at": None,
    }


def normalize_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", user_id.strip().lower())
    return cleaned.strip("-") or "guest"


def user_record_path(user_id: str) -> Path:
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    return MEMORY_ROOT / f"{normalize_user_id(user_id)}.json"


def load_user_record(user_id: str) -> dict:
    path = user_record_path(user_id)
    if not path.exists():
        return _empty_record(normalize_user_id(user_id))
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "user_id": data.get("user_id", normalize_user_id(user_id)),
        "profile": data.get("profile", asdict(_empty_profile())),
        "chat_messages": data.get("chat_messages", []),
        "turn_summaries": data.get("turn_summaries", []),
        "updated_at": data.get("updated_at"),
    }


def save_user_record(user_id: str, profile: ShopperProfile, chat_messages: list[dict], *, turn_summary: dict | None = None) -> dict:
    record = load_user_record(user_id)
    record["profile"] = asdict(profile)
    record["chat_messages"] = chat_messages
    if turn_summary:
        record["turn_summaries"].append(turn_summary)
        record["turn_summaries"] = record["turn_summaries"][-30:]
    record["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    path = user_record_path(user_id)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def reset_user_record(user_id: str) -> dict:
    record = _empty_record(normalize_user_id(user_id))
    path = user_record_path(user_id)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def profile_from_record(record: dict) -> ShopperProfile:
    return ShopperProfile(**record.get("profile", asdict(_empty_profile())))


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def retrieve_memory_snippets(record: dict, query: str, *, top_k: int = 4) -> list[str]:
    query_tokens = _tokenize(query)
    scored: list[tuple[float, str]] = []
    for msg in record.get("chat_messages", []):
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        tokens = _tokenize(content)
        overlap = len(query_tokens.intersection(tokens))
        score = overlap + (0.25 if msg.get("role") == "user" else 0.0)
        if score > 0:
            scored.append((score, f"Prior {msg.get('role', 'message')} message: {content}"))
    for turn in record.get("turn_summaries", []):
        summary = str(turn.get("summary", "")).strip()
        if not summary:
            continue
        tokens = _tokenize(summary)
        overlap = len(query_tokens.intersection(tokens))
        score = overlap + 0.5
        if score > 0:
            scored.append((score, f"Stored preference summary: {summary}"))
    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    snippets: list[str] = []
    for _, text in scored:
        if text in seen:
            continue
        snippets.append(text)
        seen.add(text)
        if len(snippets) >= top_k:
            break
    return snippets
