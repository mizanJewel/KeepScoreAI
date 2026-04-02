from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")


def ollama_chat(messages: list[dict[str, str]], *, timeout: float = 20.0) -> dict[str, Any]:
    payload = {
        "model": DEFAULT_OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    req = request.Request(
        f"{DEFAULT_OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "model": DEFAULT_OLLAMA_MODEL}

    message = body.get("message", {}) if isinstance(body, dict) else {}
    content = message.get("content", "").strip() if isinstance(message, dict) else ""
    if not content:
        return {"ok": False, "error": "empty response from Ollama", "model": DEFAULT_OLLAMA_MODEL}
    return {"ok": True, "content": content, "model": DEFAULT_OLLAMA_MODEL}
