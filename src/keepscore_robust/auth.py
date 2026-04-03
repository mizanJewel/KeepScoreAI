from __future__ import annotations

import hashlib
import json
from pathlib import Path

from keepscore_robust.data import DATA_ROOT


ACCOUNTS_PATH = DATA_ROOT / "accounts.json"


def load_accounts() -> list[dict]:
    return json.loads(ACCOUNTS_PATH.read_text(encoding="utf-8"))


def get_account(username: str) -> dict | None:
    target = username.strip().lower()
    for account in load_accounts():
        if account.get("username", "").lower() == target:
            return account
    return None


def verify_login(username: str, password: str) -> dict | None:
    account = get_account(username)
    if account is None:
        return None
    digest = hashlib.sha256(f"{account['salt']}:{password}".encode("utf-8")).hexdigest()
    if digest != account.get("password_hash"):
        return None
    return {
        "username": account["username"],
        "display_name": account.get("display_name", account["username"]),
        "role": account.get("role", "user"),
    }
