from __future__ import annotations

import json
from pathlib import Path

from keepscore_robust.models import Product, ReviewSnippet


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
PRODUCTS_PATH = DATA_ROOT / "products.json"
REVIEWS_PATH = DATA_ROOT / "reviews.json"


def _read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_products() -> list[Product]:
    return [Product(**row) for row in _read_json(PRODUCTS_PATH)]


def load_reviews() -> list[ReviewSnippet]:
    return [ReviewSnippet(**row) for row in _read_json(REVIEWS_PATH)]
