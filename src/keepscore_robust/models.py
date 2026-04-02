from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReviewSnippet:
    product_id: str
    text: str
    tags: list[str]
    sentiment: str


@dataclass
class Product:
    product_id: str
    name: str
    brand: str
    category: str
    gender: str
    use_cases: list[str]
    colors: list[str]
    widths: list[str]
    price: float
    sale_price: float
    release_date: str
    cushioning: str
    support: str
    premium_level: str
    weight_grams: int
    waterproof: bool
    stability: str
    comfort_rating: float
    avg_rating: float
    review_count: int
    stock: int
    return_rate: float
    conversion_rate: float
    click_through_rate: float
    add_to_cart_rate: float
    wishlist_count: int
    trend_velocity: float
    quality_score: float
    reliability_score: float
    style_tags: list[str] = field(default_factory=list)
    reasons_seed: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


@dataclass
class ParsedTurn:
    raw_text: str
    budget_max: float | None = None
    category: str | None = None
    gender: str | None = None
    color: str | None = None
    width: str | None = None
    softness: float = 0.0
    premium: float = 0.0
    lightweight: float = 0.0
    support: float = 0.0
    waterproof: bool | None = None
    preferred_shelves: list[str] = field(default_factory=list)
    reject_current: bool = False
    wants_similar: bool = False


@dataclass
class ShopperProfile:
    history: list[str] = field(default_factory=list)
    budget_max: float | None = None
    category: str | None = None
    gender: str | None = None
    color_preferences: list[str] = field(default_factory=list)
    width_need: str | None = None
    objectives: dict[str, float] = field(
        default_factory=lambda: {
            "softness": 0.0,
            "premium": 0.0,
            "lightweight": 0.0,
            "support": 0.0,
        }
    )
    preferred_shelves: list[str] = field(default_factory=list)
    rejected_product_ids: list[str] = field(default_factory=list)
    last_recommended_ids: list[str] = field(default_factory=list)
    transition_label: str = "initial"
    transition_reason: str = "first turn"


@dataclass
class EvidenceItem:
    product_id: str
    text: str
    tags: list[str]
    score: float
    source: str = "reviews"


@dataclass
class Recommendation:
    product: Product
    keep_score: float
    fit_confidence: float
    return_risk: float
    preference_match: float
    budget_compatibility: float
    expected_utility: float
    reasons: list[str]
    cautions: list[str]
    score_breakdown: dict[str, float]


@dataclass
class EngineResult:
    profile: ShopperProfile
    parsed_turn: ParsedTurn
    recommendations: list[Recommendation]
    shelves: dict[str, list[Recommendation]]
    evidence: dict[str, list[EvidenceItem]]
    explanation: str
    why_changed: list[str]
    memory_snippets: list[str] = field(default_factory=list)
    llm_model: str | None = None
    llm_error: str | None = None
