from __future__ import annotations

from datetime import date

from keepscore_robust.models import Product, Recommendation, ShopperProfile


TODAY = date(2026, 4, 2)
LEVEL_SCORE = {"low": 0.35, "medium": 0.65, "high": 0.87, "max": 0.96}


ALLOWED_GENDER_MAP = {
    None: {"men", "women", "unisex"},
    "men": {"men", "unisex"},
    "women": {"women", "unisex"},
    "unisex": {"unisex", "men", "women"},
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _level(value: str) -> float:
    return LEVEL_SCORE.get(value.lower(), 0.6)


def _profile_specificity(profile: ShopperProfile) -> float:
    active_objectives = sum(1 for value in profile.objectives.values() if value > 0)
    flags = [
        profile.budget_max is not None,
        profile.category is not None,
        profile.gender is not None,
        bool(profile.color_preferences),
        profile.width_need is not None,
        active_objectives > 0,
    ]
    return min(sum(flags) / 6.0, 1.0)


def adaptive_context(profile: ShopperProfile) -> dict[str, float | str]:
    specificity = _profile_specificity(profile)
    history_depth = min(len(profile.history) / 6.0, 1.0)
    rejection_pressure = min(len(profile.rejected_product_ids) / 4.0, 1.0)
    active_objectives = sum(1 for value in profile.objectives.values() if value > 0)
    exploration = max(0.0, 1.0 - specificity)
    refinement = min((history_depth + specificity) / 2.0, 1.0)
    objective_focus = min(active_objectives / 3.0, 1.0)
    return {
        "specificity": round(specificity, 3),
        "history_depth": round(history_depth, 3),
        "rejection_pressure": round(rejection_pressure, 3),
        "objective_focus": round(objective_focus, 3),
        "mode": "refine" if refinement >= 0.55 else "explore",
        "weight_preference": round(0.18 + 0.10 * specificity + 0.05 * objective_focus + 0.04 * rejection_pressure, 3),
        "weight_fit": round(0.16 + 0.05 * refinement + 0.03 * rejection_pressure, 3),
        "weight_keep_probability": round(0.13 + 0.04 * history_depth, 3),
        "weight_budget": round(0.07 + (0.06 if profile.budget_max is not None else 0.0), 3),
        "weight_quality": round(0.09 + 0.04 * exploration, 3),
        "weight_reliability": round(0.08 + 0.04 * refinement, 3),
        "weight_expected_utility": round(0.09 + 0.05 * exploration, 3),
        "weight_return_penalty": round(0.12 + 0.06 * rejection_pressure + 0.03 * specificity, 3),
    }


def _gender_score(product: Product, profile: ShopperProfile) -> float:
    if profile.gender is None:
        return 0.82
    if product.gender == profile.gender:
        return 1.0
    if product.gender == "unisex":
        return 0.90
    if profile.gender == "unisex":
        return 0.86
    return 0.10


def _budget_compatibility(product: Product, profile: ShopperProfile) -> tuple[float, list[str], list[str]]:
    reasons: list[str] = []
    cautions: list[str] = []
    if profile.budget_max is None:
        return 0.70, reasons, cautions
    if product.sale_price <= profile.budget_max:
        reasons.append("within your budget")
        return 1.0, reasons, cautions
    overshoot = product.sale_price - profile.budget_max
    score = _clamp(1.0 - overshoot / 40.0)
    if overshoot <= 15:
        cautions.append("slightly above your stated budget")
    else:
        cautions.append("well above your stated budget")
    return score, reasons, cautions


def _preference_match(product: Product, profile: ShopperProfile) -> tuple[float, list[str], list[str], dict[str, float]]:
    reasons: list[str] = []
    cautions: list[str] = []
    breakdown: dict[str, float] = {}

    softness_score = _level(product.cushioning)
    premium_score = _level(product.premium_level)
    lightweight_score = _clamp(1.0 - (product.weight_grams - 210) / 140.0)
    support_score = _level(product.support)
    gender_score = _gender_score(product, profile)
    color_score = 1.0 if not profile.color_preferences else (1.0 if set(profile.color_preferences).intersection(set(product.colors)) else 0.3)
    width_score = 0.72 if not profile.width_need else (1.0 if profile.width_need in [w.lower() for w in product.widths] else 0.2)
    category_score = 0.78 if not profile.category else (1.0 if profile.category == product.category else 0.25)

    breakdown.update(
        {
            "softness": softness_score,
            "premium": premium_score,
            "lightweight": lightweight_score,
            "support": support_score,
            "gender": gender_score,
            "color": color_score,
            "width": width_score,
            "category": category_score,
        }
    )

    objective_weights = profile.objectives.copy()
    total_weight = sum(objective_weights.values())
    if total_weight == 0:
        objective_weights = {"softness": 0.30, "premium": 0.15, "lightweight": 0.10, "support": 0.20}
        total_weight = sum(objective_weights.values())
    objective_weights = {k: v / total_weight for k, v in objective_weights.items()}

    obj_mix = (
        objective_weights.get("softness", 0) * softness_score
        + objective_weights.get("premium", 0) * premium_score
        + objective_weights.get("lightweight", 0) * lightweight_score
        + objective_weights.get("support", 0) * support_score
    )

    final = _clamp(0.50 * obj_mix + 0.10 * gender_score + 0.10 * color_score + 0.10 * width_score + 0.20 * category_score)

    if profile.gender:
        if product.gender == profile.gender:
            reasons.append(f"matches your {profile.gender} selection")
        elif product.gender == "unisex":
            reasons.append("unisex option still fits your gender selection")

    if objective_weights.get("softness", 0) > 0.15 and softness_score >= 0.85:
        reasons.append("strong softness match")
    if objective_weights.get("premium", 0) > 0.15 and premium_score >= 0.85:
        reasons.append("premium feel aligns well")
    if objective_weights.get("lightweight", 0) > 0.15 and lightweight_score >= 0.80:
        reasons.append("lighter than many alternatives")
    if objective_weights.get("support", 0) > 0.15 and support_score >= 0.85:
        reasons.append("support-focused fit")
    if color_score >= 1.0 and profile.color_preferences:
        reasons.append("matches your color direction")
    if width_score >= 1.0 and profile.width_need:
        reasons.append("available in your preferred width")

    if gender_score < 0.4:
        cautions.append("gender targeting is weaker than your current preference")
    if objective_weights.get("support", 0) > 0.15 and support_score < 0.55:
        cautions.append("support is weaker than your current preference")
    if objective_weights.get("lightweight", 0) > 0.15 and lightweight_score < 0.60:
        cautions.append("not especially light")
    if objective_weights.get("premium", 0) > 0.15 and premium_score < 0.60:
        cautions.append("does not feel as premium as top alternatives")

    return final, reasons, cautions, breakdown


def _fit_confidence(product: Product, preference_match: float, budget_compatibility: float, profile: ShopperProfile) -> float:
    rating_component = product.avg_rating / 5.0
    width_component = 0.88 if "wide" in [w.lower() for w in product.widths] else 0.72
    gender_component = _gender_score(product, profile)
    return _clamp(
        0.28 * preference_match
        + 0.18 * budget_compatibility
        + 0.18 * rating_component
        + 0.12 * product.reliability_score
        + 0.12 * width_component
        + 0.12 * gender_component
    )


def _return_risk(product: Product, fit_confidence: float, preference_match: float, profile: ShopperProfile) -> float:
    gender_penalty = 0.08 if _gender_score(product, profile) < 0.4 else 0.0
    raw = 0.46 * product.return_rate + 0.22 * (1.0 - fit_confidence) + 0.16 * (1.0 - preference_match) + 0.08 * (1.0 - product.reliability_score) + gender_penalty
    return _clamp(raw)


def personalized_score(product: Product, profile: ShopperProfile, *, context: dict[str, float | str] | None = None) -> Recommendation:
    context = context or adaptive_context(profile)
    budget_compatibility, budget_reasons, budget_cautions = _budget_compatibility(product, profile)
    preference_match, pref_reasons, pref_cautions, pref_breakdown = _preference_match(product, profile)
    fit_confidence = _fit_confidence(product, preference_match, budget_compatibility, profile)
    return_risk = _return_risk(product, fit_confidence, preference_match, profile)
    keep_probability = _clamp(
        0.22 * preference_match
        + 0.20 * fit_confidence
        + 0.16 * budget_compatibility
        + 0.16 * product.quality_score
        + 0.12 * product.reliability_score
        + 0.14 * (1.0 - return_risk)
    )
    expected_utility = _clamp(0.45 * keep_probability + 0.20 * product.conversion_rate + 0.15 * product.avg_rating / 5.0 + 0.20 * product.quality_score)
    w_pref = float(context["weight_preference"])
    w_fit = float(context["weight_fit"])
    w_keep = float(context["weight_keep_probability"])
    w_budget = float(context["weight_budget"])
    w_quality = float(context["weight_quality"])
    w_reliability = float(context["weight_reliability"])
    w_expected = float(context["weight_expected_utility"])
    w_penalty = float(context["weight_return_penalty"])
    keep_score = round(
        100 * _clamp(
            w_pref * preference_match
            + w_fit * fit_confidence
            + w_keep * keep_probability
            + w_budget * budget_compatibility
            + w_quality * product.quality_score
            + w_reliability * product.reliability_score
            + w_expected * expected_utility
            - w_penalty * return_risk
        ),
        1,
    )

    reasons = list(dict.fromkeys(pref_reasons + budget_reasons + product.reasons_seed))[:4]
    cautions = list(dict.fromkeys(pref_cautions + budget_cautions + product.cautions))[:3]

    return Recommendation(
        product=product,
        keep_score=keep_score,
        fit_confidence=round(fit_confidence, 3),
        return_risk=round(return_risk, 3),
        preference_match=round(preference_match, 3),
        budget_compatibility=round(budget_compatibility, 3),
        expected_utility=round(expected_utility, 3),
        reasons=reasons,
        cautions=cautions,
        score_breakdown={
            "preference_match": round(preference_match, 3),
            "fit_confidence": round(fit_confidence, 3),
            "budget_compatibility": round(budget_compatibility, 3),
            "keep_probability": round(keep_probability, 3),
            "return_risk": round(return_risk, 3),
            "quality": round(product.quality_score, 3),
            "reliability": round(product.reliability_score, 3),
            "adaptive_mode": str(context["mode"]),
            "adaptive_specificity": round(float(context["specificity"]), 3),
            "adaptive_history_depth": round(float(context["history_depth"]), 3),
            "adaptive_rejection_pressure": round(float(context["rejection_pressure"]), 3),
            "adaptive_weight_preference": round(w_pref, 3),
            "adaptive_weight_fit": round(w_fit, 3),
            "adaptive_weight_keep_probability": round(w_keep, 3),
            "adaptive_weight_budget": round(w_budget, 3),
            "adaptive_weight_quality": round(w_quality, 3),
            "adaptive_weight_reliability": round(w_reliability, 3),
            "adaptive_weight_expected_utility": round(w_expected, 3),
            "adaptive_weight_return_penalty": round(w_penalty, 3),
            **{f"pref_{k}": round(v, 3) for k, v in pref_breakdown.items()},
        },
    )


def trending_score(product: Product, relevance_bonus: float = 0.0) -> float:
    return round(
        100
        * _clamp(
            0.25 * product.trend_velocity
            + 0.20 * product.click_through_rate
            + 0.15 * product.add_to_cart_rate
            + 0.15 * product.conversion_rate
            + 0.10 * min(product.stock / 80.0, 1.0)
            + 0.10 * (product.avg_rating / 5.0)
            + 0.05 * relevance_bonus
        ),
        1,
    )


def launch_score(product: Product, relevance_bonus: float = 0.0) -> float:
    days_old = max((TODAY - date.fromisoformat(product.release_date)).days, 1)
    freshness = _clamp(1.0 - (days_old / 180.0))
    return round(
        100
        * _clamp(
            0.35 * freshness
            + 0.20 * product.trend_velocity
            + 0.15 * (product.avg_rating / 5.0)
            + 0.10 * min(product.stock / 60.0, 1.0)
            + 0.10 * product.click_through_rate
            + 0.10 * relevance_bonus
        ),
        1,
    )


def high_keep_score(product: Product, rec: Recommendation) -> float:
    return round(
        100
        * _clamp(
            0.35 * (rec.keep_score / 100.0)
            + 0.20 * rec.fit_confidence
            + 0.15 * product.reliability_score
            + 0.10 * rec.budget_compatibility
            + 0.10 * product.quality_score
            + 0.10 * rec.preference_match
            - 0.15 * rec.return_risk
        ),
        1,
    )
