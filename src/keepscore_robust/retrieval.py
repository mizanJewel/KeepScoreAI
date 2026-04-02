from __future__ import annotations

from keepscore_robust.models import EvidenceItem, Product, ReviewSnippet, ShopperProfile


ALLOWED_GENDER_MAP = {
    None: {"men", "women", "unisex"},
    "men": {"men", "unisex"},
    "women": {"women", "unisex"},
    "unisex": {"unisex", "men", "women"},
}


def _gender_allows(product: Product, requested_gender: str | None) -> bool:
    allowed = ALLOWED_GENDER_MAP.get(requested_gender, {"men", "women", "unisex"})
    return product.gender in allowed


def candidate_retrieval(products: list[Product], profile: ShopperProfile) -> list[Product]:
    candidates: list[Product] = []
    rejected = set(profile.rejected_product_ids)
    for product in products:
        if product.product_id in rejected:
            continue
        if not _gender_allows(product, profile.gender):
            continue
        if profile.category and product.category != profile.category:
            continue
        if profile.budget_max is not None and product.sale_price > profile.budget_max + 25:
            continue
        if profile.width_need and profile.width_need not in [w.lower() for w in product.widths]:
            continue
        if profile.color_preferences:
            wanted = set(profile.color_preferences)
            if not wanted.intersection(set(product.colors)):
                continue
        candidates.append(product)

    if candidates:
        return candidates

    # graceful fallback: relax color first, then category, but keep gender and budget constraints
    for product in products:
        if product.product_id in rejected:
            continue
        if not _gender_allows(product, profile.gender):
            continue
        if profile.category and product.category != profile.category:
            continue
        if profile.budget_max is not None and product.sale_price > profile.budget_max + 25:
            continue
        candidates.append(product)

    if candidates:
        return candidates
    return [p for p in products if p.product_id not in rejected and _gender_allows(p, profile.gender)]


TAG_KEYWORDS = {
    "softness": {"softness", "soft", "plush"},
    "premium": {"premium"},
    "lightweight": {"lightweight", "speed"},
    "support": {"support", "stability", "walking"},
}


def retrieve_evidence(product_ids: list[str], reviews: list[ReviewSnippet], profile: ShopperProfile, top_k: int = 2) -> dict[str, list[EvidenceItem]]:
    targets = [key for key, value in profile.objectives.items() if value > 0]
    output: dict[str, list[EvidenceItem]] = {pid: [] for pid in product_ids}
    for pid in product_ids:
        ranked: list[EvidenceItem] = []
        for snippet in reviews:
            if snippet.product_id != pid:
                continue
            score = 0.35
            for objective in targets:
                if TAG_KEYWORDS.get(objective, set()).intersection(set(snippet.tags)):
                    score += 0.25
            if snippet.sentiment == "positive":
                score += 0.15
            elif snippet.sentiment == "mixed":
                score += 0.05
            ranked.append(EvidenceItem(product_id=pid, text=snippet.text, tags=snippet.tags, score=min(score, 0.95), source="reviews"))
        ranked.sort(key=lambda item: item.score, reverse=True)
        output[pid] = ranked[:top_k]
    return output
