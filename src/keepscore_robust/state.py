from __future__ import annotations

from keepscore_robust.models import ParsedTurn, ShopperProfile


COMPATIBLE = {
    frozenset({"softness", "premium"}),
    frozenset({"softness", "walking"}),
    frozenset({"support", "walking"}),
}
CONFLICTS = {frozenset({"support", "lightweight"})}


def _objective_order(profile: ShopperProfile) -> list[str]:
    return [k for k, _ in sorted(profile.objectives.items(), key=lambda kv: kv[1], reverse=True) if _ > 0]


def update_profile(profile: ShopperProfile, parsed: ParsedTurn) -> tuple[ShopperProfile, list[str]]:
    why_changed: list[str] = []
    previous_order = _objective_order(profile)
    profile.history.append(parsed.raw_text)

    if parsed.budget_max is not None:
        old_budget = profile.budget_max
        profile.budget_max = parsed.budget_max
        if old_budget is None:
            why_changed.append(f"Budget cap set to ${parsed.budget_max:.0f}.")
        elif old_budget != parsed.budget_max:
            why_changed.append(f"Budget cap changed from ${old_budget:.0f} to ${parsed.budget_max:.0f}.")

    if parsed.category:
        if profile.category and profile.category != parsed.category:
            profile.transition_label = "override"
            profile.transition_reason = f"category changed from {profile.category} to {parsed.category}"
            why_changed.append(f"Category switched from {profile.category} to {parsed.category}.")
        profile.category = parsed.category

    if parsed.gender:
        if profile.gender and profile.gender != parsed.gender:
            why_changed.append(f"Gender preference changed from {profile.gender} to {parsed.gender}.")
            profile.transition_label = "override"
            profile.transition_reason = f"gender changed from {profile.gender} to {parsed.gender}"
        elif profile.gender is None:
            why_changed.append(f"Gender preference set to {parsed.gender}.")
        profile.gender = parsed.gender

    if parsed.color and parsed.color not in profile.color_preferences:
        profile.color_preferences.append(parsed.color)
        why_changed.append(f"Added color preference for {parsed.color}.")

    if parsed.width:
        if profile.width_need != parsed.width:
            profile.width_need = parsed.width
            why_changed.append("Wide-width preference is now active.")

    if parsed.preferred_shelves:
        profile.preferred_shelves = list(dict.fromkeys(parsed.preferred_shelves))

    for key in ["softness", "premium", "lightweight", "support"]:
        bump = getattr(parsed, key)
        if bump > 0:
            old = profile.objectives.get(key, 0.0)
            profile.objectives[key] = min(1.0, old * 0.65 + bump)
            if old == 0:
                why_changed.append(f"Activated {key} as a new priority.")
            else:
                why_changed.append(f"Increased weight on {key}.")

    if parsed.reject_current and profile.last_recommended_ids:
        rejected = profile.last_recommended_ids[0]
        if rejected not in profile.rejected_product_ids:
            profile.rejected_product_ids.append(rejected)
            why_changed.append("The previous top recommendation was rejected and removed from future ranking.")
        profile.transition_label = "override"
        profile.transition_reason = "user rejected current top choice"

    current_order = _objective_order(profile)
    if not previous_order and current_order:
        profile.transition_label = "add"
        profile.transition_reason = f"activated a new leading priority: {current_order[0]}"
    elif previous_order and current_order:
        if previous_order[:1] == current_order[:1]:
            profile.transition_label = "refine"
            profile.transition_reason = f"latest turn reinforced {current_order[0]}"
        else:
            pair = frozenset({previous_order[0], current_order[0]})
            if pair in CONFLICTS:
                profile.transition_label = "coexist"
                profile.transition_reason = f"{current_order[0]} added beside {previous_order[0]}"
            elif pair in COMPATIBLE:
                profile.transition_label = "refine"
                profile.transition_reason = f"{current_order[0]} narrowed the direction while staying compatible with {previous_order[0]}"
            else:
                profile.transition_label = "override"
                profile.transition_reason = f"leading priority changed from {previous_order[0]} to {current_order[0]}"

    if not why_changed:
        why_changed.append("The latest turn stayed broadly compatible with the current shopping profile.")

    return profile, why_changed
