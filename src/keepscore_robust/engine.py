from __future__ import annotations

from dataclasses import replace
from typing import Any

from keepscore_robust.agents import (
    EvidenceAgent,
    ExplanationAgent,
    ParserAgent,
    ProfileAgent,
    RecommendationAgent,
    RetrievalAgent,
    ShelfAgent,
    VisionAgent,
)
from keepscore_robust.data import load_products, load_reviews
from keepscore_robust.image_analysis import analyze_uploaded_shoe_image
from keepscore_robust.llm import ollama_chat
from keepscore_robust.mcp import MCPToolRegistry
from keepscore_robust.models import EngineResult, Product, Recommendation, ShopperProfile
from keepscore_robust.parsing import parse_turn
from keepscore_robust.retrieval import candidate_retrieval, retrieve_evidence
from keepscore_robust.scoring import adaptive_context, high_keep_score, launch_score, personalized_score, trending_score
from keepscore_robust.state import update_profile


class KeepScoreEngine:
    def __init__(self, products: list[Product] | None = None):
        self.products = products or load_products()
        self.reviews = load_reviews()
        self.registry = self._build_registry()
        self.parser_agent = ParserAgent(self.registry)
        self.profile_agent = ProfileAgent(self.registry)
        self.vision_agent = VisionAgent(self.registry)
        self.retrieval_agent = RetrievalAgent(self.registry)
        self.recommendation_agent = RecommendationAgent(self.registry)
        self.evidence_agent = EvidenceAgent(self.registry)
        self.shelf_agent = ShelfAgent(self.registry)
        self.explanation_agent = ExplanationAgent(self.registry)

    def new_profile(self) -> ShopperProfile:
        return ShopperProfile()

    def _build_registry(self) -> MCPToolRegistry:
        registry = MCPToolRegistry()
        registry.register("parse_turn", lambda text: parse_turn(text))
        registry.register("update_profile", lambda profile, parsed: update_profile(profile, parsed))
        registry.register("analyze_image", lambda image_bytes, filename: analyze_uploaded_shoe_image(image_bytes, filename))
        registry.register("retrieve_candidates", lambda profile: candidate_retrieval(self.products, profile))
        registry.register("score_candidates", self._score_candidates)
        registry.register("retrieve_evidence", lambda product_ids, profile: retrieve_evidence(product_ids, self.reviews, profile))
        registry.register("build_shelves", self._build_shelves)
        registry.register("compose_explanation", self._compose_explanation)
        return registry

    def _score_candidates(self, candidates: list[Product], profile: ShopperProfile) -> list[Recommendation]:
        context = adaptive_context(profile)
        profile.adaptive_state = dict(context)
        recs = [personalized_score(product, profile, context=context) for product in candidates]
        recs.sort(key=lambda rec: (rec.keep_score, rec.expected_utility), reverse=True)
        return recs

    def _compute_result(
        self,
        profile: ShopperProfile,
        parsed: Any,
        why_changed: list[str],
        *,
        message: str,
        chat_history: list[dict] | None = None,
        memory_snippets: list[str] | None = None,
        image_description: str | None = None,
        image_search_query: str | None = None,
        image_analysis: dict | None = None,
        agent_trace: list[dict[str, Any]] | None = None,
    ) -> EngineResult:
        agent_trace = agent_trace or []
        memory_snippets = memory_snippets or []

        candidates = self.retrieval_agent.run(profile, agent_trace)
        recs = self.recommendation_agent.run(profile, candidates, agent_trace)
        recommendations = recs[:6]
        profile.last_recommended_ids = [rec.product.product_id for rec in recommendations]
        evidence = self.evidence_agent.run([rec.product.product_id for rec in recommendations[:4]], profile, agent_trace)
        shelves = self.shelf_agent.run(profile, agent_trace)
        explanation, llm_model, llm_error = self.explanation_agent.run(
            message=message,
            profile=profile,
            recommendations=recommendations,
            evidence=evidence,
            why_changed=why_changed,
            chat_history=chat_history or [],
            memory_snippets=memory_snippets,
            trace=agent_trace,
        )
        return EngineResult(
            profile=profile,
            parsed_turn=parsed,
            recommendations=recommendations,
            shelves=shelves,
            evidence=evidence,
            explanation=explanation,
            why_changed=why_changed,
            memory_snippets=memory_snippets,
            llm_model=llm_model,
            llm_error=llm_error,
            image_description=image_description,
            image_search_query=image_search_query,
            image_analysis=image_analysis or {},
            agent_trace=agent_trace,
            mcp_trace=self.registry.consume_trace(),
        )

    def process_turn(
        self,
        message: str,
        profile: ShopperProfile | None = None,
        *,
        chat_history: list[dict] | None = None,
        memory_snippets: list[str] | None = None,
    ) -> EngineResult:
        profile = profile or self.new_profile()
        agent_trace: list[dict[str, Any]] = []
        parsed = self.parser_agent.run(message, agent_trace)
        profile, why_changed = self.profile_agent.run(profile, parsed, agent_trace)
        return self._compute_result(
            profile,
            parsed,
            why_changed,
            message=message,
            chat_history=chat_history,
            memory_snippets=memory_snippets,
            agent_trace=agent_trace,
        )

    def process_uploaded_image(
        self,
        image_bytes: bytes,
        filename: str,
        profile: ShopperProfile | None = None,
        *,
        chat_history: list[dict] | None = None,
        memory_snippets: list[str] | None = None,
    ) -> EngineResult:
        profile = profile or self.new_profile()
        agent_trace: list[dict[str, Any]] = []
        analysis = self.vision_agent.run(image_bytes, filename, agent_trace)
        image_query = str(analysis.get("search_query") or analysis.get("description") or "shoe")
        parsed = self.parser_agent.run(image_query, agent_trace)
        profile, why_changed = self.profile_agent.run(profile, parsed, agent_trace)
        why_changed = [f"Image upload analyzed as: {analysis.get('description', 'shoe image uploaded.')}", *why_changed]
        return self._compute_result(
            profile,
            parsed,
            why_changed,
            message=f"Image upload: {image_query}",
            chat_history=chat_history,
            memory_snippets=memory_snippets,
            image_description=str(analysis.get("description") or ""),
            image_search_query=image_query,
            image_analysis=analysis,
            agent_trace=agent_trace,
        )

    def refresh(
        self,
        profile: ShopperProfile | None = None,
        *,
        chat_history: list[dict] | None = None,
        memory_snippets: list[str] | None = None,
    ) -> EngineResult:
        profile = profile or self.new_profile()
        agent_trace: list[dict[str, Any]] = []
        parsed = self.parser_agent.run("refresh", agent_trace)
        why_changed = ["Recommendations were recomputed from the current stored profile."]
        return self._compute_result(
            profile,
            parsed,
            why_changed,
            message="refresh",
            chat_history=chat_history,
            memory_snippets=memory_snippets,
            agent_trace=agent_trace,
        )

    def _relevance_bonus(self, product: Product, profile: ShopperProfile) -> float:
        bonus = 0.0
        if profile.gender:
            if product.gender == profile.gender:
                bonus += 0.25
            elif product.gender == "unisex":
                bonus += 0.15
            else:
                bonus -= 0.20
        if profile.category and profile.category == product.category:
            bonus += 0.30
        if profile.color_preferences and set(profile.color_preferences).intersection(set(product.colors)):
            bonus += 0.15
        if profile.width_need and profile.width_need in [w.lower() for w in product.widths]:
            bonus += 0.15
        if profile.objectives.get("softness", 0) > 0 and product.cushioning in {"high", "max"}:
            bonus += 0.10
        if profile.objectives.get("premium", 0) > 0 and product.premium_level == "high":
            bonus += 0.10
        if profile.objectives.get("lightweight", 0) > 0 and product.weight_grams <= 260:
            bonus += 0.10
        if profile.objectives.get("support", 0) > 0 and product.support == "high":
            bonus += 0.10
        return min(max(bonus, 0.0), 1.0)

    def _discovery_candidates(self, profile: ShopperProfile, min_count: int = 12) -> list[Product]:
        staged_profiles = [
            profile,
            replace(profile, color_preferences=[]),
            replace(profile, color_preferences=[], width_need=None),
            replace(profile, category=None, color_preferences=[], width_need=None),
            replace(profile, budget_max=None, category=None, color_preferences=[], width_need=None),
        ]

        merged: list[Product] = []
        seen_product_ids: set[str] = set()
        for staged_profile in staged_profiles:
            for product in candidate_retrieval(self.products, staged_profile):
                if product.product_id in seen_product_ids:
                    continue
                merged.append(product)
                seen_product_ids.add(product.product_id)
            if len(merged) >= min_count:
                break
        return merged

    def _build_shelves(self, profile: ShopperProfile) -> dict[str, list[Recommendation]]:
        shelf_candidates = self._discovery_candidates(profile)
        context = adaptive_context(profile)
        personalized = [personalized_score(product, profile, context=context) for product in shelf_candidates]
        personalized.sort(key=lambda rec: (rec.keep_score, rec.expected_utility), reverse=True)

        trending = sorted(
            personalized,
            key=lambda rec: trending_score(rec.product, self._relevance_bonus(rec.product, profile)),
            reverse=True,
        )[:3]
        new_launch = sorted(
            personalized,
            key=lambda rec: launch_score(rec.product, self._relevance_bonus(rec.product, profile)),
            reverse=True,
        )[:3]
        high_keep = sorted(
            personalized,
            key=lambda rec: high_keep_score(rec.product, rec),
            reverse=True,
        )[:3]

        return {
            "Recommended Matches": personalized[:6],
            "Trending Shoes": [replace(rec, keep_score=trending_score(rec.product, self._relevance_bonus(rec.product, profile))) for rec in trending],
            "New Launch": [replace(rec, keep_score=launch_score(rec.product, self._relevance_bonus(rec.product, profile))) for rec in new_launch],
            "High KeepScore": [replace(rec, keep_score=high_keep_score(rec.product, rec)) for rec in high_keep],
        }

    def _compose_explanation(
        self,
        *,
        message: str,
        profile: ShopperProfile,
        recommendations: list[Recommendation],
        evidence: dict[str, list],
        why_changed: list[str],
        chat_history: list[dict],
        memory_snippets: list[str],
    ) -> tuple[str, str | None, str | None]:
        fallback = self._heuristic_explanation(profile, recommendations, evidence, why_changed, memory_snippets)
        if not recommendations:
            return fallback, None, None

        top = recommendations[0]
        review_lines = [
            f"- Review evidence ({item.score:.2f}): {item.text}"
            for item in evidence.get(top.product.product_id, [])[:2]
        ] or ["- Review evidence is limited for the current top pick."]

        recent_history = chat_history[-4:]
        history_lines = [f"- {msg.get('role', 'message')}: {msg.get('content', '')}" for msg in recent_history if msg.get("content")]
        memory_lines = [f"- {snippet}" for snippet in memory_snippets[:4]]
        adaptive_lines = [f"- {key}: {value}" for key, value in profile.adaptive_state.items()]
        profile_lines = [
            f"- Gender: {profile.gender or 'unspecified'}",
            f"- Category: {profile.category or 'unspecified'}",
            f"- Budget max: {f'${profile.budget_max:.0f}' if profile.budget_max is not None else 'unspecified'}",
            f"- Colors: {', '.join(profile.color_preferences) if profile.color_preferences else 'none'}",
            f"- Width: {profile.width_need or 'unspecified'}",
            f"- Priorities: {', '.join([k for k, v in sorted(profile.objectives.items(), key=lambda kv: kv[1], reverse=True) if v > 0][:4]) or 'balanced value'}",
        ]
        rec_lines = [
            f"- #{idx}: {rec.product.name} ({rec.product.product_id}), KeepScore {rec.keep_score:.1f}, fit {rec.fit_confidence:.0%}, return risk {rec.return_risk:.0%}, reasons: {', '.join(rec.reasons[:3])}"
            for idx, rec in enumerate(recommendations[:3], start=1)
        ]

        system_prompt = (
            "You are a shoe shopping assistant in a multi-agent recommendation system. "
            "Use the ranked recommendations as the source of truth. "
            "Use retrieved review evidence, adaptive scoring context, and stored user memory to explain comfort, fit, and tradeoffs. "
            "Do not invent products or claims. Keep the answer concise and personalized."
        )
        user_prompt = "\n".join(
            [
                f"Latest user request: {message}",
                "Current shopper profile:",
                *profile_lines,
                "Adaptive scoring state:",
                *(adaptive_lines or ["- none"]),
                "Why the profile changed:",
                *[f"- {item}" for item in why_changed[:4]],
                "Top ranked recommendations:",
                *rec_lines,
                "Retrieved review evidence for the top pick:",
                *review_lines,
                "Relevant prior memory:",
                *(memory_lines or ["- No prior memory snippets were retrieved."]),
                "Recent conversation:",
                *(history_lines or ["- No recent history yet."]),
                "Write 1 short paragraph plus a brief second sentence if needed.",
            ]
        )
        result = ollama_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        if result.get("ok"):
            return result["content"], result.get("model"), None
        return fallback, None, result.get("error")

    def _heuristic_explanation(
        self,
        profile: ShopperProfile,
        recommendations: list[Recommendation],
        evidence: dict[str, list],
        why_changed: list[str],
        memory_snippets: list[str],
    ) -> str:
        if not recommendations:
            return "I could not find a strong match from the current constraints, so the system needs a broader search."
        top = recommendations[0]
        snippets = evidence.get(top.product.product_id, [])
        evidence_line = (
            snippets[0].text
            if snippets
            else "Evidence coverage is light, so this explanation is based mostly on product attributes and behavior metrics."
        )
        goals = [k for k, v in sorted(profile.objectives.items(), key=lambda kv: kv[1], reverse=True) if v > 0]
        goals_text = ", ".join(goals[:2]) if goals else "balanced everyday value"
        gender_text = f" for {profile.gender} shoppers" if profile.gender else ""
        mode_text = profile.adaptive_state.get("mode", "adaptive")
        memory_line = f" I also remembered: {memory_snippets[0]}" if memory_snippets else ""
        return (
            f"Top pick: {top.product.name} with KeepScore {top.keep_score:.1f}. "
            f"The system is currently in {mode_text} mode, and this shoe rose because your priorities emphasize {goals_text}{gender_text}, with fit confidence {top.fit_confidence:.2f} and return risk {top.return_risk:.2f}. "
            f"Evidence: {evidence_line} Latest state change: {why_changed[0]}{memory_line}"
        )
