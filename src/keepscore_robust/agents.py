from __future__ import annotations

from dataclasses import asdict
from typing import Any

from keepscore_robust.models import ShopperProfile
from keepscore_robust.mcp import MCPToolRegistry


class BaseAgent:
    name = "base-agent"

    def __init__(self, registry: MCPToolRegistry):
        self.registry = registry

    def log(self, trace: list[dict[str, Any]], summary: str, **extra: Any) -> None:
        payload = {"agent": self.name, "summary": summary}
        payload.update(extra)
        trace.append(payload)


class ParserAgent(BaseAgent):
    name = "parser-agent"

    def run(self, message: str, trace: list[dict[str, Any]]) -> Any:
        parsed = self.registry.call("parse_turn", agent=self.name, text=message)
        self.log(trace, "Parsed shopper input into structured turn state.", raw_text=message)
        return parsed


class ProfileAgent(BaseAgent):
    name = "profile-agent"

    def run(self, profile: ShopperProfile, parsed: Any, trace: list[dict[str, Any]]) -> tuple[ShopperProfile, list[str]]:
        updated_profile, why_changed = self.registry.call("update_profile", agent=self.name, profile=profile, parsed=parsed)
        self.log(
            trace,
            "Updated shopper profile and adaptive session state.",
            transition=updated_profile.transition_label,
            history_depth=len(updated_profile.history),
        )
        return updated_profile, why_changed


class VisionAgent(BaseAgent):
    name = "vision-agent"

    def run(self, image_bytes: bytes, filename: str, trace: list[dict[str, Any]]) -> dict:
        analysis = self.registry.call("analyze_image", agent=self.name, image_bytes=image_bytes, filename=filename)
        self.log(trace, "Analyzed uploaded shoe image.", filename=filename, mode=analysis.get("analysis_mode"))
        return analysis


class RetrievalAgent(BaseAgent):
    name = "retrieval-agent"

    def run(self, profile: ShopperProfile, trace: list[dict[str, Any]]) -> list[Any]:
        candidates = self.registry.call("retrieve_candidates", agent=self.name, profile=profile)
        self.log(trace, "Retrieved candidate shoes from the catalog.", candidate_count=len(candidates))
        return candidates


class RecommendationAgent(BaseAgent):
    name = "recommendation-agent"

    def run(self, profile: ShopperProfile, candidates: list[Any], trace: list[dict[str, Any]]) -> list[Any]:
        recs = self.registry.call("score_candidates", agent=self.name, candidates=candidates, profile=profile)
        self.log(trace, "Scored candidates with adaptive KeepScore.", recommendation_count=len(recs))
        return recs


class EvidenceAgent(BaseAgent):
    name = "evidence-agent"

    def run(self, product_ids: list[str], profile: ShopperProfile, trace: list[dict[str, Any]]) -> dict[str, list[Any]]:
        evidence = self.registry.call("retrieve_evidence", agent=self.name, product_ids=product_ids, profile=profile)
        self.log(trace, "Retrieved review evidence for top products.", product_count=len(product_ids))
        return evidence


class ShelfAgent(BaseAgent):
    name = "shelf-agent"

    def run(self, profile: ShopperProfile, trace: list[dict[str, Any]]) -> dict[str, list[Any]]:
        shelves = self.registry.call("build_shelves", agent=self.name, profile=profile)
        self.log(trace, "Built discovery shelves.", shelf_names=list(shelves.keys()))
        return shelves


class ExplanationAgent(BaseAgent):
    name = "explanation-agent"

    def run(
        self,
        *,
        message: str,
        profile: ShopperProfile,
        recommendations: list[Any],
        evidence: dict[str, list[Any]],
        why_changed: list[str],
        chat_history: list[dict],
        memory_snippets: list[str],
        trace: list[dict[str, Any]],
    ) -> tuple[str, str | None, str | None]:
        explanation, llm_model, llm_error = self.registry.call(
            "compose_explanation",
            agent=self.name,
            message=message,
            profile=profile,
            recommendations=recommendations,
            evidence=evidence,
            why_changed=why_changed,
            chat_history=chat_history,
            memory_snippets=memory_snippets,
        )
        self.log(
            trace,
            "Generated final assistant response.",
            llm_model=llm_model or "heuristic-fallback",
            used_memory=bool(memory_snippets),
        )
        return explanation, llm_model, llm_error


def profile_snapshot(profile: ShopperProfile) -> dict[str, Any]:
    payload = asdict(profile)
    payload["history"] = payload.get("history", [])[-3:]
    return payload
