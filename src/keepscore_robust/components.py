from __future__ import annotations

from collections import Counter

import streamlit as st

from keepscore_robust.models import EvidenceItem, Recommendation, ShopperProfile


def render_nav() -> None:
    st.markdown('<div class="ks-page-title">KeepScore Agentic Shopping</div>', unsafe_allow_html=True)
    st.caption("Stateful chat-guided recommendation with stronger shelves, grounded evidence, and NB-side analytics.")


def render_hero() -> None:
    st.markdown(
        """
        <div class="ks-hero">
          <div class="ks-eyebrow">AGENTIC SHOPPING INTELLIGENCE</div>
          <h1>Find the right shoe with chat that remembers your direction.</h1>
          <p>The chat carries forward your earlier constraints, the shelves show discovery options, and the dashboard summarizes what is trending, risky, or likely to be kept.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_strip() -> None:
    st.markdown(
        """
        <div class="ks-help-strip">
          <div><b>Refine naturally</b><br/>Try: “more soft one”, “more premium”, “lighter than this”, or “same idea but with more support”.</div>
          <div><b>Shop + browse together</b><br/>Recommended Matches use the live chat state. Trending, New Launch, and High KeepScore stay visible for discovery.</div>
          <div><b>Traceable scoring</b><br/>KeepScore is not just popularity. It blends fit confidence, return risk, budget match, quality, and chat-derived priorities.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_summary(profile: ShopperProfile) -> None:
    pieces: list[str] = []
    if profile.budget_max:
        pieces.append(f"Budget: up to ${profile.budget_max:.0f}")
    if profile.gender:
        pieces.append(f"Gender: {profile.gender}")
    if profile.category:
        pieces.append(f"Category: {profile.category}")
    if profile.color_preferences:
        pieces.append("Colors: " + ", ".join(profile.color_preferences))
    if profile.width_need:
        pieces.append(f"Width: {profile.width_need}")
    active_objectives = [k for k, v in sorted(profile.objectives.items(), key=lambda kv: kv[1], reverse=True) if v > 0]
    if active_objectives:
        pieces.append("Priorities: " + " > ".join(active_objectives[:4]))
    if profile.rejected_product_ids:
        pieces.append(f"Rejected products tracked: {len(profile.rejected_product_ids)}")
    pieces.append(f"Transition: {profile.transition_label}")
    pieces.append(f"Reason: {profile.transition_reason}")
    if not pieces:
        st.caption("No shopping context yet. Start with something like: 'I need black walking shoes under $130.'")
        return
    st.caption("Current shopping context")
    st.write(" • ".join(pieces))


def render_recommendation(rec: Recommendation, rank: int) -> None:
    product = rec.product
    with st.container(border=True):
        top_cols = st.columns([2.6, 1.4])
        with top_cols[0]:
            st.markdown(f"#### {rank}. {product.name}")
            st.write(f"${product.sale_price:.2f} • {product.category.title()} • {product.gender.title()} • Widths: {', '.join(product.widths)}")
            st.caption(
                f"Cushioning: {product.cushioning} • Support: {product.support} • Premium: {product.premium_level} • Colors: {', '.join(product.colors[:4])}"
            )
        with top_cols[1]:
            st.metric("KeepScore", f"{rec.keep_score:.1f}")
            st.caption("Personalized keep likelihood")

        row1 = st.columns(4)
        row1[0].metric("Fit confidence", f"{rec.fit_confidence:.0%}")
        row1[1].metric("Return risk", f"{rec.return_risk:.0%}")
        row1[2].metric("Preference match", f"{rec.preference_match:.0%}")
        row1[3].metric("Budget match", f"{rec.budget_compatibility:.0%}")

        row2 = st.columns(3)
        row2[0].metric("Expected utility", f"{rec.expected_utility:.2f}")
        row2[1].metric("Rating", f"{product.avg_rating:.1f}")
        row2[2].metric("Reviews", f"{product.review_count}")

        if rec.reasons:
            st.write("**Why it ranks well:** " + "; ".join(rec.reasons[:4]) + ".")
        if rec.cautions:
            st.warning("; ".join(rec.cautions[:2]))

        with st.expander("Score breakdown"):
            st.json(rec.score_breakdown)


def render_shelf_card(rec: Recommendation, shelf_name: str) -> None:
    product = rec.product
    reasons = " • ".join(rec.reasons[:2]) if rec.reasons else "Good overall fit"
    st.markdown(
        f"""
        <div class="ks-card">
            <div class="ks-card-title">{product.name}</div>
            <div class="ks-card-sub">{shelf_name} • ${product.sale_price:.2f} • {product.category.title()} • {product.gender.title()}</div>
            <div class="ks-stat-row">
                <div class="ks-stat">
                    <div class="ks-stat-label">Keep</div>
                    <div class="ks-stat-value">{rec.keep_score:.1f}</div>
                </div>
                <div class="ks-stat">
                    <div class="ks-stat-label">Rate</div>
                    <div class="ks-stat-value">{product.avg_rating:.1f}</div>
                </div>
                <div class="ks-stat">
                    <div class="ks-stat-label">Return</div>
                    <div class="ks-stat-value">{rec.return_risk * 100:.0f}%</div>
                </div>
            </div>
            <div class="ks-card-reasons">{reasons}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence(evidence: dict[str, list[EvidenceItem]]) -> None:
    nonempty = {k: v for k, v in evidence.items() if v}
    if not nonempty:
        st.info("Evidence coverage is light for the current picks.")
        return
    st.markdown("### Grounded evidence")
    for product_id, items in nonempty.items():
        with st.container(border=True):
            st.write(f"**{product_id}**")
            for item in items:
                st.caption(f"score {item.score:.2f} • tags: {', '.join(item.tags)}")
                st.write(item.text)


def render_dashboard_snapshot(metrics: list[dict], summary_bullets: list[str]) -> None:
    cols = st.columns(len(metrics)) if metrics else []
    for idx, metric in enumerate(metrics):
        cols[idx].metric(metric["label"], metric["value"], metric.get("delta"))
    if summary_bullets:
        st.markdown("### Summary")
        for bullet in summary_bullets:
            st.write(f"- {bullet}")


def render_dashboard_answer(answer: dict | None) -> None:
    if not answer:
        return
    st.markdown("### Dashboard assistant")
    st.write(answer.get("answer", ""))
    for bullet in answer.get("bullets", []):
        st.write(f"- {bullet}")
    with st.expander("Tool trace"):
        st.json(answer.get("tool_trace", {}))


def build_use_case_mentions(recommendations: list[Recommendation]) -> list[tuple[str, int]]:
    counter = Counter()
    for rec in recommendations:
        counter.update(rec.product.use_cases)
    return counter.most_common(6)
