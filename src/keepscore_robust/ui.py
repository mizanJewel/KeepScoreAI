from __future__ import annotations

import pandas as pd
import streamlit as st

from keepscore_robust.components import (
    build_use_case_mentions,
    render_dashboard_answer,
    render_dashboard_snapshot,
    render_evidence,
    render_help_strip,
    render_image_analysis,
    render_nav,
    render_profile_summary,
    render_recommendation,
    render_shelf_card,
)
from keepscore_robust.engine import KeepScoreEngine
from keepscore_robust.memory import (
    load_user_record,
    normalize_user_id,
    profile_from_record,
    reset_user_record,
    retrieve_memory_snippets,
    save_user_record,
)
from keepscore_robust.models import EngineResult, Recommendation
from keepscore_robust.scoring import high_keep_score, launch_score, personalized_score, trending_score
from keepscore_robust.theme import apply_theme

STARTER_PROMPTS = [
    "I need women walking shoes in UK 6, black, under $130.",
    "Show me something more soft and a bit more premium.",
    "I want men wide shoes for standing all day. Budget is about $150.",
]

DASHBOARD_PROMPTS = [
    "What are the top shopper intents right now?",
    "Which products are on the risk watchlist?",
    "How reliable are the current recommendations?",
]


def _load_user_into_state(user_id: str) -> None:
    normalized = normalize_user_id(user_id)
    record = load_user_record(normalized)
    st.session_state.active_user_id = normalized
    st.session_state.user_id_input = normalized
    st.session_state.profile = profile_from_record(record)
    st.session_state.chat_messages = record.get("chat_messages", [])
    st.session_state.last_result = None
    st.session_state.dashboard_answer = None
    st.session_state.uploaded_shoe_image = None


def _ensure_state() -> None:
    ss = st.session_state
    if "engine" not in ss:
        ss.engine = KeepScoreEngine()
    if "active_user_id" not in ss:
        ss.active_user_id = "guest"
    if "user_id_input" not in ss:
        ss.user_id_input = ss.active_user_id
    if "profile" not in ss or "chat_messages" not in ss:
        _load_user_into_state(ss.active_user_id)
    if "last_result" not in ss:
        ss.last_result = None
    if "top_k_ui" not in ss:
        ss.top_k_ui = 6
    if "budget_value" not in ss:
        ss.budget_value = int(ss.profile.budget_max or 130)
    if "budget_widget" not in ss:
        ss.budget_widget = ss.budget_value
    if "pending_budget_widget_sync" not in ss:
        ss.pending_budget_widget_sync = False
    if "dashboard_answer" not in ss:
        ss.dashboard_answer = None
    if "ui_notice" not in ss:
        ss.ui_notice = None
    if "uploaded_shoe_image" not in ss:
        ss.uploaded_shoe_image = None
    if "just_reset" not in ss:
        ss.just_reset = False


def _run_turn(message: str) -> None:
    engine = st.session_state.engine
    record = load_user_record(st.session_state.active_user_id)
    memory_snippets = retrieve_memory_snippets(record, message)
    result = engine.process_turn(
        message,
        st.session_state.profile,
        chat_history=st.session_state.chat_messages,
        memory_snippets=memory_snippets,
    )
    st.session_state.profile = result.profile
    st.session_state.last_result = result
    st.session_state.chat_messages.append({"role": "user", "content": message})
    st.session_state.chat_messages.append({"role": "assistant", "content": result.explanation})
    save_user_record(
        st.session_state.active_user_id,
        result.profile,
        st.session_state.chat_messages,
        turn_summary={
            "summary": result.explanation,
            "user_message": message,
            "top_product_ids": [rec.product.product_id for rec in result.recommendations[:3]],
            "why_changed": result.why_changed,
        },
    )
    if result.profile.budget_max is not None:
        st.session_state.budget_value = int(result.profile.budget_max)
        st.session_state.pending_budget_widget_sync = True


def _run_image_turn(image_bytes: bytes, filename: str) -> None:
    engine = st.session_state.engine
    query = f"shoe image {filename}"
    record = load_user_record(st.session_state.active_user_id)
    memory_snippets = retrieve_memory_snippets(record, query)
    result = engine.process_uploaded_image(
        image_bytes,
        filename,
        st.session_state.profile,
        chat_history=st.session_state.chat_messages,
        memory_snippets=memory_snippets,
    )
    st.session_state.profile = result.profile
    st.session_state.last_result = result
    st.session_state.uploaded_shoe_image = {"bytes": image_bytes, "filename": filename}
    st.session_state.chat_messages.append({"role": "user", "content": f"Uploaded shoe image: {filename}"})
    st.session_state.chat_messages.append({"role": "assistant", "content": result.explanation})
    save_user_record(
        st.session_state.active_user_id,
        result.profile,
        st.session_state.chat_messages,
        turn_summary={
            "summary": result.explanation,
            "user_message": f"Uploaded image: {filename}",
            "top_product_ids": [rec.product.product_id for rec in result.recommendations[:3]],
            "why_changed": result.why_changed,
            "image_analysis": result.image_analysis,
        },
    )


def _refresh_if_needed() -> EngineResult | None:
    if st.session_state.just_reset:
        return None
    result = st.session_state.get("last_result")
    if result is None:
        if not st.session_state.chat_messages and not st.session_state.profile.history:
            return None
        record = load_user_record(st.session_state.active_user_id)
        seed_query = st.session_state.chat_messages[-1]["content"] if st.session_state.chat_messages else "refresh"
        result = st.session_state.engine.refresh(
            st.session_state.profile,
            chat_history=st.session_state.chat_messages,
            memory_snippets=retrieve_memory_snippets(record, seed_query),
        )
        st.session_state.last_result = result
    return result


def _sync_budget_control() -> None:
    if st.session_state.just_reset:
        return
    profile = st.session_state.profile
    budget_target = st.session_state.budget_value
    if profile.budget_max != budget_target:
        profile.budget_max = float(budget_target)
        if st.session_state.last_result is not None:
            record = load_user_record(st.session_state.active_user_id)
            st.session_state.last_result = st.session_state.engine.refresh(
                profile,
                chat_history=st.session_state.chat_messages,
                memory_snippets=retrieve_memory_snippets(record, f"budget {budget_target}"),
            )
            save_user_record(st.session_state.active_user_id, profile, st.session_state.chat_messages)
            st.session_state.ui_notice = f"Budget updated to ${budget_target}. Recommendations reranked."


def _on_budget_widget_change() -> None:
    current = st.session_state.get("budget_widget", st.session_state.get("budget_value", 130))
    st.session_state.budget_value = int(current)


def _apply_pending_budget_widget_sync() -> None:
    if st.session_state.pending_budget_widget_sync:
        st.session_state.budget_widget = st.session_state.budget_value
        st.session_state.pending_budget_widget_sync = False


def _dashboard_snapshot(result: EngineResult | None) -> tuple[list[dict], list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = st.session_state.engine
    profile = st.session_state.profile
    rows = []
    risk_rows = []
    for product in engine.products:
        rec = personalized_score(product, profile)
        rel_bonus = engine._relevance_bonus(product, profile)
        trend = trending_score(product, rel_bonus)
        launch = launch_score(product, rel_bonus)
        keep_proxy = high_keep_score(product, rec)
        rows.append(
            {
                "product_name": product.name,
                "category": product.category,
                "clicks": round(product.click_through_rate * 1000, 1),
                "impressions": int(product.review_count * 8 + product.wishlist_count),
                "avg_keep_score": round(rec.keep_score, 1),
                "avg_risk": round(rec.return_risk, 3),
                "trend_score": round(trend, 1),
                "launch_score": round(launch, 1),
                "quality": round(product.quality_score, 2),
                "reliability": round(product.reliability_score, 2),
                "high_keep_proxy": round(keep_proxy, 1),
            }
        )
        if rec.return_risk >= 0.28 or rec.keep_score <= 70:
            risk_rows.append(
                {
                    "product_name": product.name,
                    "category": product.category,
                    "avg_risk": round(rec.return_risk, 3),
                    "avg_keep_score": round(rec.keep_score, 1),
                    "impressions": int(product.review_count * 8 + product.wishlist_count),
                }
            )

    product_df = pd.DataFrame(rows).sort_values(["clicks", "avg_keep_score"], ascending=False)
    risk_df = pd.DataFrame(risk_rows).sort_values(["avg_risk", "avg_keep_score"], ascending=[False, True])

    recommendations = result.recommendations if result else []
    top_use_cases = build_use_case_mentions(recommendations)
    signal_df = pd.DataFrame(top_use_cases, columns=["use_case", "mentions"]) if top_use_cases else pd.DataFrame(columns=["use_case", "mentions"])

    metrics = [
        {"label": "Catalog size", "value": len(engine.products), "delta": None},
        {"label": "Avg top KeepScore", "value": f"{sum(r.keep_score for r in recommendations[:3]) / max(1, min(3, len(recommendations))):.1f}" if recommendations else "0.0", "delta": None},
        {"label": "Risk watchlist", "value": len(risk_df), "delta": None},
        {"label": "Tracked rejections", "value": len(profile.rejected_product_ids), "delta": None},
    ]

    summary_bullets = []
    if recommendations:
        summary_bullets.append(f"Current top recommendation is {recommendations[0].product.name} with KeepScore {recommendations[0].keep_score:.1f}.")
    if profile.transition_reason:
        summary_bullets.append(f"Latest state transition: {profile.transition_reason}.")
    if not signal_df.empty:
        summary_bullets.append(f"Top shopper mission in current recommendations: {signal_df.iloc[0]['use_case']}.")
    if not risk_df.empty:
        summary_bullets.append(f"Highest return-risk item in the catalog snapshot is {risk_df.iloc[0]['product_name']}.")

    return metrics, summary_bullets, product_df, risk_df, signal_df


def _answer_dashboard_prompt(prompt: str, result: EngineResult | None) -> dict:
    prompt_l = prompt.lower()
    metrics, summary_bullets, product_df, risk_df, signal_df = _dashboard_snapshot(result)
    bullets: list[str] = []
    answer = ""

    if "intent" in prompt_l or "mission" in prompt_l:
        if signal_df.empty:
            answer = "The current catalog snapshot does not show a strong repeated shopper mission yet."
        else:
            top = signal_df.iloc[0]
            answer = f"The strongest active shopper mission in the current recommendation set is {top['use_case']}."
            bullets = [f"{row.use_case}: {int(row.mentions)} mentions" for row in signal_df.itertuples(index=False)]
    elif "risk" in prompt_l or "watchlist" in prompt_l:
        if risk_df.empty:
            answer = "No products are currently flagged into the risk watchlist from the active scoring snapshot."
        else:
            answer = f"{risk_df.iloc[0]['product_name']} is the clearest risk-watchlist product right now because it combines higher return risk with weaker keep score."
            bullets = [
                f"{row.product_name}: return risk {row.avg_risk:.0%}, KeepScore {row.avg_keep_score:.1f}"
                for row in risk_df.head(5).itertuples(index=False)
            ]
    elif "reliable" in prompt_l or "confidence" in prompt_l:
        answer = "Recommendation reliability is strongest when the chat state is specific about category, budget, and at least one performance goal such as softness or support."
        profile = st.session_state.profile
        bullets = [
            f"Category locked: {'yes' if profile.category else 'no'}",
            f"Budget locked: {'yes' if profile.budget_max else 'no'}",
            f"Active priority count: {sum(1 for v in profile.objectives.values() if v > 0)}",
            f"Current transition: {profile.transition_label}",
        ]
    else:
        answer = summary_bullets[0] if summary_bullets else "The dashboard has a current snapshot ready, but no specific question matched a prepared analysis route."
        bullets = summary_bullets[1:]

    return {
        "answer": answer,
        "bullets": bullets,
        "tool_trace": {
            "prompt": prompt,
            "snapshot_metrics": metrics,
            "top_rows_preview": product_df.head(5).to_dict(orient="records"),
        },
    }


def _render_starter_prompts() -> None:
    cols = st.columns(len(STARTER_PROMPTS))
    for idx, prompt in enumerate(STARTER_PROMPTS):
        with cols[idx]:
            if st.button(prompt, key=f"starter_{idx}", use_container_width=True):
                _run_turn(prompt)
                st.rerun()


def _render_dashboard_prompts(result: EngineResult | None) -> None:
    cols = st.columns(len(DASHBOARD_PROMPTS))
    for idx, prompt in enumerate(DASHBOARD_PROMPTS):
        with cols[idx]:
            if st.button(prompt, key=f"dash_prompt_{idx}", use_container_width=True):
                st.session_state.dashboard_answer = _answer_dashboard_prompt(prompt, result)
                st.rerun()


def _render_dashboard_charts(product_df: pd.DataFrame, risk_df: pd.DataFrame, signal_df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Most engaged products")
        if not product_df.empty:
            st.bar_chart(product_df.set_index("product_name")["clicks"])
    with c2:
        st.markdown("#### Risk watchlist")
        if not risk_df.empty:
            scatter = risk_df.rename(columns={"avg_risk": "x", "avg_keep_score": "y"})
            st.scatter_chart(scatter, x="x", y="y", size="impressions")

    if not signal_df.empty:
        st.markdown("#### Top shopper missions")
        st.bar_chart(signal_df.set_index("use_case")["mentions"])


def _render_shelf_grid(shelf_name: str, recs: list[Recommendation], *, limit: int) -> None:
    display_recs = recs[:limit]
    for start in range(0, len(display_recs), 3):
        cols = st.columns(3)
        for col, rec in zip(cols, display_recs[start : start + 3]):
            with col:
                render_shelf_card(rec, shelf_name)
                if st.button(f"Explore {rec.product.product_id}", key=f"shelf_{shelf_name}_{rec.product.product_id}", use_container_width=True):
                    _run_turn(f"Show me options similar to {rec.product.name}.")
                    st.rerun()


def _render_shop_tab(result: EngineResult | None) -> None:
    st.markdown("### Quick starts")
    _render_starter_prompts()

    left, right = st.columns([1.05, 1.05], gap="large")
    with left:
        st.markdown("### Chat with the assistant")
        if not st.session_state.chat_messages:
            st.info(
                "Start with your use case, budget, category, color, or width. "
                "Example: 'I need black walking shoes under $130.'"
            )
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        user_input = st.chat_input("Tell the assistant what you need")
        if user_input and user_input.strip():
            _run_turn(user_input)
            st.rerun()

    with right:
        st.markdown("### Top pick")
        if result and result.explanation:
            st.success(result.explanation)
        if result and result.image_analysis:
            if st.session_state.uploaded_shoe_image:
                st.image(
                    st.session_state.uploaded_shoe_image["bytes"],
                    caption=f"Uploaded shoe: {st.session_state.uploaded_shoe_image['filename']}",
                    use_container_width=True,
                )
            render_image_analysis(result.image_analysis)
        if result and result.llm_model:
            st.caption(f"Response model: {result.llm_model}")
        elif result and result.llm_error:
            st.caption("Response model unavailable, using heuristic fallback.")
        with st.expander("Current shopping context", expanded=False):
            render_profile_summary(st.session_state.profile)
        if result and result.recommendations:
            top_rec = result.recommendations[0]
            render_recommendation(top_rec, 1)
            a, b, c = st.columns(3)
            if a.button(f"Why this one? · {top_rec.product.product_id}", key=f"why_{top_rec.product.product_id}", use_container_width=True):
                _run_turn(f"Why is {top_rec.product.name} a good choice for me?")
                st.rerun()
            if b.button(f"Show similar · {top_rec.product.product_id}", key=f"similar_{top_rec.product.product_id}", use_container_width=True):
                _run_turn(f"Show me options similar to {top_rec.product.name}.")
                st.rerun()
            if c.button(f"Reject this one · {top_rec.product.product_id}", key=f"reject_{top_rec.product.product_id}", use_container_width=True):
                _run_turn("I do not want this current top choice. Show me something else.")
                st.rerun()
        else:
            st.info("No top pick is available yet. Try describing your size, use case, or budget.")

    st.markdown("### Discovery shelves")
    top_product_id = result.recommendations[0].product.product_id if result and result.recommendations else None
    discovery_limit = max(3, st.session_state.top_k_ui - 1)
    ranked_discovery = [rec for rec in (result.recommendations[1 : 1 + discovery_limit] if result else []) if rec.product.product_id != top_product_id]
    if ranked_discovery:
        st.markdown("#### More matches for you")
        _render_shelf_grid("More matches for you", ranked_discovery, limit=discovery_limit)

    for shelf_name, recs in (result.shelves.items() if result else []):
        filtered_recs = []
        seen_product_ids = set()
        for rec in recs:
            if rec.product.product_id == top_product_id or rec.product.product_id in seen_product_ids:
                continue
            filtered_recs.append(rec)
            seen_product_ids.add(rec.product.product_id)
        if filtered_recs:
            st.markdown(f"#### {shelf_name}")
            _render_shelf_grid(shelf_name, filtered_recs, limit=3)


def _render_dashboard_tab(result: EngineResult | None) -> None:
    st.markdown("### NB DTC team workspace")
    st.caption(
        "This dashboard tracks shopper intent, risk, and recommendation reliability. "
        "The dashboard assistant summarizes which signals are strong, weak, or contradictory."
    )
    metrics, summary_bullets, product_df, risk_df, signal_df = _dashboard_snapshot(result)
    render_dashboard_snapshot(metrics, summary_bullets)
    st.markdown("### Ask the dashboard assistant")
    _render_dashboard_prompts(result)
    dash_q = st.chat_input(
        "Ask about shopper intents, risk, reliability, width demand, or budget behavior",
        key="dashboard_chat",
    )
    if dash_q and dash_q.strip():
        st.session_state.dashboard_answer = _answer_dashboard_prompt(dash_q, result)
        st.rerun()
    render_dashboard_answer(st.session_state.dashboard_answer)
    st.markdown("### Charts")
    _render_dashboard_charts(product_df, risk_df, signal_df)


def _render_trace_tab(result: EngineResult | None) -> None:
    if result is None:
        st.info("No conversation yet. Start a chat or upload a shoe image to see the grounded trace.")
        return
    st.markdown("### Latest assistant answer")
    st.info(result.explanation)
    with st.expander("Current shopping context", expanded=True):
        render_profile_summary(result.profile)
    render_evidence(result.evidence)
    st.markdown("### Decision trace")
    st.json(
        {
            "active_user_id": st.session_state.active_user_id,
            "llm_model": result.llm_model,
            "llm_error": result.llm_error,
            "memory_snippets": result.memory_snippets,
            "image_description": result.image_description,
            "image_search_query": result.image_search_query,
            "image_analysis": result.image_analysis,
            "parsed_turn": result.parsed_turn.__dict__,
            "why_changed": result.why_changed,
            "top_recommendations": [
                {
                    "product_id": rec.product.product_id,
                    "name": rec.product.name,
                    "keep_score": rec.keep_score,
                    "fit_confidence": rec.fit_confidence,
                    "return_risk": rec.return_risk,
                    "score_breakdown": rec.score_breakdown,
                }
                for rec in result.recommendations[:3]
            ],
        }
    )


def run_app() -> None:
    st.set_page_config(page_title="KeepScore Robust Merge", layout="wide")
    apply_theme()
    _ensure_state()

    with st.sidebar:
        st.header("Live shopping controls")
        user_value = st.text_input("Shopper ID", key="user_id_input", help="Chat history and profile memory are stored per shopper ID.")
        normalized = normalize_user_id(user_value)
        if normalized != st.session_state.active_user_id:
            _load_user_into_state(normalized)
            st.session_state.budget_value = int(st.session_state.profile.budget_max or 130)
            st.session_state.pending_budget_widget_sync = True
            st.rerun()
        st.caption(f"Persistent record: data/users/{st.session_state.active_user_id}.json")
        uploaded_file = st.file_uploader("Upload a shoe image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            st.image(uploaded_bytes, caption=uploaded_file.name, use_container_width=True)
            if st.button("Analyze uploaded shoe", use_container_width=True):
                _run_image_turn(uploaded_bytes, uploaded_file.name)
                st.rerun()
        _apply_pending_budget_widget_sync()
        c1, c2, c3 = st.columns([1, 2, 1])
        if c1.button("− $10", use_container_width=True):
            st.session_state.budget_value = max(40, st.session_state.budget_value - 10)
            st.session_state.pending_budget_widget_sync = True
            _sync_budget_control()
            st.rerun()
        c2.number_input(
            "Budget target",
            min_value=40,
            max_value=300,
            step=5,
            key="budget_widget",
            on_change=_on_budget_widget_change,
        )
        if c3.button("+ $10", use_container_width=True):
            st.session_state.budget_value = min(300, st.session_state.budget_value + 10)
            st.session_state.pending_budget_widget_sync = True
            _sync_budget_control()
            st.rerun()
        st.slider("Number of products", 1, 10, key="top_k_ui")
        if st.button("Reset conversation", use_container_width=True):
            reset_user_record(st.session_state.active_user_id)
            _load_user_into_state(st.session_state.active_user_id)
            st.session_state.budget_value = 130
            st.session_state.pending_budget_widget_sync = True
            st.session_state.uploaded_shoe_image = None
            st.session_state.just_reset = True
            st.rerun()

    if not st.session_state.just_reset:
        _sync_budget_control()
    result = _refresh_if_needed()
    if st.session_state.just_reset:
        st.session_state.just_reset = False
    render_nav()
    render_help_strip()
    if st.session_state.get("ui_notice"):
        st.info(st.session_state.ui_notice)
        st.session_state.ui_notice = None

    shop_tab, dashboard_tab, trace_tab = st.tabs(["Shop", "NB DTC Dashboard", "Grounded trace"])
    with shop_tab:
        _render_shop_tab(result)
    with dashboard_tab:
        _render_dashboard_tab(result)
    with trace_tab:
        _render_trace_tab(result)
