from __future__ import annotations

from io import BytesIO

from PIL import Image

from keepscore_robust.engine import KeepScoreEngine
from keepscore_robust.memory import load_user_record, profile_from_record, reset_user_record, save_user_record


def main() -> None:
    engine = KeepScoreEngine()
    profile = engine.new_profile()

    first = engine.process_turn("I need men black running shoes under $150", profile)
    assert first.recommendations, "expected recommendations on first turn"
    assert first.profile.gender == "men", "gender should be stored on the shopper profile"
    assert all(rec.product.gender in {"men", "unisex"} for rec in first.recommendations), "main recommendations should respect requested gender"
    assert all(rec.product.gender in {"men", "unisex"} for rec in first.shelves["Recommended Matches"]), "recommended shelf should respect requested gender"
    assert all(rec.product.gender in {"men", "unisex"} for rec in first.shelves["Trending Shoes"]), "trending shelf should respect requested gender"
    first_top = first.recommendations[0].product.name

    second = engine.process_turn("more soft one", first.profile)
    assert second.recommendations, "expected recommendations on second turn"
    second_top = second.recommendations[0].product.name

    assert second.profile.objectives["softness"] > 0, "softness objective should be active"
    assert "Trending Shoes" in second.shelves, "expected storefront shelves"
    assert second.evidence, "expected evidence retrieval before explanation"
    assert all(rec.product.gender in {"men", "unisex"} for rec in second.shelves["High KeepScore"]), "high keep shelf should still respect gender"
    assert second.llm_model in {None, "gpt-oss:120b-cloud"}, "unexpected model metadata"

    reset_user_record("smoke-test")
    save_user_record(
        "smoke-test",
        second.profile,
        [
            {"role": "user", "content": "I need men black running shoes under $150"},
            {"role": "assistant", "content": second.explanation},
        ],
        turn_summary={"summary": second.explanation},
    )
    record = load_user_record("smoke-test")
    restored = profile_from_record(record)
    assert restored.gender == "men", "stored profile should persist gender"
    assert record["chat_messages"], "stored chat history should persist"

    image = Image.new("RGB", (48, 24), color=(20, 20, 20))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_result = engine.process_uploaded_image(buffer.getvalue(), "black-running-shoe.png", second.profile)
    assert image_result.image_analysis, "image upload should produce analysis"
    assert image_result.image_description, "image upload should produce a description"
    assert image_result.recommendations, "image upload should still produce matches"

    print("First top recommendation:", first_top)
    print("Second top recommendation:", second_top)
    print("Stored gender:", second.profile.gender)
    print("Transition:", second.profile.transition_label, "-", second.profile.transition_reason)
    print("Why changed:", second.why_changed)
    print("Explanation:", second.explanation)
    print("LLM model:", second.llm_model or "heuristic fallback")
    print("Image description:", image_result.image_description)


if __name__ == "__main__":
    main()
