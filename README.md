# KeepScore Robust Merge

A merged and more robust shopping intelligence demo that combines:

- **stateful chat and preference carryover** inspired by the first project
- **storefront shelves** inspired by the second project
- **separate score families** for personalized recommendations, trending, new launches, and high KeepScore
- **retrieval before explanation**
- **lightweight dashboard analytics**
- **Ollama-backed natural-language responses** using `gpt-oss:120b-cloud`

## Features

- Chat-aware shopper memory with add / refine / override behavior
- Recommended Matches, Trending Shoes, New Launch, and High KeepScore shelves
- Evidence retrieval from structured review snippets before explanation is generated
- Transparent score breakdown and "why changed" notes
- Streamlit UI with Home, Chat, and Dashboard pages
- JSON-backed product, review, and shopper-memory storage
- Ranked recommendations stay heuristic and deterministic, while the response layer can use Ollama for more natural suggestions grounded in review evidence and prior user history
- Shoe-image upload for visual description, likely matches, and related suggestions

## Quick start

```bash
cd merged_keepscore_robust
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Ollama

The app expects an Ollama-compatible chat endpoint and defaults to:

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:120b-cloud
```

If the model is unavailable, recommendation ranking still works and the UI falls back to a heuristic explanation.

## Demo accounts

- Admin: `admin_demo` / `AdminDemo!123`
- User: `user_demo` / `UserDemo!123`
- Guest: no login required

Only admins can view the `NB DTC Dashboard`. Logged-in users and guests can still use the shopping assistant and grounded trace views.

## Agents and communication

This project is not a multi-agent system.

- Active app-side agent count: `1`
- Optional external model endpoint: `1` Ollama model connection
- Internal recommendation/scoring workers: `0` separate agents, these are normal Python modules in the same process

How communication works:

1. The Streamlit UI collects chat input, login state, budget changes, and optional image uploads.
2. The UI sends that request to `KeepScoreEngine` in `src/keepscore_robust/engine.py`.
3. The engine calls parsing, state, retrieval, and scoring modules directly in-process.
4. The engine reads and writes shopper memory from JSON files in `data/users/`.
5. For natural-language explanation, the engine optionally sends a request to the Ollama endpoint configured by `OLLAMA_HOST` and `OLLAMA_MODEL`.
6. The Ollama response, or the local heuristic fallback, is returned to the UI and shown to the user.

So in practice, there is one application orchestrator inside the Streamlit app, and one optional external LLM endpoint. The Python modules communicate by direct function calls and shared return values, not by agent-to-agent messaging.

## Data layout

- `data/products.json`: catalog seed data
- `data/reviews.json`: review evidence seed data
- `data/users/<shopper-id>.json`: persisted shopper profile, prior chats, and turn summaries for later personalization

## Image upload

Upload a shoe photo from the sidebar and the app will:

- analyze the image with Ollama when the configured model supports image input
- fall back to heuristic color and filename analysis when vision is unavailable
- generate a short shoe description
- turn that visual read into catalog matches and related suggestions

## Smoke test

```bash
PYTHONPATH=src python -m tests.smoke_test
```

## Layout

```text
merged_keepscore_robust/
├── app.py
├── requirements.txt
├── src/keepscore_robust/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── parsing.py
│   ├── state.py
│   ├── retrieval.py
│   ├── scoring.py
│   ├── engine.py
│   └── ui.py
└── tests/
    └── smoke_test.py
```
