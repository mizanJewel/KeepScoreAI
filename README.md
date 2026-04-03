# KeepScore Robust Merge

`KeepScore Robust Merge` is a Streamlit shopping intelligence demo for shoe discovery. It combines stateful chat, adaptive recommendation scoring, review-grounded explanation, image-assisted matching, role-based dashboard access, and a lightweight multi-agent orchestration layer.

The app is designed around one central idea:

- use structured product and review data to rank shoes deterministically
- use session memory to make the ranking more personalized over time
- use an optional Ollama model to explain the ranking naturally
- expose the reasoning trace so the system stays inspectable

## What the app does

- lets a shopper describe what they want in natural language
- remembers constraints and preferences across turns
- ranks shoes with an adaptive `KeepScore`
- retrieves review evidence before generating an explanation
- supports uploaded shoe images for visual matching
- persists shopper history to JSON
- provides an admin-only NB DTC Dashboard with catalog and commerce-style analytics

## Quick start

```bash
cd merged_keepscore_robust
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Ollama

The app expects an Ollama-compatible endpoint for text generation and optional image analysis.

Default configuration:

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gpt-oss:120b-cloud
```

If Ollama is unavailable, recommendation ranking still works. The app falls back to a local heuristic explanation path, and image upload falls back to heuristic image analysis.

## Demo accounts

- Admin: `admin_demo` / `AdminDemo!123`
- User: `user_demo` / `UserDemo!123`
- Guest: no login required

Access rules:

- guests can browse the storefront and grounded trace
- logged-in users can browse the storefront and grounded trace
- only admins can access `NB DTC Dashboard`

## Architecture

The architecture has four major layers:

1. `UI layer`
2. `Coordinator and agent layer`
3. `Tool and scoring layer`
4. `Data and memory layer`

### 1. UI layer

The Streamlit UI lives in:

- `app.py`
- `src/keepscore_robust/ui.py`
- `src/keepscore_robust/components.py`
- `src/keepscore_robust/theme.py`

Responsibilities:

- login/logout and guest mode
- chat input
- image upload
- dashboard rendering
- grounded trace rendering
- session-state handling for the current browser session

The UI does not compute ranking logic itself. It delegates work to `KeepScoreEngine` and only renders the returned result.

### 2. Coordinator and agent layer

The orchestrator is:

- `src/keepscore_robust/engine.py`

The coordinator owns the end-to-end flow for:

- chat turns
- image turns
- refreshes
- explanation generation
- trace assembly

The system now uses a lightweight in-process multi-agent pattern. These are not separate processes or remote workers. They are specialized Python components coordinated by `KeepScoreEngine`.

Current agents:

- `parser-agent`
- `profile-agent`
- `retrieval-agent`
- `recommendation-agent`
- `evidence-agent`
- `explanation-agent`
- `vision-agent` for image uploads

These agents live in:

- `src/keepscore_robust/agents.py`

What each one does:

- `parser-agent`
  Parses raw user text into a structured `ParsedTurn`
- `profile-agent`
  Applies the parsed turn to the current `ShopperProfile`
- `retrieval-agent`
  Pulls candidate shoes from the catalog
- `recommendation-agent`
  Scores and ranks candidate shoes
- `evidence-agent`
  Retrieves review snippets for top-ranked products
- `explanation-agent`
  Produces the final natural-language response
- `vision-agent`
  Analyzes uploaded shoe images and converts them into searchable attributes

### 3. Tool and scoring layer

The agents do not directly hardcode all logic. They call shared tools through a lightweight MCP-style registry:

- `src/keepscore_robust/mcp.py`

This registry behaves like a local tool protocol:

- a tool is registered by name
- agents call tools by name with structured inputs
- every tool call is traced
- the final trace is shown in the `Grounded trace` tab

Registered tool families include:

- parsing
- profile updates
- image analysis
- candidate retrieval
- recommendation scoring
- review evidence retrieval
- shelf generation
- explanation generation

This is not full external MCP server infrastructure. It is an MCP-style internal protocol: named tools, structured inputs, traceable calls, shared registry.

### 4. Data and memory layer

The app is JSON-backed.

Main data sources:

- `data/products.json`
- `data/reviews.json`
- `data/accounts.json`
- `data/users/<shopper-id>.json`

Loader and persistence modules:

- `src/keepscore_robust/data.py`
- `src/keepscore_robust/memory.py`
- `src/keepscore_robust/auth.py`

This keeps the demo portable and transparent: you can inspect and edit the catalog, reviews, users, and accounts without a database.

## End-to-end request flow

### Text conversation flow

1. The shopper sends a message in the chat UI.
2. `ui.py` sends that message plus current profile and memory context to `KeepScoreEngine`.
3. `parser-agent` calls the parsing tool to build a `ParsedTurn`.
4. `profile-agent` updates the `ShopperProfile`.
5. `retrieval-agent` gets candidate products from the catalog.
6. `recommendation-agent` scores candidates with adaptive `KeepScore`.
7. `evidence-agent` retrieves review snippets for the top products.
8. `shelf` construction runs for discovery sections like Trending, New Launch, and High KeepScore.
9. `explanation-agent` optionally calls Ollama for a natural-language answer.
10. The engine returns an `EngineResult` to the UI.
11. The UI renders the answer, recommendations, shelves, and trace.
12. The updated shopper profile and chat log are written to `data/users/<shopper-id>.json`.

### Image upload flow

1. The shopper uploads a shoe image in the chat area.
2. `vision-agent` analyzes the image.
3. If the model supports image input, the app uses Ollama-assisted image analysis.
4. If not, the app falls back to heuristic image analysis using image color and filename hints.
5. The image analysis becomes a search-style query.
6. The rest of the flow then follows the same path as a normal text turn.

## Protocols and communication

There are three communication styles in the app.

### UI to coordinator

This is direct Python invocation inside the same Streamlit process.

- the UI calls engine methods
- the engine returns an `EngineResult`

### Agent to tool

This is the MCP-style internal protocol.

Properties:

- tool name
- structured arguments
- trace entry with calling agent
- returned Python object

This protocol is implemented locally through `MCPToolRegistry`.

### Coordinator to Ollama

This is the only network-style boundary in the app.

- module: `src/keepscore_robust/llm.py`
- protocol: HTTP JSON request to Ollama-compatible `/api/chat`
- usage:
  - natural-language explanation
  - optional image analysis

If the external model call fails, the app falls back to heuristic behavior.

## Adaptive KeepScore

The ranking logic lives in:

- `src/keepscore_robust/scoring.py`

KeepScore is no longer fully fixed-weight. It adapts based on the current session.

### Core scoring signals

Each recommendation still uses the same major inputs:

- `preference_match`
- `fit_confidence`
- `keep_probability`
- `budget_compatibility`
- `quality_score`
- `reliability_score`
- `expected_utility`
- `return_risk`

### Why it is adaptive now

Before, these components had fixed weights.

Now the app computes adaptive context from the shopper profile:

- profile specificity
- history depth
- rejection pressure
- objective focus

From those signals it chooses dynamic weights for:

- preference match
- fit confidence
- keep probability
- budget compatibility
- quality
- reliability
- expected utility
- return-risk penalty

This means:

- early exploratory sessions lean more on broad quality and utility
- specific sessions lean more on fit and preference match
- rejection-heavy sessions apply a stronger return-risk penalty
- longer sessions move the system into a stronger refinement mode

The adaptive state is stored on the profile in `adaptive_state` and is visible in the trace and recommendation score breakdown.

## Recommendation data model

Important core models live in:

- `src/keepscore_robust/models.py`

Main objects:

- `Product`
  Catalog item with merchandising, comfort, behavioral, and quality fields
- `ReviewSnippet`
  Short review evidence item with tags and sentiment
- `ParsedTurn`
  Structured interpretation of a single user message
- `ShopperProfile`
  Persistent shopper state across turns
- `Recommendation`
  Scored product with reasons, cautions, and score breakdown
- `EngineResult`
  Full result object returned from the engine, including traces

### Key `Product` fields

The current catalog includes:

- identity fields
  - `product_id`, `name`, `brand`
- fit and merchandising fields
  - `category`, `gender`, `colors`, `widths`
- price fields
  - `price`, `sale_price`
- comfort and product attributes
  - `cushioning`, `support`, `premium_level`, `weight_grams`, `waterproof`, `stability`
- quality and sentiment proxies
  - `comfort_rating`, `avg_rating`, `quality_score`, `reliability_score`
- demand and commerce behavior proxies
  - `review_count`, `stock`, `return_rate`, `conversion_rate`, `click_through_rate`, `add_to_cart_rate`, `wishlist_count`, `trend_velocity`

### Key `ShopperProfile` fields

The profile tracks:

- conversation history
- budget
- category
- gender
- color preferences
- width need
- objective weights
  - softness
  - premium
  - lightweight
  - support
- rejected product IDs
- last recommended IDs
- transition label and reason
- adaptive scoring state

## Data files

### `data/products.json`

The structured product catalog.

Used by:

- candidate retrieval
- scoring
- shelves
- dashboard analytics

### `data/reviews.json`

Short review snippets used for RAG-like evidence retrieval.

Used by:

- explanation grounding
- top-product evidence display

### `data/accounts.json`

Static demo account list with:

- username
- display name
- role
- salt
- password hash

### `data/users/<shopper-id>.json`

Per-shopper persistent memory.

Stores:

- the current profile
- prior chat messages
- prior turn summaries
- update timestamp

## Auth and role model

Auth lives in:

- `src/keepscore_robust/auth.py`

Roles:

- `guest`
- `user`
- `admin`

Permissions:

- `guest`: shopping chat, image upload, shelves, grounded trace
- `user`: same as guest, but with a named persistent account
- `admin`: all of the above plus `NB DTC Dashboard`

## Dashboard data assumptions

The dashboard is intentionally honest about its data.

Some charts are based on direct catalog fields, such as:

- click-through rate
- add-to-cart rate
- conversion rate
- return-rate proxies
- wishlist counts

Some dashboard metrics are derived proxies, not true transactional finance data. For example:

- revenue proxy
- funnel counts
- purchases from impressions

These are generated from catalog behavior fields so the dashboard can demonstrate analytics structure without requiring a live warehouse or event pipeline.

## Image analysis

Image support lives in:

- `src/keepscore_robust/image_analysis.py`

Behavior:

- tries Ollama-assisted image analysis first
- falls back to heuristic image feature extraction if needed
- produces:
  - description
  - category guess
  - color guess
  - search query
  - related suggestions

## Grounded trace

The `Grounded trace` tab is the main observability surface for the app.

It shows:

- parsed turn data
- why the profile changed
- top recommendations
- recommendation score breakdown
- adaptive scoring state
- review evidence
- agent trace
- MCP tool-call trace
- image analysis trace when relevant

This makes the system inspectable instead of behaving like a black box.

## Smoke test

```bash
PYTHONPATH=src python -m tests.smoke_test
```

The smoke test currently verifies:

- demo login works
- recommendations are generated
- gender preferences persist
- adaptive scoring state exists
- agent and MCP traces exist
- image upload flow returns an analysis result

## Project layout

```text
merged_keepscore_robust/
├── app.py
├── data/
│   ├── accounts.json
│   ├── products.json
│   ├── reviews.json
│   └── users/
├── requirements.txt
├── src/keepscore_robust/
│   ├── __init__.py
│   ├── agents.py
│   ├── auth.py
│   ├── components.py
│   ├── data.py
│   ├── engine.py
│   ├── image_analysis.py
│   ├── llm.py
│   ├── mcp.py
│   ├── memory.py
│   ├── models.py
│   ├── parsing.py
│   ├── retrieval.py
│   ├── scoring.py
│   ├── state.py
│   ├── theme.py
│   └── ui.py
└── tests/
    └── smoke_test.py
```

## Current limits

- the MCP layer is local, not a remote server mesh
- the agents are in-process components, not distributed workers
- the dashboard uses proxy analytics rather than real transactional event streams
- image understanding depends on model support and otherwise uses fallback heuristics
- the recommendation engine is still structured and rules-driven rather than end-to-end learned

## Future directions

- add a dedicated `dashboard-agent`
- connect the MCP layer to external services instead of only local tools
- replace proxy analytics with real event or warehouse data
- learn adaptive weights from historical outcomes instead of handcrafted session logic
- add richer catalog embeddings and semantic retrieval
