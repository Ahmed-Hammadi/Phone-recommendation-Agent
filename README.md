## Phone Recommendation Agent

An AI co-pilot that narrows phone choices in seconds. The backend runs a Model Context Protocol (MCP) toolchain orchestrated by a GPT-5-Codex planner, while the new glassmorphism React interface delivers a premium experience with session memory, reasoning transparency, and live Tunisian pricing.

---

### ✨ Highlights

- **Agentic reasoning with memory**: The orchestrator fuses LLM planning with heuristics and persists conversation context, so follow-up questions automatically reuse the previously discussed device.
- **Real-time market data**: A concurrent scraper (requests + cloudscraper + BeautifulSoup) queries Tunisianet, Zoom, and Spacenet in parallel and normalises pricing metadata for quick comparisons.
- **Rich insights**: Tools tap into specs, Reddit sentiment, alternative recommendations, and requirement validation to surface actionable talking points.
- **Glass UI**: A Vite + React 18 + Tailwind frontend wraps everything in a frosted-glass dashboard, complete with chat history, reasoning toggles, and a playful robot badge.
- **Standards-based integrations**: JSON-RPC 2.0 and MCP-compatible tool schemas make it easy to embed the agent in external products.

---

### 🧱 Architecture Overview

```
┌─────────────────────────────────┐
│           React Frontend        │
│  - ChatPane + Reasoning Panel   │
│  - Tailwind / Framer Motion     │
│  - React Query session client   │
└───────────────▲────────────────┘
								│ REST / JSON-RPC
┌───────────────┴────────────────┐
│         FastAPI MCP Server     │
│  ToolOrchestrator (LLM+rules)  │
│  Session memory, logging       │
│  JSON-RPC dispatcher           │
└───────────────▲────────────────┘
								│ Tool calls (thread pool)
┌───────────────┴────────────────┐
│           Tool Suite           │
│  phone_specs / specs_analyzer  │
│  price_extractor (concurrent)  │
│  web_scraper (Reddit PRAW)     │
│  sentiment / alternatives      │
│  spec_validator                │
└───────────────┴────────────────┘
								│
				 CSV / Postgres + APIs
```

---

### 🛠 Backend Stack

- **FastAPI + JSON-RPC**: Exposes REST helpers and an `/rpc` endpoint with JSON-RPC 2.0 responses, CORS, and thread-pooled tool execution.
- **ToolOrchestrator**: Combines heuristic intent detection with optional `mistralai` planning, maintains session context, and orchestrates MCP tool calls.
- **Specs data layer**: Defaults to a `pandas` DataFrame sourced from `Backend/database/mobile.csv`, with an optional PostgreSQL + `SQLAlchemy` backend and trigram similarity.
- **Pricing spiders**: Uses `requests`, `cloudscraper`, and `BeautifulSoup` within a `ThreadPoolExecutor` to gather TND pricing signals resiliently.
- **Review ingestion**: Relies on `praw` to authenticate against Reddit and filter submissions/comments for rumour-free sentiment inputs.
- **Similarity heuristics**: Applies `rapidfuzz` scoring and custom numeric parsing helpers to keep fuzzy matches explainable and deterministic.

---

### 🚀 Getting Started

#### Backend

1. **Create and activate a virtual environment**
	 ```bash
	 python -m venv venv
	 source venv/bin/activate  # Windows: venv\Scripts\activate
	 ```

2. **Install dependencies**
	 ```bash
	 pip install -r requirements.txt
	 ```

3. **Configure environment** (Backend/.env)
	 ```env
	 MISTRAL_API_KEY=your_mistral_key               enables LLM synthesis
	 MCP_SERVER_URL=http://127.0.0.1:8000
	 FRONTEND_ORIGIN=http://localhost:5173
	 # Optional PostgreSQL backing store
	 POSTGRES_HOST=...
	 POSTGRES_DB=...
	 POSTGRES_USER=...
	 POSTGRES_PASSWORD=...
	 ```

4. **Run the server**
	 ```bash
	 uvicorn Backend.mcp_server:app --host 0.0.0.0 --port 8000 --reload
	 ```

#### Frontend

1. ```bash
	 cd frontend
	 npm install
	 npm run dev
	 ```
2. Visit `http://localhost:5173` for the live glass UI.

---

### 🧠 Conversational Agent

- **Endpoint**: `POST /agent/chat`
- **Request**:
	```json
	{
		"query": "What is the price of iPhone 14 Pro?",
		"session_id": "optional-stable-id",
		"include_reasoning": true
	}
	```
- **Response**: natural-language answer plus tool reasoning snapshot (when requested).
- Conversations automatically reuse the latest phone context for follow-up questions.

For direct MCP / JSON-RPC usage, send structured calls to `POST /rpc` using methods:

| Method             | Purpose                                          |
|--------------------|--------------------------------------------------|
| `mcp.list_tools`   | Enumerate available tool schemas.                |
| `mcp.call_tool`    | Invoke a single tool with keyword arguments.     |
| `agent.recommend`  | Run the RAG + recommendation pipeline.           |
| `agent.chat`       | Same behaviour as the REST helper.               |
| `system.ping`      | Lightweight health check.                        |

`Backend/mcp_client.py` contains ready-to-use Python wrappers.

---

### 🛠️ Tool Catalogue

- **phone_specs** – `pandas` + `rapidfuzz` fuzzy matching with optional PostgreSQL trigram search through `SQLAlchemy`.
- **web_scraper** – Reddit aggregation powered by `praw`, with rumour filtering and auth diagnostics.
- **specs_analyzer** – heuristically categorises performance/battery/camera/display traits.
- **sentiment_analyzer** – keyword-driven pros/cons extraction over Reddit comment batches.
- **price_extractor** – `requests`/`cloudscraper`/`BeautifulSoup` concurrency for Tunisian retailers with timing metrics.
- **alternative_recommender** – pure-Python spec distance heuristics for budget-aware alternates.
- **spec_validator** – rule-based requirement checks with regex parsing helpers.

All tool results adhere to a shared `ToolOutput` contract for consistent logging.

---

### 🧰 Frontend Features

- **ChatPane** with animated message bubbles, optimistic updates, and reasoning toggle.
- **Session persistence** via localStorage so reloading keeps context intact.
- **Tailwind design tokens** (`src/theme/tokens.ts`) for rapid theming.
- **Robot badge** that anchors the brand identity without GPU overhead.

The application structure mirrors atomic design principles and is ready for component documentation via Storybook (not yet configured).

---

### 📂 Project Structure

```
Backend/
	mcp_server.py           # FastAPI app, JSON-RPC dispatcher, CORS setup
	orchestrator.py         # LLM + heuristic planner with session memory
	tools/                  # All MCP tool implementations
frontend/
	src/
		App.tsx               # Glass dashboard + chat integration
		components/           # Layout, chat, and common UI pieces
		hooks/useAgentChat.ts # React Query powered chat state
		utils/                # env, id, and session helpers
	public/assets/robot-sticker.svg
``` 
### 🔖 License

This repository is licensed under the MIT License. See `LICENSE` for details.