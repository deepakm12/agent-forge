# Agent Forge — Data Analysis Agent

A multi-agent AI system that turns raw CSV files into structured insights. Upload a dataset, answer a few targeted questions, and receive statistical analysis, visualizations, and a narrative summary — all generated and executed autonomously.

---

## Features

- **Conversational clarification** — the agent asks 2–3 focused questions before running analysis, so results are relevant to your actual goals
- **Multi-specialist pipeline** — separate agents handle statistics, visualization, and narrative insight in parallel
- **Self-critique loop** — after analysis, the orchestrator reflects on completeness and replans if needed (up to 2 iterations)
- **Safe code execution** — generated Python runs in a sandboxed subprocess with network and filesystem write access blocked
- **Session persistence** — conversations survive browser refreshes via SQLite-backed memory
- **Structured outputs throughout** — every LLM call uses Pydantic-validated responses; no raw dict parsing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│  CSV upload  │  Chat interface  │  Streaming response panel │
└──────────────────────────┬──────────────────────────────────┘
                           │ in-process
┌──────────────────────────▼──────────────────────────────────┐
│                    Orchestrator Agent                        │
│   LangGraph StateGraph  ·  GPT-4o  ·  Session memory        │
│                                                              │
│  [ingest] → [clarify] → [plan] → [dispatch] → [respond]     │
│                 ↑                    │                       │
│                 └────────[reflect]◄──┘                       │
└──────────────────────────┬──────────────────────────────────┘
          ┌────────────────┼─────────────────┐
          ▼                ▼                 ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  Stats      │  │  Viz        │  │  Insight    │
   │  Agent      │  │  Agent      │  │  Agent      │
   │  (pandas/   │  │  (matplotlib│  │  (narrative │
   │   scipy)    │  │   /seaborn) │  │   summary)  │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          └────────────────┼─────────────────┘
                           ▼
                   ┌───────────────┐
                   │  Code Executor│  (sandboxed subprocess)
                   └───────────────┘
```

### Agent Loop

`ingest → clarify → [user answers] → plan → dispatch → reflect → respond`

| Node | Responsibility |
|---|---|
| **ingest** | Load and profile the CSV — shape, dtypes, null counts, token estimate |
| **clarify** | Generate 2–3 targeted questions about analysis goals |
| **plan** | Chain-of-thought decomposition into subtasks for specialist agents |
| **dispatch** | Route each subtask to the appropriate Stats / Viz / Insight specialist |
| **reflect** | Self-critique: is the analysis complete? Replans if gaps remain (max 2 iterations) |
| **respond** | Assemble and stream the final narrative response |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o (structured output via `beta.chat.completions.parse`) |
| Agent framework | LangGraph StateGraph |
| Data analysis | pandas, scipy |
| Visualization | matplotlib, seaborn |
| Validation | Pydantic v2 (no raw dicts anywhere) |
| Retries | tenacity — on `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError` |
| Session memory | SQLite-backed persistence |
| UI | Streamlit |
| Logging | structlog (structured JSON per node step) |

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key with GPT-4o access
- Docker (optional, for containerized runs)

### Local Setup

```bash
# Clone and install dependencies
git clone https://github.com/your-org/agent-forge.git
cd agent-forge
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and set: OPENAI_API_KEY=sk-...

# Launch the Streamlit UI
PYTHONPATH=. streamlit run src/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Docker

```bash
cp .env.example .env   # add your OPENAI_API_KEY
docker-compose up --build
# Open http://localhost:8501
```

---

## Running Tests

```bash
# Unit tests — no API key required
OPENAI_API_KEY=test-key pytest tests/ -v --ignore=tests/test_scenarios.py

# Integration evaluation — requires a real OPENAI_API_KEY
PYTHONPATH=. pytest tests/test_scenarios.py -v -m integration
```

Per-run traces are written to `tests/traces/<scenario_id>_trace.json`.

---

## Project Structure

```
agent-forge/
├── src/
│   ├── agent/
│   │   ├── orchestrator.py       # LangGraph StateGraph, run_clarify_phase, run_analysis_phase
│   │   ├── state.py              # AgentState definition
│   │   ├── nodes/                # ingest, clarify, plan, dispatch, reflect, respond
│   │   └── specialists/          # stats_agent, viz_agent, insight_agent
│   ├── tools/                    # csv_loader, code_executor, chart_renderer
│   ├── prompts/                  # System prompts, few-shot examples, Jinja2 templates
│   ├── memory/                   # SQLite session store
│   ├── schemas/models.py         # All Pydantic DTOs
│   ├── config.py                 # pydantic-settings
│   ├── logging_config.py         # structlog setup
│   └── app.py                    # Streamlit entry point
├── tests/
│   ├── sample_csvs/              # 5 representative datasets
│   ├── scenarios/scenarios.json  # Test scenarios with expected outcomes
│   ├── traces/                   # Per-run JSON traces (generated at runtime)
│   └── test_scenarios.py         # Evaluation runner
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                # black / ruff / isort / mypy configuration
├── requirements.txt
├── .env.example
└── AGENT_RUN_REPORT.md
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over a custom loop | Explicit state transitions, conditional edges for the replan loop, and built-in streaming — without hand-rolling control flow |
| Pydantic DTOs throughout | Validation at parse time, not at consumption time; type errors surface immediately rather than as runtime surprises |
| Two-phase execution (clarify → analyze) | Separates question-gathering from analysis so the UI can display questions and wait for user input before running anything expensive |
| Sandboxed subprocess for code execution | Security isolation; network and filesystem-write imports are blocked; a 30-second timeout prevents runaway loops |
| SQLite for session memory | Zero-infrastructure persistence across browser sessions; straightforward for single-user deployments |
| Tenacity on transient errors only | Auth errors and bad requests fail fast to surface configuration problems; only genuinely transient failures are retried |
| structlog JSON logging | Machine-readable traces that are easy to filter by `session_id` or node name |
