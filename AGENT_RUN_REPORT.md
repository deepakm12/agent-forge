# Agent Run Report

This document describes the agent's execution architecture, documents key design decisions and their trade-offs, and outlines the evaluation framework used to verify correctness across representative test scenarios.

---

## Architecture Overview

The system is built as a two-phase LangGraph pipeline. The clarification phase surfaces questions to the user before any analysis runs; the analysis phase executes once answers are provided.

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                Orchestrator (LangGraph StateGraph)           │
│                                                              │
│  Phase 1: [ingest] → [clarify] → END (awaits user input)    │
│                                                              │
│  Phase 2: [plan] → [dispatch] → [reflect] ─→ [plan]         │
│                                            └─→ [respond]    │
└──────────────────────────┬──────────────────────────────────┘
          ┌────────────────┼─────────────────┐
          ▼                ▼                 ▼
      Stats Agent      Viz Agent       Insight Agent
     (pandas/scipy)  (matplotlib)    (GPT-4o prose)
          └────────────────┼─────────────────┘
                           ▼
                Code Executor (subprocess, 30s timeout)
```

**Execution flow:** `ingest → clarify → [user answers] → plan → dispatch → reflect → respond`

The reflect node performs a self-critique pass: if it determines the analysis is incomplete or missing a dimension, it loops back to plan for up to two additional iterations before forcing a final response.

---

## Design Decisions & Trade-offs

**LangGraph StateGraph over a custom agent loop**
LangGraph provides explicit, inspectable state transitions and conditional edges that make the replan loop straightforward to express and reason about. A hand-rolled loop would work but would require reimplementing state management, streaming, and branch logic from scratch. The main trade-off is a framework dependency and an initial learning curve.

**Two-phase execution (clarify → analyze)**
Returning the clarification questions to the UI and waiting for user input — rather than making assumptions upfront — produces materially better analysis results. The trade-off is two separate graph invocations per session rather than one, which adds a small amount of overhead but is worth the quality improvement.

**Pydantic DTOs for all LLM outputs**
Every response from the LLM is parsed into a typed Pydantic model before it touches application state. This means validation failures are loud and immediate rather than silent and downstream. Using `beta.chat.completions.parse` with structured output mode reinforces this at the API level.

**Sandboxed subprocess for code execution**
Generated code runs in an isolated subprocess. Network-accessing imports (`requests`, `urllib`, `socket`, etc.) and filesystem-write operations are blocked at import time. A 30-second hard timeout prevents infinite loops from hanging the UI. The main trade-off is that code blocks within a session cannot share in-memory state — each block starts fresh.

**SQLite for session memory**
SQLite gives zero-infrastructure persistence: conversation state survives browser refreshes and server restarts with no external dependencies. This is the right trade-off for a single-user deployment. For multi-tenant production use, this would need to be replaced with a proper database and per-user isolation.

**Tenacity retries scoped to transient errors**
Only `RateLimitError`, `APITimeoutError`, `APIConnectionError`, and `InternalServerError` are retried. Authentication errors, bad request errors, and validation failures are allowed to fail immediately so that configuration problems surface quickly rather than being masked by retry loops.

**structlog with JSON output**
All log entries are structured JSON keyed on `session_id` and `node`. This makes it practical to filter traces per session or per pipeline node, and the output is directly ingestible by log aggregation tools without further parsing.

---

## Test Scenarios

Five scenarios cover the range of analysis tasks the agent is expected to handle. All integration scenarios require a live `OPENAI_API_KEY`.

| # | Scenario | Dataset | What is verified |
|---|---|---|---|
| 1 | Basic descriptive stats | `sales_data.csv` (20 rows) | Stats agent runs, insight narrative produced, regional breakdown mentioned |
| 2 | Correlation analysis | `marketing_metrics.csv` (10 rows) | Stats agent computes correlations, insight agent summarizes relationship strength |
| 3 | Time-series trend | `monthly_revenue.csv` (16 months) | Trend direction identified, growth percentage calculated, seasonality noted |
| 4 | Outlier detection | `sensor_readings.csv` (outliers at 98.5°C and −15.2°C) | Anomalies flagged, z-score analysis present |
| 5 | Missing data handling | `incomplete_survey.csv` (intentional nulls) | Null percentage reported per column, data quality summary included |

Run integration tests:

```bash
PYTHONPATH=. pytest tests/test_scenarios.py -v -m integration
```

Traces are saved to `tests/traces/<scenario_id>_trace.json` after each run.

---

## Evaluation Criteria

Each scenario is scored against the following checks:

| Check | Description |
|---|---|
| `has_final_response` | Agent produced a non-empty final response |
| `final_response_min_chars` | Response is at least 200 characters |
| `has_stats` | Stats specialist ran at least one subtask |
| `has_insight` | Insight specialist ran at least one subtask |
| Scenario-specific checks | e.g., `mentions_region`, `mentions_null`, `mentions_correlation` |

A scenario passes only when all applicable checks return `true`. The evaluation runner writes a `passed` boolean into each trace for quick inspection.

---

## Sample Trace Structure

The following shows the shape of a full evaluation trace. Actual values are populated at runtime with a real API key.

```json
{
  "scenario_id": "scenario_1",
  "name": "Basic descriptive stats",
  "session_id": "3f8a1c2e-...",
  "elapsed_s": 45.2,
  "evaluation": {
    "has_final_response": true,
    "final_response_min_chars": true,
    "has_stats": true,
    "has_insight": true,
    "mentions_region": true,
    "passed": true
  },
  "final_response_preview": "## Sales Data Analysis\n\nThe dataset covers 10 days of sales across 4 regions...",
  "subtask_results": [
    {
      "id": "1",
      "agent": "stats",
      "success": true,
      "output_preview": "Revenue by Region:\n  East: $9,470.00 (26.7%)\n  North: $7,961.25 (22.4%)..."
    },
    {
      "id": "2",
      "agent": "viz",
      "success": true,
      "output_preview": "Saved: revenue_by_region.png"
    },
    {
      "id": "3",
      "agent": "insight",
      "success": true,
      "output_preview": "Key Insight: East region leads with 26.7% of total revenue..."
    }
  ]
}
```
