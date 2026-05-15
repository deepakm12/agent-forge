"""Integration tests that run full end-to-end analysis scenarios and evaluate the results."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

SCENARIOS_PATH = Path("tests/scenarios/scenarios.json")
TRACES_DIR = Path("tests/traces")


def load_scenarios() -> list[dict[str, Any]]:
    with open(SCENARIOS_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


def evaluate_result(state: Any, expected: dict[str, Any]) -> dict[str, Any]:
    final = state.final_response or ""
    results: dict[str, Any] = {}
    results["has_final_response"] = bool(final)
    results["final_response_min_chars"] = len(final) >= expected.get("final_response_min_chars", 0)
    if expected.get("has_stats"):
        results["has_stats"] = any(r.agent_type.value == "stats" for r in state.subtask_results)
    if expected.get("has_insight"):
        results["has_insight"] = any(r.agent_type.value == "insight" for r in state.subtask_results)
    if expected.get("mentions_region"):
        results["mentions_region"] = "region" in final.lower() or any("region" in (r.output or "").lower() for r in state.subtask_results)
    if expected.get("mentions_null"):
        results["mentions_null"] = "null" in final.lower() or "missing" in final.lower()
    results["passed"] = all(v for v in results.values() if isinstance(v, bool))
    return results


@pytest.mark.integration
def test_all_scenarios() -> None:
    """Integration test: runs all 5 scenarios end-to-end. Requires OPENAI_API_KEY."""
    from src.agent.orchestrator import run_analysis_phase, run_clarify_phase

    scenarios = load_scenarios()
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for scenario in scenarios:
        session_id = str(uuid.uuid4())
        start = time.time()

        clarify_state = run_clarify_phase(
            csv_path=scenario["csv"],
            session_id=session_id,
            user_query=scenario["user_query"],
        )
        final_state = run_analysis_phase(clarify_state, answers=scenario["clarifying_answers"])

        elapsed = time.time() - start
        eval_result = evaluate_result(final_state, scenario["expected_outcomes"])

        trace = {
            "scenario_id": scenario["id"],
            "name": scenario["name"],
            "session_id": session_id,
            "elapsed_s": round(elapsed, 2),
            "evaluation": eval_result,
            "final_response_preview": (final_state.final_response or "")[:500],
            "subtask_results": [
                {
                    "id": r.subtask_id,
                    "agent": r.agent_type.value,
                    "success": r.success,
                    "output_preview": (r.output or "")[:200],
                }
                for r in final_state.subtask_results
            ],
        }
        trace_path = TRACES_DIR / f"{scenario['id']}_trace.json"
        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2)

        summary.append({"id": scenario["id"], "name": scenario["name"], "passed": eval_result["passed"]})
        print(f"\n{'PASS' if eval_result['passed'] else 'FAIL'} [{scenario['name']}] ({elapsed:.1f}s)")

    print("\n=== Evaluation Summary ===")
    for row in summary:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"  {status}  {row['name']}")

    passed = sum(1 for r in summary if r["passed"])
    print(f"\n{passed}/{len(summary)} scenarios passed")
