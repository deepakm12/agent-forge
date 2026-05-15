"""Tests for build_graph compilation and should_replan routing logic."""

from __future__ import annotations

from src.schemas.models import AgentState, ReflectionResult


def test_orchestrator_build_returns_compiled_graph() -> None:
    from src.agent.orchestrator import build_graph

    graph = build_graph()
    assert graph is not None


def test_should_replan_routes_to_respond_when_complete() -> None:
    from src.agent.orchestrator import should_replan

    state = AgentState(
        session_id="s1",
        csv_path="/f.csv",
        reflection=ReflectionResult(is_complete=True, critique="ok", replan_needed=False),
    )
    assert should_replan(state) == "respond"


def test_should_replan_routes_to_plan_when_needed() -> None:
    from src.agent.orchestrator import should_replan

    state = AgentState(
        session_id="s1",
        csv_path="/f.csv",
        reflection=ReflectionResult(is_complete=False, critique="missing viz", replan_needed=True),
        iteration=0,
    )
    assert should_replan(state) == "plan"


def test_should_replan_routes_to_respond_on_error() -> None:
    from src.agent.orchestrator import should_replan

    state = AgentState(session_id="s1", csv_path="/f.csv", error="something failed")
    assert should_replan(state) == "respond"
