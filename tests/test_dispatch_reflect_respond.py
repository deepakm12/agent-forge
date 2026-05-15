"""Tests for the dispatch, reflect, and respond LangGraph nodes using mocked specialist agents."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.schemas.models import (
    AgentState,
    AgentType,
    AnalysisPlan,
    ColumnProfile,
    DataFrameProfile,
    ReflectionResult,
    Subtask,
    SubtaskResult,
)


@pytest.fixture
def state_with_plan() -> AgentState:
    profile = DataFrameProfile(
        shape=(10, 2),
        columns=[ColumnProfile(name="a", dtype="float64", null_count=0, null_pct=0.0, unique_count=10, sample_values=[1.0])],
        token_estimate=10,
        sample_rows=[],
    )
    plan = AnalysisPlan(
        goal="test goal",
        reasoning="r",
        subtasks=[Subtask(id="1", description="stats", agent_type=AgentType.STATS)],
    )
    return AgentState(session_id="s1", csv_path="/f.csv", dataframe_profile=profile, plan=plan)


def test_dispatch_node_runs_subtasks(state_with_plan: AgentState) -> None:
    from src.agent.nodes.dispatch import dispatch_node

    mock_result = SubtaskResult(subtask_id="1", agent_type=AgentType.STATS, output="mean=5", success=True)
    with patch("src.agent.nodes.dispatch.run_stats_agent", return_value=mock_result):
        result = dispatch_node(state_with_plan)
        assert len(result["subtask_results"]) == 1
        assert result["subtask_results"][0].output == "mean=5"


def test_reflect_node_marks_complete(state_with_plan: AgentState) -> None:
    from src.agent.nodes.reflect import reflect_node

    mock_reflection = ReflectionResult(is_complete=True, critique="good", replan_needed=False)
    state = state_with_plan.model_copy(update={"subtask_results": [SubtaskResult(subtask_id="1", agent_type=AgentType.STATS, output="ok", success=True)]})
    with patch("src.agent.nodes.reflect._call_llm", return_value=mock_reflection):
        result = reflect_node(state)
        assert result["reflection"].is_complete is True
        assert result["reflection"].replan_needed is False


def test_respond_node_sets_final_response(state_with_plan: AgentState) -> None:
    from src.agent.nodes.respond import respond_node

    state = state_with_plan.model_copy(
        update={
            "subtask_results": [SubtaskResult(subtask_id="1", agent_type=AgentType.STATS, output="stats here", success=True)],
            "reflection": ReflectionResult(is_complete=True, critique="good", replan_needed=False),
        }
    )
    with patch("src.agent.nodes.respond._call_llm", return_value="Final answer here"):
        result = respond_node(state)
        assert result["final_response"] == "Final answer here"
