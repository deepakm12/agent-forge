"""Tests for Pydantic schema defaults, enum values, validation constraints, and model_copy."""

import pytest
from pydantic import ValidationError

from src.schemas.models import (
    AgentState,
    AgentType,
    AnalysisInsight,
    ReflectionResult,
    Subtask,
    SubtaskResult,
)


def test_agent_state_defaults() -> None:
    state = AgentState(session_id="abc", csv_path="/tmp/test.csv")
    assert state.messages == []
    assert state.iteration == 0
    assert state.plan is None


def test_subtask_agent_type_enum() -> None:
    st = Subtask(id="1", description="compute stats", agent_type=AgentType.STATS)
    assert st.agent_type == AgentType.STATS


def test_analysis_insight_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        AnalysisInsight(title="t", body="b", confidence=1.5)


def test_reflection_result_defaults() -> None:
    r = ReflectionResult(is_complete=True, critique="good", replan_needed=False)
    assert r.replan_reason is None


def test_subtask_result_success_default() -> None:
    sr = SubtaskResult(subtask_id="1", agent_type=AgentType.VIZ)
    assert sr.success is True
    assert sr.charts == []


def test_agent_state_model_copy() -> None:
    state = AgentState(session_id="s1", csv_path="/tmp/f.csv")
    updated = state.model_copy(update={"iteration": 1, "error": "oops"})
    assert updated.iteration == 1
    assert updated.error == "oops"
    assert state.iteration == 0
