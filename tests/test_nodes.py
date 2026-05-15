"""Tests for the ingest, clarify, and plan LangGraph nodes."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from src.schemas.models import (
    AgentState,
    AgentType,
    AnalysisPlan,
    ColumnProfile,
    DataFrameProfile,
    Message,
    Role,
    Subtask,
)


@pytest.fixture
def minimal_profile() -> DataFrameProfile:
    return DataFrameProfile(
        shape=(10, 2),
        columns=[
            ColumnProfile(
                name="revenue",
                dtype="float64",
                null_count=0,
                null_pct=0.0,
                unique_count=10,
                sample_values=[1.0],
            ),
            ColumnProfile(
                name="region",
                dtype="object",
                null_count=0,
                null_pct=0.0,
                unique_count=3,
                sample_values=["A"],
            ),
        ],
        token_estimate=50,
        sample_rows=[{"revenue": 1.0, "region": "A"}],
    )


def test_ingest_node_sets_profile(tmp_path: Any) -> None:
    csv = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv, index=False)

    from src.agent.nodes.ingest import ingest_node

    state = AgentState(session_id="s1", csv_path=str(csv))
    result = ingest_node(state)
    assert result["dataframe_profile"] is not None
    assert result["dataframe_profile"].shape == (2, 2)


def test_ingest_node_sets_error_on_missing_file() -> None:
    from src.agent.nodes.ingest import ingest_node

    state = AgentState(session_id="s1", csv_path="/nonexistent.csv")
    result = ingest_node(state)
    assert result["error"] is not None


def test_clarify_node_returns_questions(minimal_profile: DataFrameProfile) -> None:
    from src.agent.nodes.clarify import ClarifyingQuestions, clarify_node

    mock_result = ClarifyingQuestions(questions=["Q1?", "Q2?"])
    with patch("src.agent.nodes.clarify._call_llm", return_value=mock_result):
        state = AgentState(
            session_id="s1",
            csv_path="/f.csv",
            dataframe_profile=minimal_profile,
            messages=[Message(role=Role.USER, content="analyze")],
        )
        result = clarify_node(state)
        assert result["clarifying_questions"] == ["Q1?", "Q2?"]


def test_plan_node_returns_plan(minimal_profile: DataFrameProfile) -> None:
    from src.agent.nodes.plan import plan_node

    mock_plan = AnalysisPlan(
        goal="test goal",
        reasoning="because",
        subtasks=[Subtask(id="1", description="stats", agent_type=AgentType.STATS)],
    )
    with patch("src.agent.nodes.plan._call_llm", return_value=mock_plan):
        state = AgentState(
            session_id="s1",
            csv_path="/f.csv",
            dataframe_profile=minimal_profile,
            clarifying_questions=["Q1?"],
            clarifying_answers=["A1"],
        )
        result = plan_node(state)
        assert result["plan"] is not None
        assert result["plan"].goal == "test goal"
