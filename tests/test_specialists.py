"""Tests for stats, viz, and insight specialist agents using mocked LLM calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from src.agent.specialists.insight_agent import run_insight_agent
from src.agent.specialists.stats_agent import run_stats_agent
from src.agent.specialists.viz_agent import run_viz_agent
from src.schemas.models import AgentType, Subtask, SubtaskResult


def test_stats_agent_returns_result(tmp_path: Any) -> None:
    """Test that the stats agent returns a SubtaskResult with expected values."""
    with patch("src.agent.specialists.stats_agent._call_llm", return_value="print('stats done')"):
        subtask = Subtask(id="1", description="compute stats", agent_type=AgentType.STATS)
        result = run_stats_agent(
            subtask,
            csv_path="/tmp/nonexistent.csv",
            columns="col1, col2",
            dependency_outputs="",
            output_dir=str(tmp_path),
        )
        assert isinstance(result, SubtaskResult)
        assert result.subtask_id == "1"
        assert result.agent_type == AgentType.STATS


def test_viz_agent_returns_result(tmp_path: Any) -> None:
    """Test that the viz agent returns a SubtaskResult with expected values."""
    with patch("src.agent.specialists.viz_agent._call_llm", return_value="print('chart done')"):
        subtask = Subtask(id="2", description="create chart", agent_type=AgentType.VIZ)
        result = run_viz_agent(
            subtask,
            csv_path="/tmp/nonexistent.csv",
            columns="col1, col2",
            dependency_outputs="some stats",
            output_dir=str(tmp_path),
        )
        assert isinstance(result, SubtaskResult)
        assert result.agent_type == AgentType.VIZ


def test_insight_agent_returns_result() -> None:
    """Test that the insight agent returns a SubtaskResult with expected values."""
    with patch("src.agent.specialists.insight_agent._call_llm", return_value="Key insight: data looks good."):
        subtask = Subtask(id="3", description="write insights", agent_type=AgentType.INSIGHT)
        result = run_insight_agent(subtask, dependency_outputs="stats output here")
        assert isinstance(result, SubtaskResult)
        assert result.success is True
        assert result.output == "Key insight: data looks good."
