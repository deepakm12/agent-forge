"""Pydantic data models and enums shared across the entire agent-forge codebase."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Agent types for different stages of the analysis process."""

    STATS = "stats"
    VIZ = "viz"
    INSIGHT = "insight"


class Role(str, Enum):
    """Roles for messages exchanged between the user and the agent."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """A message exchanged between the user and the agent."""

    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_name: str | None = None


class ColumnProfile(BaseModel):
    """Column profile containing metadata and statistics about a single DataFrame column."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]


class DataFrameProfile(BaseModel):
    """DataFrame profile containing metadata and statistics about the entire DataFrame."""

    shape: tuple[int, int]
    columns: list[ColumnProfile]
    token_estimate: int
    sample_rows: list[dict[str, Any]]


class Subtask(BaseModel):
    """Subtask assigned to a specific agent type as part of the overall analysis plan."""

    id: str
    description: str
    agent_type: AgentType
    dependencies: list[str] = []


class AnalysisPlan(BaseModel):
    """Analysis plan containing the goal, reasoning, and subtasks for the analysis."""

    goal: str
    reasoning: str
    subtasks: list[Subtask]


class SubtaskResult(BaseModel):
    """Subtask result containing the output, charts, and any errors from executing a subtask."""

    subtask_id: str
    agent_type: AgentType
    code: str | None = None
    output: str | None = None
    charts: list[str] = []
    error: str | None = None
    success: bool = True


class ReflectionResult(BaseModel):
    """Reflection result containing the agent's self-assessment of the analysis process and any insights gained."""

    is_complete: bool
    critique: str
    replan_needed: bool
    replan_reason: str | None = None


class AnalysisInsight(BaseModel):
    """Analysis insight containing the title, body, confidence, and supporting statistics for an insight."""

    title: str
    body: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_stats: list[str] = []


class AgentState(BaseModel):
    """Agent state containing all relevant information about the current state of the agent's analysis process."""

    session_id: str
    csv_path: str
    dataframe_profile: DataFrameProfile | None = None
    messages: list[Message] = []
    clarifying_questions: list[str] = []
    clarifying_answers: list[str] = []
    plan: AnalysisPlan | None = None
    subtask_results: list[SubtaskResult] = []
    reflection: ReflectionResult | None = None
    final_response: str | None = None
    error: str | None = None
    iteration: int = 0
