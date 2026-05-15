"""LangGraph node that dispatches each subtask in the analysis plan to the appropriate specialist agent."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.agent.specialists.insight_agent import run_insight_agent
from src.agent.specialists.stats_agent import run_stats_agent
from src.agent.specialists.viz_agent import run_viz_agent
from src.config import settings
from src.logging_config import get_logger
from src.schemas.models import AgentState, AgentType, Message, Role, SubtaskResult

logger = get_logger(__name__)


def dispatch_node(state: AgentState) -> dict[str, Any]:
    """Run each plan subtask through its designated specialist agent and collect the results."""
    logger.info("node_start", node="dispatch", session_id=state.session_id)
    if state.plan is None:
        return {"error": "No plan available for dispatch"}
    run_id = uuid.uuid4().hex[:8]
    output_dir = str(Path(settings.output_dir) / run_id)
    abs_csv_path = str(Path(state.csv_path).resolve())
    columns = ", ".join(f"{c.name}({c.dtype})" for c in state.dataframe_profile.columns) if state.dataframe_profile else ""
    results: list[SubtaskResult] = []
    completed_outputs: dict[str, str] = {}
    for subtask in state.plan.subtasks:
        dep_outputs = "\n\n".join(f"[{sid}]: {completed_outputs[sid]}" for sid in subtask.dependencies if sid in completed_outputs)
        if subtask.agent_type == AgentType.STATS:
            result = run_stats_agent(subtask, csv_path=abs_csv_path, columns=columns, dependency_outputs=dep_outputs, output_dir=output_dir)
        elif subtask.agent_type == AgentType.VIZ:
            result = run_viz_agent(subtask, csv_path=abs_csv_path, columns=columns, dependency_outputs=dep_outputs, output_dir=output_dir)
        else:
            result = run_insight_agent(subtask, dependency_outputs=dep_outputs)
        results.append(result)
        if result.output:
            completed_outputs[subtask.id] = result.output
    msg = Message(
        role=Role.ASSISTANT,
        content=f"Dispatched {len(results)} subtask(s). {sum(1 for r in results if r.success)} succeeded.",
        node_name="dispatch",
    )
    logger.info("node_complete", node="dispatch", results_count=len(results))
    return {"subtask_results": results, "messages": state.messages + [msg]}
