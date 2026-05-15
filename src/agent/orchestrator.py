"""Builds and runs the two-phase LangGraph workflow (clarify → plan → dispatch → reflect → respond)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.nodes.clarify import clarify_node
from src.agent.nodes.dispatch import dispatch_node
from src.agent.nodes.ingest import ingest_node
from src.agent.nodes.plan import plan_node
from src.agent.nodes.reflect import reflect_node
from src.agent.nodes.respond import respond_node
from src.config import settings
from src.logging_config import get_logger
from src.schemas.models import AgentState, Message, Role

logger = get_logger(__name__)


def should_replan(state: AgentState) -> str:
    """Return the next node name ('plan' or 'respond') based on the reflection result and iteration count."""
    if state.error:
        return "respond"
    if state.reflection is None:
        return "respond"
    if state.reflection.replan_needed and state.iteration < settings.max_iterations:
        return "plan"
    return "respond"


def build_graph() -> StateGraph[Any]:
    """Compile and return the full LangGraph state machine for the two-phase analysis workflow."""
    graph: StateGraph[Any] = StateGraph(AgentState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("respond", respond_node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "clarify")
    graph.add_edge("clarify", END)
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "reflect")
    graph.add_conditional_edges(
        "reflect",
        should_replan,
        {"plan": "plan", "respond": "respond"},
    )
    graph.add_edge("respond", END)
    return graph.compile()


def run_clarify_phase(csv_path: str, session_id: str, user_query: str) -> AgentState:
    """Run the ingest → clarify subgraph and return the state with clarifying questions populated."""
    initial_state = AgentState(
        session_id=session_id,
        csv_path=csv_path,
        messages=[Message(role=Role.USER, content=user_query)],
    )
    graph = build_graph()
    result = graph.invoke(initial_state)
    return AgentState.model_validate(result)


def run_analysis_phase(state: AgentState, answers: list[str]) -> AgentState:
    """Run the plan → dispatch → reflect → respond subgraph and return the final state."""
    updated_state = state.model_copy(update={"clarifying_answers": answers})
    analysis_graph: StateGraph[Any] = StateGraph(AgentState)
    analysis_graph.add_node("plan", plan_node)
    analysis_graph.add_node("dispatch", dispatch_node)
    analysis_graph.add_node("reflect", reflect_node)
    analysis_graph.add_node("respond", respond_node)
    analysis_graph.set_entry_point("plan")
    analysis_graph.add_edge("plan", "dispatch")
    analysis_graph.add_edge("dispatch", "reflect")
    analysis_graph.add_conditional_edges(
        "reflect",
        should_replan,
        {"plan": "plan", "respond": "respond"},
    )
    analysis_graph.add_edge("respond", END)
    compiled = analysis_graph.compile()
    result = compiled.invoke(updated_state)
    return AgentState.model_validate(result)
