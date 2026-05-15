"""LangGraph node that loads a CSV file and builds a DataFrameProfile for downstream nodes."""

from __future__ import annotations

from typing import Any

from src.logging_config import get_logger
from src.schemas.models import AgentState, Message, Role
from src.tools.csv_loader import CSVLoadError, load_csv

logger = get_logger(__name__)


def ingest_node(state: AgentState) -> dict[str, Any]:
    """Load the CSV at state.csv_path and attach its DataFrameProfile to the state."""
    logger.info("node_start", node="ingest", session_id=state.session_id)
    try:
        _, profile = load_csv(state.csv_path)
        msg = Message(
            role=Role.SYSTEM,
            content=f"CSV loaded: {profile.shape[0]} rows, {profile.shape[1]} columns.",
            node_name="ingest",
        )
        logger.info("node_complete", node="ingest", shape=profile.shape)
        return {"dataframe_profile": profile, "messages": state.messages + [msg]}
    except CSVLoadError as exc:
        logger.error("node_error", node="ingest", error=str(exc))
        return {"error": str(exc)}
