"""LangGraph node that synthesizes all subtask outputs into a final human-readable response."""

from __future__ import annotations

from typing import Any

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent._utils import RETRYABLE_ERRORS
from src.config import settings
from src.logging_config import get_logger
from src.prompts.templates import RESPOND_TEMPLATE
from src.schemas.models import AgentState, AgentType, Message, Role

logger = get_logger(__name__)

_client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(settings.max_retries),
)
def _call_llm(prompt: str) -> str:
    """Call the LLM to generate the final response based on the provided prompt, using a simple user message format."""
    response = _client.responses.create(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        input=[{"role": "user", "content": prompt}],
    )
    return response.output_text


def respond_node(state: AgentState) -> dict[str, Any]:
    """Compose the final response by combining stats, insight, and chart outputs."""
    logger.info("node_start", node="respond", session_id=state.session_id)
    stats_output = "\n".join(r.output or "" for r in state.subtask_results if r.agent_type == AgentType.STATS and r.output)
    insight_output = "\n".join(r.output or "" for r in state.subtask_results if r.agent_type == AgentType.INSIGHT and r.output)
    all_charts = [c for r in state.subtask_results for c in r.charts]
    goal = state.plan.goal if state.plan else "data analysis"
    prompt = RESPOND_TEMPLATE.render(
        goal=goal,
        stats_output=stats_output or "No statistical output available.",
        chart_count=len(all_charts),
        insight_output=insight_output or "No narrative insights available.",
    )
    try:
        response = _call_llm(prompt)
        msg = Message(role=Role.ASSISTANT, content=response, node_name="respond")
        logger.info("node_complete", node="respond", response_chars=len(response))
        return {"final_response": response, "messages": state.messages + [msg]}
    except Exception as exc:
        logger.error("node_error", node="respond", error=str(exc))
        fallback = f"Analysis complete.\n\nStats:\n{stats_output[:500]}\n\nInsights:\n{insight_output[:500]}"
        return {"final_response": fallback, "error": str(exc)}
