"""LangGraph node that evaluates subtask results and decides whether the analysis is complete or needs replanning."""

from __future__ import annotations

from typing import Any

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent._utils import RETRYABLE_ERRORS
from src.config import settings
from src.logging_config import get_logger
from src.prompts.few_shot import REFLECT_FEW_SHOT
from src.prompts.system import REFLECT_SYSTEM
from src.prompts.templates import REFLECT_TEMPLATE
from src.schemas.models import AgentState, Message, ReflectionResult, Role

logger = get_logger(__name__)

_client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(settings.max_retries),
)
def _call_llm(prompt: str) -> ReflectionResult:
    """Call the LLM to evaluate subtask results and determine if the analysis is complete, using few-shot examples and a system message for guidance."""
    response = _client.responses.parse(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        input=[
            {"role": "system", "content": REFLECT_SYSTEM},
            *[{"role": m["role"], "content": m["content"]} for m in REFLECT_FEW_SHOT],
            {"role": "user", "content": prompt},
        ],
        text_format=ReflectionResult,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for reflect node")
    return parsed


def reflect_node(state: AgentState) -> dict[str, Any]:
    """Evaluate subtask results and return a ReflectionResult indicating completeness or need to replan."""
    logger.info("node_start", node="reflect", session_id=state.session_id, iteration=state.iteration)
    if state.plan is None:
        return {"error": "No plan to reflect on"}
    if state.iteration >= settings.max_iterations:
        logger.warning("max_iterations_reached", iteration=state.iteration)
        reflection = ReflectionResult(
            is_complete=True,
            critique="Max iterations reached; proceeding with available results.",
            replan_needed=False,
        )
        return {"reflection": reflection, "iteration": state.iteration + 1}
    prompt = REFLECT_TEMPLATE.render(goal=state.plan.goal, results=state.subtask_results)
    try:
        reflection = _call_llm(prompt)
        status = "complete" if reflection.is_complete else "needs replan"
        msg = Message(
            role=Role.ASSISTANT,
            content=f"Reflection: {status}. {reflection.critique[:200]}",
            node_name="reflect",
        )
        logger.info(
            "node_complete",
            node="reflect",
            is_complete=reflection.is_complete,
            replan=reflection.replan_needed,
        )
        return {"reflection": reflection, "iteration": state.iteration + 1, "messages": state.messages + [msg]}
    except Exception as exc:
        logger.error("node_error", node="reflect", error=str(exc))
        return {"error": f"Reflect node failed: {exc}"}
