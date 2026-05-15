"""LangGraph node that creates a structured analysis plan decomposed into specialist subtasks."""

from __future__ import annotations

from typing import Any

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent._utils import RETRYABLE_ERRORS
from src.config import settings
from src.logging_config import get_logger
from src.prompts.few_shot import PLAN_FEW_SHOT
from src.prompts.system import PLAN_SYSTEM
from src.prompts.templates import PLAN_TEMPLATE
from src.schemas.models import AgentState, AnalysisPlan, Message, Role

logger = get_logger(__name__)

_client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(settings.max_retries),
)
def _call_llm(prompt: str) -> AnalysisPlan:
    """Call the LLM to generate an analysis plan based on the provided prompt, using few-shot examples and a system message for guidance."""
    response = _client.responses.parse(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        input=[
            {"role": "system", "content": PLAN_SYSTEM},
            *[{"role": m["role"], "content": m["content"]} for m in PLAN_FEW_SHOT],
            {"role": "user", "content": prompt},
        ],
        text_format=AnalysisPlan,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for plan node")
    return parsed


def plan_node(state: AgentState) -> dict[str, Any]:
    """Generate an AnalysisPlan with ordered subtasks based on the dataset profile and user answers."""
    logger.info("node_start", node="plan", session_id=state.session_id)
    if state.dataframe_profile is None:
        return {"error": "No dataframe profile available for planning"}
    pairs = list(zip(state.clarifying_questions, state.clarifying_answers))
    prompt = PLAN_TEMPLATE.render(profile=state.dataframe_profile, pairs=pairs)
    try:
        plan = _call_llm(prompt)
        msg = Message(
            role=Role.ASSISTANT,
            content=f"Analysis plan created with {len(plan.subtasks)} subtask(s): {plan.goal}",
            node_name="plan",
        )
        logger.info("node_complete", node="plan", subtask_count=len(plan.subtasks))
        return {"plan": plan, "messages": state.messages + [msg]}
    except Exception as exc:
        logger.error("node_error", node="plan", error=str(exc))
        return {"error": f"Plan node failed: {exc}"}
