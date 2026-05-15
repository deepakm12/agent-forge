"""Specialist agent that synthesizes statistical outputs into narrative insights for non-technical stakeholders."""

from __future__ import annotations

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent._utils import RETRYABLE_ERRORS
from src.config import settings
from src.logging_config import get_logger
from src.prompts.system import INSIGHT_AGENT_SYSTEM
from src.schemas.models import AgentType, Subtask, SubtaskResult

logger = get_logger(__name__)

_client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(settings.max_retries),
)
def _call_llm(prompt: str) -> str:
    """Call the LLM with the given prompt, returning the generated insight text."""
    response = _client.responses.create(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        input=[
            {"role": "system", "content": INSIGHT_AGENT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text


def run_insight_agent(subtask: Subtask, dependency_outputs: str) -> SubtaskResult:
    """Generate a narrative insight for the given subtask using outputs from prior analyses."""
    logger.info("specialist_start", agent="insight", subtask_id=subtask.id)
    prompt = f"Task: {subtask.description}\n\nData from prior analyses:\n{dependency_outputs}"
    try:
        output = _call_llm(prompt)
        logger.info("specialist_complete", agent="insight", output_chars=len(output))
        return SubtaskResult(
            subtask_id=subtask.id,
            agent_type=AgentType.INSIGHT,
            output=output,
            success=True,
        )
    except Exception as exc:
        logger.error("specialist_error", agent="insight", error=str(exc))
        return SubtaskResult(subtask_id=subtask.id, agent_type=AgentType.INSIGHT, error=str(exc), success=False)
