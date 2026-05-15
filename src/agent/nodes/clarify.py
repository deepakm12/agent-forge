"""LangGraph node that asks the LLM to generate clarifying questions about a CSV dataset."""

from __future__ import annotations

from typing import Any

import openai
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent._utils import RETRYABLE_ERRORS
from src.config import settings
from src.logging_config import get_logger
from src.prompts.few_shot import CLARIFY_FEW_SHOT
from src.prompts.system import CLARIFY_SYSTEM
from src.prompts.templates import CLARIFY_TEMPLATE
from src.schemas.models import AgentState, Message, Role

logger = get_logger(__name__)

_client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


class ClarifyingQuestions(BaseModel):
    """Structured output holding the LLM-generated clarifying questions."""

    questions: list[str]


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(settings.max_retries),
)
def _call_llm(prompt: str) -> ClarifyingQuestions:
    """Call the LLM to generate clarifying questions based on the provided prompt, using few-shot examples and a system message for guidance."""
    response = _client.responses.parse(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        input=[
            {"role": "system", "content": CLARIFY_SYSTEM},
            *[{"role": m["role"], "content": m["content"]} for m in CLARIFY_FEW_SHOT],
            {"role": "user", "content": prompt},
        ],
        text_format=ClarifyingQuestions,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for clarify node")
    return parsed


def clarify_node(state: AgentState) -> dict[str, Any]:
    """Return up to three clarifying questions derived from the dataset profile and user query."""
    logger.info("node_start", node="clarify", session_id=state.session_id)
    if state.dataframe_profile is None:
        return {"error": "No dataframe profile available for clarification"}
    user_query = next(
        (m.content for m in reversed(state.messages) if m.role == Role.USER),
        "Please analyze this data.",
    )
    prompt = CLARIFY_TEMPLATE.render(profile=state.dataframe_profile, user_query=user_query)
    try:
        result = _call_llm(prompt)
        questions = result.questions[:3]
        msg = Message(
            role=Role.ASSISTANT,
            content="\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)),
            node_name="clarify",
        )
        logger.info("node_complete", node="clarify", question_count=len(questions))
        return {"clarifying_questions": questions, "messages": state.messages + [msg]}
    except Exception as exc:
        logger.error("node_error", node="clarify", error=str(exc))
        return {"error": f"Clarify node failed: {exc}"}
