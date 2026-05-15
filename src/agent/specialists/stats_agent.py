"""Specialist agent that generates and executes statistical analysis Python code on a CSV file."""

from __future__ import annotations

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.agent._utils import RETRYABLE_ERRORS
from src.agent.specialists._utils import _strip_code_fences
from src.config import settings
from src.logging_config import get_logger
from src.prompts.system import STATS_AGENT_SYSTEM
from src.prompts.templates import AGENT_CODE_TEMPLATE
from src.schemas.models import AgentType, Subtask, SubtaskResult
from src.tools.chart_renderer import collect_charts_from_dir
from src.tools.code_executor import execute_code

logger = get_logger(__name__)

_client = openai.OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)


@retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(settings.max_retries),
)
def _call_llm(prompt: str) -> str:
    """Call the LLM with the given prompt and return the generated code."""
    response = _client.responses.create(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        input=[
            {"role": "system", "content": STATS_AGENT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.output_text


def run_stats_agent(
    subtask: Subtask,
    csv_path: str,
    columns: str,
    dependency_outputs: str,
    output_dir: str,
) -> SubtaskResult:
    """Generate and execute statistical analysis Python code, returning the captured output."""
    logger.info("specialist_start", agent="stats", subtask_id=subtask.id)
    prompt = AGENT_CODE_TEMPLATE.render(
        csv_path=csv_path,
        columns=columns,
        task_description=subtask.description,
        dependency_outputs=dependency_outputs,
    )
    try:
        code = _strip_code_fences(_call_llm(prompt))
        exec_result = execute_code(code, output_dir=output_dir, extra_preamble=f"csv_path = r'{csv_path}'")
        charts = collect_charts_from_dir(output_dir)
        logger.info("specialist_complete", agent="stats", success=exec_result.success)
        return SubtaskResult(
            subtask_id=subtask.id,
            agent_type=AgentType.STATS,
            code=code,
            output=exec_result.stdout if exec_result.success else exec_result.stderr,
            charts=charts,
            error=exec_result.stderr if not exec_result.success else None,
            success=exec_result.success,
        )
    except Exception as exc:
        logger.error("specialist_error", agent="stats", error=str(exc))
        return SubtaskResult(subtask_id=subtask.id, agent_type=AgentType.STATS, error=str(exc), success=False)
