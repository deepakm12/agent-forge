"""Jinja2 prompt templates for clarify, plan, agent-code, reflect, and respond nodes."""

from __future__ import annotations

from jinja2 import Template

CLARIFY_TEMPLATE = Template(
    "Dataset Profile:\n"
    "- Shape: {{ profile.shape[0] }} rows × {{ profile.shape[1] }} columns\n"
    "- Columns: {% for col in profile.columns %}{{ col.name }}({{ col.dtype }}){% if not loop.last %}, {% endif %}{% endfor %}\n"
    "- Null issues: {% for col in profile.columns %}{% if col.null_count > 0 %}{{ col.name }}: {{ col.null_pct }}% null{% if not loop.last %} | {% endif %}{% endif %}{% endfor %}\n"
    "\nUser query: {{ user_query }}\n"
    "\nGenerate 2-3 targeted clarifying questions."
)

PLAN_TEMPLATE = Template(
    "Dataset Profile:\n"
    "- Shape: {{ profile.shape[0] }} rows × {{ profile.shape[1] }} columns\n"
    "- Columns: {% for col in profile.columns %}{{ col.name }}({{ col.dtype }}){% if not loop.last %}, {% endif %}{% endfor %}\n"
    "\nUser goal from clarification:\n"
    "{% for q, a in pairs %}Q: {{ q }}\nA: {{ a }}\n{% endfor %}"
    "\nCreate a structured analysis plan decomposed into subtasks for stats, viz, and insight agents."
)

AGENT_CODE_TEMPLATE = Template(
    "CSV file path: {{ csv_path }}\n"
    "Columns (use EXACTLY these names): {{ columns }}\n"
    "\nTask: {{ task_description }}\n"
    "\nPrior results from dependencies:\n{{ dependency_outputs }}\n"
    "\nWrite complete Python code to accomplish this task.\n"
    'Load CSV with: df = pd.read_csv(r"{{ csv_path }}")\n'
    "Print all findings clearly. Save any charts as PNG files."
)

REFLECT_TEMPLATE = Template(
    "Original goal: {{ goal }}\n"
    "\nSubtask results:\n"
    "{% for r in results %}{{ r.subtask_id }} ({{ r.agent_type }}): {{ 'SUCCESS' if r.success else 'FAILED' }}\n"
    "Output: {{ (r.output or '')[:300] }}\n{% endfor %}"
    "\nAssess whether the analysis completely addresses the original goal."
)

RESPOND_TEMPLATE = Template(
    "Analysis goal: {{ goal }}\n"
    "\nStatistical findings:\n{{ stats_output }}\n"
    "\nCharts generated: {{ chart_count }} chart(s)\n"
    "\nNarrative insights:\n{{ insight_output }}\n"
    "\nCompose a comprehensive final response."
)
