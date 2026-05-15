"""Few-shot example messages for the clarify, plan, and reflect LLM prompts."""

from __future__ import annotations

from typing import Any

CLARIFY_FEW_SHOT: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": ("Profile: sales data, columns=[date, region, product, revenue, units]. " "User query: analyze this data."),
    },
    {
        "role": "assistant",
        "content": (
            '{"questions": ['
            '"Which time period should I focus on — a specific month, quarter, or the full dataset?", '
            '"Are you more interested in revenue trends, regional performance, or product mix analysis?", '
            '"Who will consume these insights — executives wanting summaries or analysts needing detail?"]}'
        ),
    },
]

PLAN_FEW_SHOT: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": ("Goal: identify top performing regions by revenue. " "Columns: date, region, product, revenue. " "Answer: focus on full year, revenue by region, for executives."),
    },
    {
        "role": "assistant",
        "content": (
            '{"goal": "Identify top performing regions by revenue", '
            '"reasoning": "We need descriptive stats by region, a bar chart for comparison, and narrative insight.", '
            '"subtasks": ['
            '{"id": "1", "description": "Compute revenue by region: sum, mean, pct of total", "agent_type": "stats", "dependencies": []}, '
            '{"id": "2", "description": "Create bar chart of revenue by region", "agent_type": "viz", "dependencies": ["1"]}, '
            '{"id": "3", "description": "Write narrative insight on regional performance", "agent_type": "insight", "dependencies": ["1", "2"]}]}'
        ),
    },
]

REFLECT_FEW_SHOT: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": ("Goal: correlation analysis. Results: stats computed matrix (SUCCESS), " "viz created heatmap (SUCCESS), insight wrote 3-paragraph summary (SUCCESS)."),
    },
    {
        "role": "assistant",
        "content": (
            '{"is_complete": true, '
            '"critique": "All subtasks completed. Correlation matrix present, heatmap generated, narrative insights provided.", '
            '"replan_needed": false, "replan_reason": null}'
        ),
    },
]
