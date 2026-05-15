"""Shared utility helpers for specialist agents."""

from __future__ import annotations

import re


def _strip_code_fences(code: str) -> str:
    """Extract code from a markdown fenced block (```python...``` or ```...```)."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip opening/closing fence lines manually
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)
