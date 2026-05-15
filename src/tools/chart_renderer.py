"""Utilities for encoding chart images to base64 and collecting PNG/JPG files from a directory."""

from __future__ import annotations

import base64
from pathlib import Path

from src.logging_config import get_logger

logger = get_logger(__name__)


def encode_chart_to_base64(chart_path: str) -> str:
    """Return the base64-encoded contents of the chart file at chart_path."""
    p = Path(chart_path)
    if not p.exists():
        raise FileNotFoundError(f"Chart not found: {chart_path}")
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    logger.info("chart_encoded", path=chart_path, bytes=p.stat().st_size)
    return data


def collect_charts_from_dir(output_dir: str) -> list[str]:
    """Return a sorted list of PNG and JPG file paths found in output_dir."""
    p = Path(output_dir)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.png")) + sorted(str(f) for f in p.glob("*.jpg"))
