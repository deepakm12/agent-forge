"""Tests for encode_chart_to_base64 and collect_charts_from_dir in chart_renderer."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from src.tools.chart_renderer import collect_charts_from_dir, encode_chart_to_base64


def test_encode_chart(tmp_path: Any) -> None:
    chart = tmp_path / "test.png"
    chart.write_bytes(b"fakepng")
    encoded = encode_chart_to_base64(str(chart))
    assert base64.b64decode(encoded) == b"fakepng"


def test_encode_missing_chart() -> None:
    with pytest.raises(FileNotFoundError):
        encode_chart_to_base64("/nonexistent/chart.png")


def test_collect_charts(tmp_path: Any) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"y")
    charts = collect_charts_from_dir(str(tmp_path))
    assert len(charts) == 2


def test_collect_charts_empty_dir(tmp_path: Any) -> None:
    assert collect_charts_from_dir(str(tmp_path)) == []
