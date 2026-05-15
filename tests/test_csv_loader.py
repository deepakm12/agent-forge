"""Tests for load_csv: profile shape, null tracking, token estimates, and error cases."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.tools.csv_loader import CSVLoadError, load_csv


@pytest.fixture
def simple_csv(tmp_path: Any) -> str:
    p = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]}).to_csv(p, index=False)
    return str(p)


def test_load_csv_returns_profile(simple_csv: str) -> None:
    df, profile = load_csv(simple_csv)
    assert profile.shape == (3, 2)
    assert len(profile.columns) == 2
    assert profile.columns[0].name == "a"
    assert profile.columns[0].null_count == 1


def test_load_csv_missing_file() -> None:
    with pytest.raises(CSVLoadError, match="not found"):
        load_csv("/nonexistent/path.csv")


def test_load_csv_token_estimate(simple_csv: str) -> None:
    _, profile = load_csv(simple_csv)
    assert profile.token_estimate > 0
