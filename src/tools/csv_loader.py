"""Loads a CSV file into a DataFrame and builds a DataFrameProfile, enforcing size and row limits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.logging_config import get_logger
from src.schemas.models import ColumnProfile, DataFrameProfile

logger = get_logger(__name__)

_MAX_SAMPLE_ROWS = 10


class CSVLoadError(Exception):
    """Raised when a CSV file cannot be loaded due to missing path, size, or row-count violations."""


def load_csv(path: str) -> tuple[pd.DataFrame, DataFrameProfile]:
    """Read the CSV at path, enforce size/row limits, and return the DataFrame with its profile."""
    p = Path(path)
    if not p.exists():
        raise CSVLoadError(f"File not found: {path}")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > settings.csv_max_size_mb:
        raise CSVLoadError(f"File too large: {size_mb:.1f}MB > {settings.csv_max_size_mb}MB limit")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    if len(df) > settings.csv_max_rows:
        raise CSVLoadError(f"Too many rows: {len(df)} > {settings.csv_max_rows} limit")
    profile = _profile_dataframe(df)
    logger.info("csv_loaded", path=path, shape=profile.shape, token_estimate=profile.token_estimate)
    return df, profile


def _profile_dataframe(df: pd.DataFrame) -> DataFrameProfile:
    """Profile the DataFrame by collecting column metadata and sample rows, estimating token count for LLM input."""
    columns: list[ColumnProfile] = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        sample_vals: list[Any] = df[col].dropna().head(5).tolist()
        columns.append(
            ColumnProfile(
                name=str(col),
                dtype=str(df[col].dtype),
                null_count=null_count,
                null_pct=round(null_count / max(len(df), 1) * 100, 2),
                unique_count=int(df[col].nunique()),
                sample_values=sample_vals,
            )
        )
    sample_rows: list[dict[str, Any]] = df.head(_MAX_SAMPLE_ROWS).fillna("").to_dict(orient="records")
    token_estimate = len(str(sample_rows)) // 4
    return DataFrameProfile(shape=(len(df), len(df.columns)), columns=columns, token_estimate=token_estimate, sample_rows=sample_rows)
