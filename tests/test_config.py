"""Tests that Settings resolves defaults and required env vars correctly."""

import os
from unittest.mock import patch


def test_settings_loads_defaults() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
        from src.config import Settings

        s = Settings()
        assert s.openai_api_key == "test-key"
        assert s.openai_model == "gpt-4o"
        assert s.max_retries == 3
        assert s.output_dir == "outputs"
