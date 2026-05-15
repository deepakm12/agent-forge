"""OpenAI exception types used as tenacity retry conditions across all LLM-calling modules."""

from __future__ import annotations

import openai

RETRYABLE_ERRORS = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)
