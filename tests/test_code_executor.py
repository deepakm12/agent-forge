"""Tests for the sandboxed code executor: success, syntax errors, runtime errors, and timeout."""

from __future__ import annotations

from typing import Any

from src.tools.code_executor import execute_code


def test_execute_simple_code(tmp_path: Any) -> None:
    result = execute_code("print('hello')", output_dir=str(tmp_path))
    assert result.success is True
    assert "hello" in result.stdout


def test_execute_bad_syntax(tmp_path: Any) -> None:
    result = execute_code("def broken(:", output_dir=str(tmp_path))
    assert result.success is False


def test_execute_runtime_error(tmp_path: Any) -> None:
    result = execute_code("x = 1/0", output_dir=str(tmp_path))
    assert result.success is False
    assert "ZeroDivisionError" in result.stderr


def test_execute_timeout(tmp_path: Any) -> None:
    result = execute_code("import time; time.sleep(999)", output_dir=str(tmp_path))
    assert result.success is False
    assert "timed out" in result.stderr.lower()
