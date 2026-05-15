"""Sandboxed subprocess executor that runs LLM-generated Python code with a hard timeout."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

_TIMEOUT_SECONDS = 30

# Block only direct HTTP-client libraries; do NOT block urllib/http/socket since
# data-science libraries (pandas, matplotlib) import them internally as dependencies.
_SANDBOX_PREAMBLE = textwrap.dedent("""\
    import builtins as _builtins
    _orig_import = _builtins.__import__
    _BLOCKED = {"requests", "httpx", "ftplib", "smtplib", "paramiko"}

    def _safe_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        root = name.split(".")[0]
        if root in _BLOCKED:
            raise ImportError(f"Import of '{name}' is blocked in sandbox")
        return _orig_import(name, *args, **kwargs)

    _builtins.__import__ = _safe_import
""")


class CodeExecutionResult:
    """Captures the stdout, stderr, and exit status of a sandboxed code run."""

    def __init__(self, stdout: str, stderr: str, success: bool) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.success = success


def execute_code(code: str, output_dir: str | None = None, extra_preamble: str = "") -> CodeExecutionResult:
    """Execute code in a subprocess with a sandbox preamble, writing any output files to output_dir."""
    effective_dir = output_dir or settings.output_dir
    Path(effective_dir).mkdir(parents=True, exist_ok=True)
    # Force non-interactive matplotlib backend before any user code runs
    _backend = "import matplotlib\nmatplotlib.use('Agg')\n"
    full_code = _SANDBOX_PREAMBLE + "\n" + _backend + (extra_preamble + "\n" if extra_preamble else "") + code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, cwd=effective_dir)
        logger.info("code_executed", returncode=proc.returncode, stdout_chars=len(proc.stdout))
        return CodeExecutionResult(stdout=proc.stdout[:10_000], stderr=proc.stderr[:2_000], success=proc.returncode == 0)
    except subprocess.TimeoutExpired:
        logger.warning("code_execution_timeout", timeout_s=_TIMEOUT_SECONDS)
        return CodeExecutionResult(stdout="", stderr=f"Execution timed out after {_TIMEOUT_SECONDS}s", success=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
