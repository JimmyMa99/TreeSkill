"""Regression tests for package import behavior."""

import subprocess
import sys
from pathlib import Path


def test_import_evoskill_without_optional_modules():
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", "import evoskill"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
