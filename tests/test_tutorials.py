"""
Smoke-run every tutorial in fast mode so they cannot silently rot.

Each script runs in a subprocess with HCNN_TUTORIAL_FAST=1 (tiny data, 1-2 epochs),
a non-interactive matplotlib backend, and outputs redirected to a temp dir.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = sorted((ROOT / "tutorials").glob("[0-9][0-9]_*.py"))


def test_tutorials_exist():
    assert len(TUTORIALS) >= 7


@pytest.mark.parametrize("script", TUTORIALS, ids=lambda p: p.stem)
def test_tutorial_runs_in_fast_mode(script, tmp_path):
    env = {
        **os.environ,
        "HCNN_TUTORIAL_FAST": "1",
        "HCNN_TUTORIAL_OUT": str(tmp_path),
        "MPLBACKEND": "Agg",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
    }
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stderr[-3000:]}"
