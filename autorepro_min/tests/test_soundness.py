"""Soundness: a reduction must reproduce the *same* failure.

This is the property that matters most and the one that silently broke.
With match_strategy="error_type" the reducer returned a 6-line dep-chain
that raised

    TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'

while the original raised

    TypeError: can only concatenate str (not "list") to str

Both are TypeErrors, so the oracle accepted it. A reducer that returns a
different bug is worse than one that reduces nothing, and nothing in the
repo checked for it.

These tests run the real pipeline end to end on the bundled examples and
compare the final line of output verbatim.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))

from multi_file import MultiFileDebugger  # noqa: E402

EXAMPLES = _ROOT / "autorepro_min" / "examples" / "multi_file"
CASES = [
    "project1_cross_file_type_error",
    "project2_dep_chain",
    "project3_side_effects",
]


def _last_line(project_dir: Path) -> str:
    proc = subprocess.run([sys.executable, "main.py"], cwd=project_dir,
                          capture_output=True, text=True, timeout=60)
    combined = (proc.stdout or "") + (proc.stderr or "")
    lines = [l for l in combined.splitlines() if l.strip()]
    return lines[-1] if lines else ""


@pytest.mark.parametrize("name", CASES)
def test_reduction_preserves_exact_error(tmp_path, name):
    src = EXAMPLES / name
    assert src.is_dir(), f"missing example {name}"

    original_error = _last_line(src)
    assert original_error, "example must fail with some output"

    work = tmp_path / name
    shutil.copytree(src, work)

    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    result = debugger.reduce_project(work, [sys.executable, "main.py"])

    assert _last_line(work) == original_error, (
        f"{name}: reduction changed the failure.\n"
        f"  before: {original_error}\n"
        f"  after : {_last_line(work)}")
    assert result.final_line_count <= result.original_line_count


@pytest.mark.parametrize("name", CASES)
def test_reduction_actually_reduces(tmp_path, name):
    """A no-op reducer would pass the soundness test trivially."""
    src = EXAMPLES / name
    work = tmp_path / name
    shutil.copytree(src, work)

    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    result = debugger.reduce_project(work, [sys.executable, "main.py"])

    assert result.final_line_count < result.original_line_count, (
        f"{name}: nothing was removed")


def test_error_type_strategy_is_documented_as_permissive():
    """error_type accepts a different bug of the same class.

    Kept as an executable note rather than a bug report: the latitude is
    intentional and available, but it must not be the default, which is
    what this asserts.
    """
    import argparse
    import cli

    parser_src = Path(cli.__file__).read_text()
    assert "default='output_match'" in parser_src, (
        "reduce-project must not default to a permissive match strategy")
