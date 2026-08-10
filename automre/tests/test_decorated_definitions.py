"""Decorated code must actually be removable by the pipeline.

The unit-level halves of this are in test_reduction_units.py. This is the
end-to-end claim: run a real reduction over a file whose dead code is
decorated, and require it to be gone.

It matters because the failure was silent in both directions. A decorated
function at module level was not a candidate, so the reducer reported
convergence with the code still there; a decorated method was a candidate
whose every removal was a syntax error, so it was rejected locally and no
query was ever spent proving it could go. Neither shows up as a failure
anywhere — only as a reduction that stops short.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from multi_file import MultiFileDebugger  # noqa: E402


MAIN = '''\
import functools


@functools.lru_cache(maxsize=None)
def dead_cached(n):
    return n * 2 + 7


@functools.lru_cache(maxsize=None)
def live_cached(n):
    return n + 1


class Config:
    @property
    def dead_property(self):
        return "never read"

    @staticmethod
    def dead_static():
        return 99


def trigger():
    values = [1, 2, 3]
    return values["not-an-index"]


if __name__ == "__main__":
    live_cached(1)
    trigger()
'''


def _build(tmp_path: Path) -> Path:
    proj = tmp_path / "decorated"
    proj.mkdir()
    (proj / "main.py").write_text(MAIN)
    return proj


def _last_line(project_dir: Path) -> str:
    proc = subprocess.run([sys.executable, "main.py"], cwd=project_dir,
                          capture_output=True, text=True, timeout=60)
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [l for l in out.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def test_dead_decorated_code_is_removed(tmp_path):
    proj = _build(tmp_path)
    original_error = _last_line(proj)
    assert "TypeError" in original_error, original_error

    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    debugger.reduce_project(proj, [sys.executable, "main.py"])

    assert _last_line(proj) == original_error, "reduction changed the failure"

    survived = (proj / "main.py").read_text()
    assert "dead_cached" not in survived, (
        "a module-level decorated function survived reduction — it was "
        "never offered as a candidate")
    assert "dead_property" not in survived, (
        "a decorated method survived reduction — removing it stranded the "
        "decorator, so it was rejected locally every time")
    assert "dead_static" not in survived
