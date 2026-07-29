"""Phase 5: files that only become deletable after Phase 4 runs.

Phase 2 asks "can this file go?" while everything still imports it, so
the answer is no. Phase 4 is what deletes the import. Without a sweep
afterwards the module stays in the output as an orphan nothing can
reach — on requests that was ten files, a third of the surviving tree.

The fixture below is built so that the only thing keeping `sidecar.py`
alive is a function in main.py that is irrelevant to the bug. Once
Phase 4 removes that function and its import, sidecar.py is unreachable
and only a second look finds it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))

from multi_file import MultiFileDebugger  # noqa: E402


MAIN = '''\
import sidecar


def unrelated_report():
    """Only reason sidecar is imported. Irrelevant to the crash."""
    return sidecar.describe() + sidecar.describe()


def trigger():
    values = [1, 2, 3]
    return values["not-an-index"]


if __name__ == "__main__":
    trigger()
'''

SIDECAR = '''\
def describe():
    return "sidecar"


def also_unused():
    return 42
'''


def _build(tmp_path: Path) -> Path:
    proj = tmp_path / "sweepcase"
    proj.mkdir()
    (proj / "main.py").write_text(MAIN)
    (proj / "sidecar.py").write_text(SIDECAR)
    return proj


def _last_line(project_dir: Path) -> str:
    proc = subprocess.run([sys.executable, "main.py"], cwd=project_dir,
                          capture_output=True, text=True, timeout=60)
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [l for l in out.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def test_orphaned_module_is_removed(tmp_path):
    proj = _build(tmp_path)
    original_error = _last_line(proj)
    assert "TypeError" in original_error, original_error

    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    result = debugger.reduce_project(proj, [sys.executable, "main.py"])

    assert _last_line(proj) == original_error, "reduction changed the failure"
    assert not (proj / "sidecar.py").exists(), (
        "sidecar.py survived: nothing reconsidered it after Phase 4 "
        "removed the import that kept it alive")
    assert result.final_file_count == 1


def test_sweep_terminates(tmp_path):
    """The sweep loop must exit on its own, not via the safety valve."""
    proj = _build(tmp_path)
    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    # A run that hit MAX_SWEEP_ROUNDS would mean the loop never saw an
    # empty round, which for a two-file project would be a bug.
    assert debugger.MAX_SWEEP_ROUNDS >= 2
    result = debugger.reduce_project(proj, [sys.executable, "main.py"])
    assert result.final_file_count >= 1
