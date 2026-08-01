"""A reduction must not succeed by destroying the thing being tested.

For a *passing* test the oracle "output still matches" has a degenerate
solution: delete the assertions. pytest then prints "1 passed" exactly as
before, so the reducer is free to delete the entire library and score a
perfect reduction.

That is not a hypothetical failure. On flask it produced a "99.6%
reduction" whose two surviving files were

    def test_blueprint_specific_error_handling(app, client):
        pass

    @pytest.fixture
    def app():
        pass

with `import flask` resolving to nothing. The run proved that an empty
test passes.

The check here is a vacuity probe rather than a size assertion: break the
library on purpose and require the test to notice. A reduction that
survives its own dependency being sabotaged was never testing that
dependency.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from multi_file import MultiFileDebugger  # noqa: E402


LIB = '''\
def add(a, b):
    return a + b


def unused_helper(x):
    return x * 2


def another_unused():
    return "dead"
'''

TEST = '''\
import mylib


def test_add_works():
    assert mylib.add(2, 3) == 5


def test_unrelated():
    assert True
'''


def _build(tmp_path: Path) -> Path:
    proj = tmp_path / "vacuity"
    proj.mkdir()
    (proj / "mylib.py").write_text(LIB)
    (proj / "test_mylib.py").write_text(TEST)
    return proj


def _run(project: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q", "--no-header"],
        cwd=project, capture_output=True, text=True, timeout=120)


def test_reduction_stays_sensitive_to_its_dependency(tmp_path):
    proj = _build(tmp_path)
    target = "test_mylib.py::test_add_works"

    assert _run(proj, target).returncode == 0, "fixture must pass first"

    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    debugger.reduce_project(proj, [sys.executable, "-m", "pytest", target,
                                   "-q", "--no-header"])

    assert _run(proj, target).returncode == 0, "reduction broke the test"

    lib = proj / "mylib.py"
    assert lib.exists(), (
        "the library under test was deleted — the reduction is vacuous")

    # Sabotage the dependency. A genuine reduction must now fail.
    lib.write_text(LIB.replace("return a + b", "return a * b + 1"))
    sabotaged = _run(proj, target)
    assert sabotaged.returncode != 0, (
        "the reduced test still passes with a sabotaged library, so it "
        "asserts nothing about it — vacuous reduction")


def test_target_test_body_is_not_stubbed(tmp_path):
    """The named test is the oracle; its body must survive intact."""
    proj = _build(tmp_path)
    target = "test_mylib.py::test_add_works"

    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    debugger.reduce_project(proj, [sys.executable, "-m", "pytest", target,
                                   "-q", "--no-header"])

    body = (proj / "test_mylib.py").read_text()
    assert "test_add_works" in body, "the target test was removed entirely"
    assert "mylib.add" in body, (
        "the target test's body was stubbed away; it no longer exercises "
        "anything")
