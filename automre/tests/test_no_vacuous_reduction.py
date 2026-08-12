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

import pytest

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


# --------------------------------------------------------------------
# Which commands actually get that protection
#
# The protection above keys off `.py` paths in the reproduction command.
# Two ordinary ways of naming a test suite do not contain one, and for
# those the reducer was handed an unprotected oracle — the same failure,
# through a different door.


def _suite(tmp_path: Path) -> Path:
    proj = tmp_path / "suite"
    (proj / "tests").mkdir(parents=True)
    (proj / "mylib.py").write_text(LIB)
    (proj / "tests" / "test_mylib.py").write_text(
        TEST.replace("import mylib", "import mylib"))
    (proj / "tests" / "conftest.py").write_text("import mylib  # noqa: F401\n")
    return proj


def test_a_directory_argument_protects_the_tests_under_it(tmp_path):
    """`pytest tests/` names the suite without naming a .py file."""
    proj = _suite(tmp_path)
    debugger = MultiFileDebugger(verbose=False)

    protected = debugger._protected_files(
        proj.resolve(), [sys.executable, "-m", "pytest", "tests/", "-q"])

    names = {p.name for p in protected}
    assert "test_mylib.py" in names, (
        "a directory argument protected nothing, so the reducer may stub "
        "these tests to `pass` and delete the library they cover")
    assert "conftest.py" in names, "the fixture file was left unprotected"


def test_a_test_run_that_protects_nothing_is_refused(tmp_path):
    """`pytest -k foo` names no path at all.

    There is nothing to infer here, and guessing would be worse than
    refusing: the reduction that follows an unprotected pytest command is
    reliably the degenerate one.
    """
    proj = _suite(tmp_path)
    debugger = MultiFileDebugger(verbose=False, python=sys.executable)

    with pytest.raises(ValueError, match="nothing to protect"):
        debugger.reduce_project(
            proj, [sys.executable, "-m", "pytest", "-k", "add", "-q"])


def test_a_projects_own_runner_counts_as_a_test_runner(tmp_path):
    """Django's suite runs only through tests/runtests.py.

    Without this the command reads as `python some_script.py`, where the
    script is the subject and reducing it is the point. `_protected_files`
    then returns nothing, the refusal guard never fires, and a *passing*
    test run this way has the degenerate solution wide open: empty the
    test, the runner still prints OK, and the library can go.
    """
    proj = _suite(tmp_path)
    (proj / "tests" / "runtests.py").write_text("raise SystemExit(0)\n")
    debugger = MultiFileDebugger(verbose=False)

    assert debugger._is_test_runner(
        [sys.executable, "tests/runtests.py", "mylib.SomeTests.test_x"])
    # A bare word is a label or a directory far more often than a script.
    assert not debugger._is_test_runner([sys.executable, "test", "-q"])


def test_a_dotted_label_protects_the_file_holding_the_test(tmp_path):
    """`runtests.py apps.tests.AppsTests.test_x` names no path.

    Only the runner matches as a path, so on Django the harness was
    protected and the file holding the test — the actual oracle — was
    left reducible. Measured on django__django-17029: 0 lines protected
    before, 1,332 after.
    """
    proj = _suite(tmp_path)
    (proj / "tests" / "runtests.py").write_text("raise SystemExit(0)\n")
    debugger = MultiFileDebugger(verbose=False)

    protected = debugger._protected_files(
        proj.resolve(),
        [sys.executable, "tests/runtests.py", "test_mylib.TestX.test_add"])

    names = {p.name for p in protected}
    assert "test_mylib.py" in names, (
        "the dotted label's file was not protected, so the reducer may "
        "cut the test that defines the behavior being preserved")


def test_a_label_naming_nothing_in_the_project_protects_nothing_extra(tmp_path):
    """Guessing at a label that resolves nowhere would be worse."""
    proj = _suite(tmp_path)
    debugger = MultiFileDebugger(verbose=False)

    found = debugger._files_named_by_label(
        proj.resolve(), ["some.module.that.is.not.here"])

    assert found == set()


def test_an_ordinary_module_named_tests_is_not_mistaken_for_a_runner():
    """A false positive here silently disables the reduction.

    "test.py" and "tests.py" are ordinary module names far more often
    than entry points. Treating one as a runner protects it as the
    oracle, so a project whose subject *is* that script reduces to
    nothing at all while reporting success.
    """
    assert not MultiFileDebugger._is_test_runner(["python3", "tests/test.py"])
    assert not MultiFileDebugger._is_test_runner(["python3", "apps/tests.py"])
    # The narrow set still covers the two that motivated it.
    assert MultiFileDebugger._is_test_runner(["python3", "tests/runtests.py"])
    assert MultiFileDebugger._is_test_runner(["python3", "bin/test", "x"])
