"""A single-file reduction driven by a command must run the candidate.

`Validator.validate` used to write the candidate to a temporary file and
then execute the caller's command, which reads the file the command names
— the *original*. Nothing ever ran the candidate, so every candidate
matched the original behavior and `automre reduce <file> -c "<command>"`
would happily delete the whole file.

This is the same category error as the four in the README's
trustworthiness section, surviving in the single-file path because every
audit went through `reduce-project`, whose ProjectFileValidator writes
candidates to the real file and was therefore always correct.

The probe here is the one that catches that class of bug: sabotage the
code the test depends on and require the test to notice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from cli import _oracle_is_the_subject  # noqa: E402
from reducer import HybridDeltaDebugger  # noqa: E402
from validator import Validator  # noqa: E402


LIB = '''\
def add(a, b):
    return a + b


def unused_one(x):
    return x * 2


def unused_two():
    return "dead"


def unused_three(n):
    total = 0
    for i in range(n):
        total += i
    return total
'''

TEST = '''\
import mylib


def test_add_works():
    assert mylib.add(2, 3) == 5
'''


def _build(tmp_path: Path) -> Path:
    proj = tmp_path / "single_file"
    proj.mkdir()
    (proj / "mylib.py").write_text(LIB)
    (proj / "test_mylib.py").write_text(TEST)
    return proj


def _pytest(project: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "test_mylib.py::test_add_works",
         "-q", "--no-header"],
        cwd=project, capture_output=True, text=True, timeout=120)


def test_a_command_without_a_target_file_is_refused():
    """The query is unanswerable, so it must not be answered.

    Silently validating the original file is what made the reduction
    vacuous. Refusing is the only honest response, and it is the response
    the multi-file path has taken since "Refuse to score a tree the
    reproduction command is not running".
    """
    validator = Validator()
    validator.set_original_behavior("1 passed", 0)

    with pytest.raises(ValueError, match="target_file"):
        validator.validate("x = 1\n", command=["true"], cwd=None)


def test_reduction_with_a_command_stays_sensitive_to_the_code(tmp_path):
    proj = _build(tmp_path)
    lib = proj / "mylib.py"
    command = [sys.executable, "-m", "pytest",
               "test_mylib.py::test_add_works", "-q", "--no-header"]

    assert _pytest(proj).returncode == 0, "fixture must pass first"

    validator = Validator(target_file=lib)
    reducer = HybridDeltaDebugger(validator=validator, verbose=False)
    result = reducer.reduce(source_code=lib.read_text(),
                            test_command=command,
                            cwd=proj,
                            file_path=lib)
    lib.write_text(result.minimized_code)

    assert _pytest(proj).returncode == 0, "reduction broke the test"
    assert "def add" in result.minimized_code, (
        "the function under test was deleted and the reduction still "
        "scored — the oracle is not running the candidate")

    # The vacuity probe: break `add` and require the test to fail.
    lib.write_text(result.minimized_code.replace("return a + b",
                                                 "return a * b + 1"))
    assert _pytest(proj).returncode != 0, (
        "the reduced file still passes with `add` sabotaged, so nothing "
        "in it was ever under test — vacuous reduction")


def test_reducing_the_passing_test_itself_is_refused(tmp_path):
    """The file being reduced must not be the file deciding the verdict.

    Making `-c` work at all made this reachable: the oracle now really
    does run the candidate, and if the candidate *is* the test, emptying
    it keeps the command passing. Same degenerate solution the multi-file
    path refuses, so refuse it here and point at that path.
    """
    proj = _build(tmp_path)
    test_file = proj / "test_mylib.py"
    command = [sys.executable, "-m", "pytest",
               "test_mylib.py::test_add_works", "-q", "--no-header"]

    reason = _oracle_is_the_subject(test_file, command)
    assert reason is not None, (
        "reducing the passing test against itself was allowed")
    assert "reduce-project" in reason, "the refusal gives no way forward"

    # Reducing the *library* against the same command is the ordinary
    # case and must stay allowed.
    assert _oracle_is_the_subject(proj / "mylib.py", command) is None


def test_a_failing_command_on_its_own_file_is_allowed(tmp_path):
    """`python bug.py` reducing bug.py is the original use case.

    It is self-protecting: the oracle compares a traceback naming the
    line that raised, so the reduction cannot fake the failure by
    deleting its cause.
    """
    proj = tmp_path / "script"
    proj.mkdir()
    bug = proj / "bug.py"
    bug.write_text("values = [1, 2, 3]\nprint(values['not-an-index'])\n")

    assert _oracle_is_the_subject(bug, [sys.executable, "bug.py"]) is None


def test_the_target_file_is_restored_after_a_rejected_candidate(tmp_path):
    """A rejected candidate must not survive on disk.

    The oracle mutates the real file to ask its question. If it did not put
    the original back, the next query would be asked about the wreckage of
    the last one.
    """
    proj = _build(tmp_path)
    lib = proj / "mylib.py"
    before = lib.read_text()

    validator = Validator(target_file=lib)
    validator.set_original_behavior("1 passed", 0)
    validator.validate("def add(a, b):\n    return None\n",
                       command=[sys.executable, "-m", "pytest",
                                "test_mylib.py::test_add_works", "-q"],
                       cwd=proj)

    assert lib.read_text() == before, "the candidate leaked onto disk"
