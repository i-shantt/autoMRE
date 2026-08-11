"""The ways `reduce-project` can be asked for something it must not do.

Two of these were found by running the CLI rather than by reading it,
and they are the same class of problem: the command took a wrong request
literally instead of refusing it.

    automre reduce-project ./proj -c "..." --output ./proj --force

deleted `./proj` — `--force` means "overwrite the output", and when the
output *is* the input that means "delete the input". The copy that was
supposed to replace it then died with FileNotFoundError, so the user was
left with a stack trace and no project.

    automre reduce-project ./proj -c "pytest tests/"

hit the oracle-protection refusal, which is correct, but the refusal was
raised rather than printed: the one user error in this CLI that answered
with a traceback, after leaving a stray copy behind that would make the
next attempt fail on "output already exists".
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

import cli  # noqa: E402


LIB = '''\
def add(a, b):
    return a + b
'''

TEST = '''\
import mylib


def test_add():
    assert mylib.add(1, 2) == 3
'''


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "tests").mkdir(parents=True)
    (proj / "mylib.py").write_text(LIB)
    (proj / "tests" / "test_mylib.py").write_text(TEST)
    return proj


def _args(project: Path, command: str, **over) -> Namespace:
    base = dict(project=str(project), command=command, in_place=False,
                output=None, force=False, verbose=False, timeout=60,
                strategy="exact", aggressive_inline=False,
                no_coverage_prune=False, use_learned_oracle=False,
                oracle_model=None, python=None)
    base.update(over)
    return Namespace(**base)


def test_output_that_is_the_project_is_refused_not_deleted(tmp_path, capsys):
    """`--output <the project> --force` used to delete the project."""
    proj = _project(tmp_path)

    code = cli.cmd_reduce_project(
        _args(proj, "python3 -m pytest tests/test_mylib.py",
              output=str(proj), force=True))

    assert code == 1
    assert "destroy the input" in capsys.readouterr().err
    # The point of the test: the source is still there.
    assert (proj / "mylib.py").read_text() == LIB
    assert (proj / "tests" / "test_mylib.py").exists()


def test_output_that_contains_the_project_is_refused(tmp_path, capsys):
    """The same deletion one level up: --output is the project's parent."""
    proj = _project(tmp_path)

    code = cli.cmd_reduce_project(
        _args(proj, "python3 -m pytest tests/test_mylib.py",
              output=str(tmp_path), force=True))

    assert code == 1
    assert "destroy the input" in capsys.readouterr().err
    assert (proj / "mylib.py").exists()


def test_unprotected_test_command_is_reported_not_raised(tmp_path, capsys):
    """A test runner naming no test path is refused in prose, not a traceback.

    And the copy made a moment earlier is cleaned up: it holds an
    unreduced tree, and leaving it turns the *next* attempt into a
    complaint about an existing directory.
    """
    proj = _project(tmp_path)

    code = cli.cmd_reduce_project(_args(proj, "python3 -m pytest"))

    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "names no test file" in err
    assert not (tmp_path / "proj_minimized").exists()


def test_a_real_reduction_still_works(tmp_path):
    """The guards above must not have made the ordinary path unreachable."""
    proj = _project(tmp_path)

    code = cli.cmd_reduce_project(
        _args(proj, "python3 -m pytest tests/test_mylib.py -x -q"))

    assert code == 0
    out = tmp_path / "proj_minimized"
    assert (out / "tests" / "test_mylib.py").read_text() == TEST
