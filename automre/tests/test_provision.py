"""The provisioning module: build an environment, find tests, judge fitness.

The gate cases are the interesting ones. Each names a failure this
benchmark actually shipped — a test that never ran, a collection error
adopted as the behavior to preserve, a package resolving outside the tree
being cut — and requires the gate to refuse it. A gate that passes
everything is the same as no gate, and that is the state every one of
those failures was discovered in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from provision import (  # noqa: E402
    ProvisionError,
    check,
    discover,
    locate,
    node_id_exists,
    provision,
    top_level_packages,
)
# Private, and tested directly: it decides the one line a user sees when
# an environment refuses to build.
from provision.environment import _diagnosis  # noqa: E402

LIB = '''\
def add(a, b):
    return a + b
'''

TEST = '''\
import mylib


def test_add():
    assert mylib.add(1, 2) == 3
'''

PYPROJECT = '''\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gatedemo"
version = "0.1.0"

[tool.setuptools]
packages = ["mylib"]
'''


def _package_project(tmp_path: Path, name: str = "proj") -> Path:
    """A real installable package, so `import mylib` means something."""
    proj = tmp_path / name
    (proj / "mylib").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "mylib" / "__init__.py").write_text(LIB)
    (proj / "tests" / "test_mylib.py").write_text(TEST)
    (proj / "pyproject.toml").write_text(PYPROJECT)
    return proj


def _flat_project(tmp_path: Path) -> Path:
    """No package, no pyproject — the shape a web upload usually has."""
    proj = tmp_path / "flat"
    (proj / "tests").mkdir(parents=True)
    (proj / "mylib.py").write_text(LIB)
    (proj / "tests" / "test_mylib.py").write_text(TEST)
    return proj


# ------------------------------------------------------------- discovery

def test_discover_finds_functions_and_methods(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "test_things.py").write_text(
        "def test_one():\n    pass\n\n"
        "class TestGroup:\n    def test_two(self):\n        pass\n\n"
        "def helper():\n    pass\n")

    found = discover(proj)

    assert "test_things.py::test_one" in found
    assert "test_things.py::TestGroup::test_two" in found
    assert not any("helper" in f for f in found)


def test_discover_skips_virtualenvs(tmp_path):
    """A venv is full of test files belonging to somebody else's project."""
    proj = tmp_path / "p"
    (proj / ".venv" / "lib").mkdir(parents=True)
    (proj / ".venv" / "lib" / "test_someone_elses.py").write_text(
        "def test_x():\n    pass\n")
    (proj / "test_mine.py").write_text("def test_mine():\n    pass\n")

    found = discover(proj)

    assert found == ["test_mine.py::test_mine"]


def test_node_id_exists_rejects_a_test_that_is_not_there(tmp_path):
    """requests-cookie_utils named a test v2.32.3 does not have."""
    proj = tmp_path / "p"
    proj.mkdir()
    (proj / "test_a.py").write_text("def test_real():\n    pass\n")

    assert node_id_exists(proj, "test_a.py::test_real")
    assert not node_id_exists(proj, "test_a.py::test_imaginary")
    assert not node_id_exists(proj, "test_missing_file.py::test_real")


def test_locate_resolves_a_bare_name_to_a_node_id(tmp_path):
    """Sympy names the function only; -k would collect the whole suite."""
    proj = _flat_project(tmp_path)
    (proj / "tests" / "test_more.py").write_text(
        "def test_issue_24543():\n    assert True\n")

    assert locate(proj, "test_issue_24543") == \
        ["tests/test_more.py::test_issue_24543"]
    assert locate(proj, "test_nothing_here") == []


def test_locate_reports_every_match_so_an_ambiguous_name_is_visible(tmp_path):
    """Two tests share a name: resolving to either one would be a guess."""
    proj = _flat_project(tmp_path)
    (proj / "tests" / "test_a.py").write_text("def test_dup():\n    pass\n")
    (proj / "tests" / "test_b.py").write_text("def test_dup():\n    pass\n")

    assert len(locate(proj, "test_dup")) == 2


# ----------------------------------------------------------- environment

def test_provision_refuses_a_venv_inside_the_project(tmp_path):
    """The reducer would treat thousands of venv files as candidates."""
    proj = _flat_project(tmp_path)

    with pytest.raises(ProvisionError, match="inside the project"):
        provision(proj, proj / "venv")


def test_provision_builds_a_usable_environment(tmp_path):
    proj = _package_project(tmp_path)

    spec = provision(proj, tmp_path / "env")

    assert Path(spec.python).exists()
    assert spec.installed_editable is True
    assert spec.pytest_source in ("project", "provisioned")
    assert all(step.ok for step in spec.steps), spec.log()
    # The point of provisioning: the project is importable afterwards.
    import subprocess
    proc = subprocess.run([spec.python, "-c", "import mylib, coverage, pytest"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_provision_skips_the_editable_install_without_a_package(tmp_path):
    proj = _flat_project(tmp_path)

    spec = provision(proj, tmp_path / "env")

    assert spec.installed_editable is False
    assert Path(spec.python).exists()


# ------------------------------------------------------- failure messages

def test_the_reason_is_the_error_not_pips_closing_boilerplate():
    """pip's last line is reliably its least useful one.

    Three real rejections out of a fifteen-instance ingest read "note:
    This error originates from a subprocess...", "hint: See above for
    details." and "╰─> matplotlib" — the second pointing at output the
    caller never sees.
    """
    output = (
        "  × Building wheel for matplotlib did not run successfully.\n"
        "  ╰─> [1863 lines of output]\n"
        "      RuntimeError: Failed to download FreeType 2.6.1\n"
        "  note: This error originates from a subprocess, and is likely "
        "not a problem with pip.\n"
        "  ERROR: Failed building wheel for matplotlib\n")

    reason = _diagnosis(output)

    assert "FreeType" in reason           # the cause, not the symptom
    assert "matplotlib" in reason         # and which package broke
    assert "originates from a subprocess" not in reason


def test_the_reason_prefers_a_root_cause_over_a_trailing_hint():
    assert _diagnosis("  SyntaxError: invalid syntax\n"
                      "  hint: See above for details.") \
        == "SyntaxError: invalid syntax"


def test_a_plain_pip_error_survives_unchanged():
    """No exception line to prefer; the ERROR line is already the point."""
    line = "ERROR: No matching distribution found for numpy==1.19.2"
    assert _diagnosis(line) == line


def test_no_output_says_so_rather_than_crashing():
    assert _diagnosis("") == "no output"
    assert _diagnosis("   \n\n  ") == "no output"


# ------------------------------------------------------------------ gate

def test_gate_accepts_a_healthy_project(tmp_path):
    proj = _flat_project(tmp_path)

    verdict = check(proj, [sys.executable, "-m", "pytest",
                           "tests/test_mylib.py::test_add", "-x", "-q"])

    assert verdict.ok, verdict.reason
    assert verdict.baseline_rc == 0
    assert verdict.command_runs >= 2


def test_gate_rejects_a_test_that_does_not_exist(tmp_path):
    proj = _flat_project(tmp_path)

    verdict = check(proj, [sys.executable, "-m", "pytest",
                           "tests/test_mylib.py::test_imaginary", "-x", "-q"])

    assert not verdict.ok
    assert "does not contain" in verdict.reason


def test_gate_rejects_a_failing_test_only_when_asked_to(tmp_path):
    """A crash is the normal reproduction command; the benchmark differs.

    Gistify-style tasks preserve the behavior of a *passing* test, so a
    non-zero exit means the task is broken. Everywhere else the command
    is the bug, and rejecting it would refuse every real bug report.
    """
    proj = _flat_project(tmp_path)
    (proj / "mylib.py").write_text("def add(a, b):\n    return a - b\n")
    command = [sys.executable, "-m", "pytest",
               "tests/test_mylib.py", "-x", "-q"]

    assert check(proj, command).ok
    strict = check(proj, command, require_pass=True)
    assert not strict.ok
    assert "did not pass" in strict.reason


def test_gate_accepts_a_crash_as_the_behavior_to_preserve(tmp_path):
    """The positive control must not read a real crash as vacuous.

    Without the package the crash becomes ModuleNotFoundError, and both
    exit 1 — so comparing exit codes alone would reject this.
    """
    proj = tmp_path / "crasher"
    (proj / "mylib").mkdir(parents=True)
    (proj / "mylib" / "__init__.py").write_text(
        "def boom():\n    return 1 / 0\n")
    (proj / "main.py").write_text("import mylib\n\nmylib.boom()\n")

    verdict = check(proj, [sys.executable, "main.py"])

    assert verdict.ok, verdict.reason
    assert verdict.baseline_rc == 1
    assert "ZeroDivisionError" in verdict.baseline_output


def test_gate_rejects_a_collection_error(tmp_path):
    """The flask tasks errored during collection and still scored 1."""
    proj = _flat_project(tmp_path)
    (proj / "tests" / "conftest.py").write_text(
        "raise RuntimeError('conftest is broken')\n")

    verdict = check(proj, [sys.executable, "-m", "pytest",
                           "tests/test_mylib.py", "-x", "-q"])

    assert not verdict.ok


def test_a_rejection_quotes_the_error_instead_of_guessing(tmp_path):
    """The category alone sends people to the wrong place.

    Two SWE-bench instances were rejected as "bad node id?" when their
    node ids were correct: sphinx imports imghdr, gone in Python 3.13,
    and xarray reads np.unicode_, gone in NumPy 2.0. Both look like a
    typo in the command until the reason says otherwise.
    """
    proj = _flat_project(tmp_path)
    (proj / "tests" / "conftest.py").write_text(
        "import nonexistent_module_xyz\n")

    verdict = check(proj, [sys.executable, "-m", "pytest",
                           "tests/test_mylib.py", "-x", "-q"])

    assert not verdict.ok
    assert "nonexistent_module_xyz" in verdict.reason


def test_gate_rejects_a_flaky_command(tmp_path):
    """A test that flips run to run records noise as behavior."""
    proj = tmp_path / "flaky"
    (proj / "tests").mkdir(parents=True)
    (proj / "tests" / "test_coin.py").write_text(
        "import pathlib\n\n\n"
        "def test_coin():\n"
        "    stamp = pathlib.Path(__file__).parent / '.seen'\n"
        "    first = not stamp.exists()\n"
        "    stamp.touch()\n"
        "    assert first\n")

    verdict = check(proj, [sys.executable, "-m", "pytest",
                           "tests/test_coin.py", "-x", "-q"])

    assert not verdict.ok
    assert "not deterministic" in verdict.reason


def test_gate_rejects_a_tree_whose_code_is_not_under_test(tmp_path):
    """The shadowing bug, reproduced: the test passes without the package.

    This is the failure that voided a whole cloud run. The test imports
    nothing from the package beside it, so deleting the package changes
    nothing and every reduction measured here would be vacuous.
    """
    proj = tmp_path / "shadowed"
    (proj / "mylib").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "mylib" / "__init__.py").write_text(LIB)
    (proj / "tests" / "test_nothing.py").write_text(
        "def test_passes_regardless():\n    assert True\n")

    verdict = check(proj, [sys.executable, "-m", "pytest",
                           "tests/test_nothing.py", "-x", "-q"])

    assert not verdict.ok
    assert "vacuous" in verdict.reason


def test_gate_restores_the_tree_it_sabotaged(tmp_path):
    """The positive control renames a package aside. It must put it back.

    The gate is handed the caller's real project, not a copy, so a
    sabotage it forgets to undo is a directory the user loses.
    """
    proj = _package_project(tmp_path)

    check(proj, [sys.executable, "-m", "pytest",
                 "tests/test_mylib.py", "-x", "-q"])

    assert (proj / "mylib" / "__init__.py").read_text() == LIB
    assert (proj / "tests" / "test_mylib.py").read_text() == TEST
    assert not list(proj.rglob("*.automre_sabotage"))


def test_top_level_packages_ignores_tests_and_dotdirs(tmp_path):
    proj = tmp_path / "p"
    for name in ("mylib", "tests", "docs", ".hidden", "_private"):
        (proj / name).mkdir(parents=True)
        (proj / name / "__init__.py").write_text("")

    assert top_level_packages(proj) == ["mylib"]
