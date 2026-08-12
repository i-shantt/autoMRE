"""Build an environment in which a project's test command actually runs.

This is the step that caps how many repositories autoMRE can be measured
on. `evaluation/gistify_tasks.json` holds ten tasks across three repos
and `oracle_training_tasks.json` eighteen across two, and both are
hand-written, because adding a repository means a person working out how
to install it. The learned oracle has two cross-validation folds for the
same reason.

SWE-Hub (arXiv 2603.00575) names this the Env Agent: raw repository
snapshot in, reproducible environment out. Its code was never released,
so what is taken here is the shape of the idea, not an implementation.

The order below is the whole design, and it is chosen so that a project
which pins its own test dependencies keeps them:

    1. a fresh virtualenv, never with --system-site-packages
    2. coverage      -- autoMRE's requirement, not the project's
    3. pip install -e .            (the project and its runtime deps)
    4. the project's test extra, or its dev-requirements file
    5. pytest, and only if steps 3-4 did not already provide one

Step 5 last is what honours a project's own pin without parsing anything
to find it. flask's conftest reads a private pytest sentinel removed in
9.1, so a harness that force-installs its own pytest over the project's
bound turns every flask task into a collection error, and the reducer
then faithfully preserves *the collection error*.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Installed into every provisioned environment because autoMRE traces the
# reproduction command with it, and the trace has to happen in the same
# interpreter the command runs under or Phase 1 measures a different
# environment than the one reductions are validated against.
_COVERAGE = "coverage"

# Used only when the project supplies no pytest of its own. Pinned rather
# than floating so a provisioned environment is the same next month.
DEFAULT_PYTEST_PIN = "pytest==9.0.0"

# Checked in order; the first that exists wins. Extras are preferred over
# requirements files because a project that declares one is telling you
# exactly what its tests need.
_TEST_EXTRAS = ("test", "tests", "dev", "testing")
_DEV_REQUIREMENTS = (
    "requirements-dev.txt",
    "requirements_dev.txt",
    "dev-requirements.txt",
    "test-requirements.txt",
    "requirements/dev.txt",
    "requirements/test.txt",
)

_PIP_TIMEOUT = 900
_VENV_TIMEOUT = 180

# pip ends a failed install with boilerplate, so the last line of output
# is reliably the least informative line in it. Three real rejections
# from a fifteen-instance ingest read "note: This error originates from a
# subprocess, and is likely not a problem with pip.", "hint: See above
# for details." and "╰─> matplotlib" — one of which points at output the
# caller never sees. Skip the furniture and look for the diagnosis.
_NOISE = (
    "note: This error originates from a subprocess",
    "hint: See above for details",
    "See above for output",
    "full command:", "cwd:", "note: This is an issue with the package",
)
_SIGNAL = (
    "error:", "ERROR:", "Error:", "error ", "Traceback",
    "No matching distribution", "Could not find a version",
    "Failed building wheel", "Failed to build",
    "requires Python", "is not supported", "SyntaxError",
    "ModuleNotFoundError", "ImportError", "fatal:",
    "command not found", "No such file or directory",
)


def _diagnosis(output: str) -> str:
    """The line from a failed step most likely to explain it.

    Searches backwards for something that reads like an error, because
    the interesting line is usually near the end but almost never *is*
    the end.
    """
    lines = [ln.strip() for ln in output.strip().splitlines()]
    lines = [ln for ln in lines
             if ln and not ln.startswith(("╰", "╷", "│", "["))
             and not any(n in ln for n in _NOISE)]
    if not lines:
        return "no output"

    # An exception line is the root cause; pip's own "Failed building
    # wheel for X" is the symptom. Prefer the cause and name the package
    # alongside it, since a failed install rarely says which one broke.
    package = ""
    for line in reversed(lines):
        match = re.search(r"Failed building wheel for (\S+)", line)
        if match:
            package = f" (building {match.group(1)})"
            break
    for line in reversed(lines):
        if re.match(r"^[A-Za-z_.]*(Error|Exception)\b.*:", line):
            return (line + package)[:300]

    for line in reversed(lines):
        if any(s in line for s in _SIGNAL):
            return line[:300]
    return lines[-1][:300]


class ProvisionError(Exception):
    """An environment could not be built, with a reason fit to show a user."""


@dataclass
class Step:
    """One command that ran, and how it went."""
    name: str
    argv: List[str]
    returncode: int
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class EnvSpec:
    """A provisioned environment, and the record of how it was built."""
    python: str
    venv_dir: Path
    project_dir: Path
    installed_editable: bool = False
    pytest_source: str = "none"      # project | provisioned | none
    steps: List[Step] = field(default_factory=list)

    def log(self) -> str:
        """Every step, in order, for a failure report or a rejects file."""
        lines = []
        for step in self.steps:
            mark = "ok " if step.ok else "FAIL"
            lines.append(f"[{mark}] {step.name}: {' '.join(step.argv)}")
            if not step.ok and step.output:
                lines.append(f"        {step.output.strip()[-500:]}")
        return "\n".join(lines)


def provision(project_dir: Path,
              venv_dir: Path,
              *,
              base_python: Optional[str] = None,
              pytest_pin: str = DEFAULT_PYTEST_PIN,
              editable: bool = True) -> EnvSpec:
    """Build a virtualenv in which `project_dir`'s tests can run.

    Args:
        project_dir: the project. Not modified.
        venv_dir: where the environment goes. Must be outside project_dir
            — see below.
        base_python: interpreter to build the venv from; defaults to the
            one running autoMRE.
        pytest_pin: what to install if the project brings no pytest.
        editable: run `pip install -e .`. Off for projects that are a
            loose collection of scripts rather than a package.

    Raises:
        ProvisionError: the venv could not be built, or the project
            declared itself installable and then failed to install. A
            missing test extra is not an error; a broken project is.
    """
    project_dir = Path(project_dir).resolve()
    venv_dir = Path(venv_dir).resolve()

    # A virtualenv holds thousands of .py files. Put one inside the
    # project and the reducer treats every one of them as a candidate for
    # deletion, so this is refused rather than merely discouraged.
    if venv_dir == project_dir or project_dir in venv_dir.parents:
        raise ProvisionError(
            f"venv_dir {venv_dir} is inside the project being provisioned. "
            "The reducer would treat the environment as part of the project. "
            "Put it alongside, not within.")

    spec = EnvSpec(python="", venv_dir=venv_dir, project_dir=project_dir)

    # --system-site-packages is deliberately absent and must stay absent.
    # It is what revived the shadowing bug on a hosted runtime: the
    # project resolved to a copy installed outside the tree under
    # reduction, every deletion became a no-op, and the whole run's
    # numbers were void without a single visible error.
    _run(spec, "venv",
         [base_python or sys.executable, "-m", "venv", str(venv_dir)],
         timeout=_VENV_TIMEOUT, fatal=True)
    spec.python = str(_venv_python(venv_dir))

    _pip(spec, "coverage", [_COVERAGE], fatal=True)

    if editable and _is_installable(project_dir):
        _pip(spec, "project", ["-e", "."], cwd=project_dir, fatal=True)
        spec.installed_editable = True

    _install_test_dependencies(spec)

    if _has_pytest(spec.python):
        spec.pytest_source = "project"
    else:
        _pip(spec, "pytest", [pytest_pin], fatal=True)
        spec.pytest_source = "provisioned"

    return spec


# ------------------------------------------------------------- internals

def _venv_python(venv_dir: Path) -> Path:
    py = venv_dir / "bin" / "python"
    return py if py.exists() else venv_dir / "Scripts" / "python.exe"


def _is_installable(project_dir: Path) -> bool:
    return ((project_dir / "pyproject.toml").exists()
            or (project_dir / "setup.py").exists()
            or (project_dir / "setup.cfg").exists())


def _has_pytest(python: str) -> bool:
    proc = subprocess.run([python, "-c", "import pytest"],
                          capture_output=True, text=True)
    return proc.returncode == 0


def _install_test_dependencies(spec: EnvSpec) -> None:
    """The project's own test dependencies, if it declares any.

    Not fatal. Plenty of projects need nothing beyond pytest, and a
    missing extra should not sink an otherwise healthy instance — the
    readiness gate will catch it if the test genuinely cannot run.
    """
    extra = _declared_test_extra(spec.project_dir)
    if extra and spec.installed_editable:
        _pip(spec, f"extra[{extra}]", ["-e", f".[{extra}]"],
             cwd=spec.project_dir)
        return

    for name in _DEV_REQUIREMENTS:
        path = spec.project_dir / name
        if path.exists():
            _pip(spec, name, ["-r", str(path)], cwd=spec.project_dir)
            return


def _declared_test_extra(project_dir: Path) -> Optional[str]:
    """The project's test extra, read from pyproject.toml.

    Returns None when there is no pyproject, no optional-dependencies
    table, no recognised name, or no tomllib (Python 3.10). All four mean
    the same thing to the caller: fall back to a requirements file.
    """
    pyproject = project_dir / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:
        return None
    try:
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError):
        return None
    extras = data.get("project", {}).get("optional-dependencies", {})
    return next((name for name in _TEST_EXTRAS if name in extras), None)


def _pip(spec: EnvSpec, name: str, args: List[str],
         cwd: Optional[Path] = None, fatal: bool = False) -> Step:
    return _run(spec, name,
                [spec.python, "-m", "pip", "install", "--quiet", *args],
                cwd=cwd, timeout=_PIP_TIMEOUT, fatal=fatal)


def _build_env() -> dict:
    """The environment a build runs in.

    A project is provisioned from a copy that deliberately has no .git —
    the reducer should never see a repository's history as deletable
    material. Projects versioned by setuptools-scm read their version out
    of that history, so without this they fail to build at all, which is
    how pytest was rejected from an ingest run with "LookupError:
    setuptools-scm was unable to detect version".

    The version is deliberately high rather than 0.0.0. A project that
    reads its own version usually does so to enforce a *minimum*: pytest's
    pyproject sets `minversion = 2.0`, so pretending 0.0.0 made pytest
    refuse to run its own test suite with "'minversion' requires
    pytest-2.0, actual pytest-0.0.0". Projects asserting an upper bound on
    themselves are rare enough to accept as the trade.

    A caller who has already set the variable keeps their value, which is
    the way to supply a real version when one is known.
    """
    env = dict(os.environ)
    env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "9999.0.0")
    return env


def _run(spec: EnvSpec, name: str, argv: List[str],
         cwd: Optional[Path] = None, timeout: int = _PIP_TIMEOUT,
         fatal: bool = False) -> Step:
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, env=_build_env())
        step = Step(name, argv, proc.returncode,
                    (proc.stderr or "") + (proc.stdout or ""))
    except subprocess.TimeoutExpired:
        step = Step(name, argv, -1, f"timed out after {timeout}s")
    except OSError as exc:
        step = Step(name, argv, -1, str(exc))

    spec.steps.append(step)
    if fatal and not step.ok:
        raise ProvisionError(
            f"could not provision the environment at step '{name}': "
            + _diagnosis(step.output))
    return step


def discard(spec: EnvSpec) -> None:
    """Delete a provisioned environment. Safe to call twice."""
    shutil.rmtree(spec.venv_dir, ignore_errors=True)
