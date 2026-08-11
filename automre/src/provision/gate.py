"""Decide whether a project is fit to be reduced, before reducing it.

Every number this project has ever had to withdraw was withdrawn for the
same reason: the harness reduced something it was not measuring, and
nothing said so. A test that never ran, a collection error adopted as the
behavior to preserve, a package resolving to a copy outside the tree
being cut. Each scored a perfect execution fidelity on the way past.

So a project is not "ready" because it looks ready. It is ready when the
named test exists, the command runs and passes, it does the same thing
twice, and deleting the implementation makes it stop. Anything else is
rejected with a reason.

SWE-Hub (arXiv 2603.00575) puts a verification gate in the same place,
between building an environment and calling it a task. The checks here
are this project's own — they are the six failures it has already paid
for, turned into preconditions.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys as _sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from validator import Validator  # noqa: E402

from .discovery import node_id_exists  # noqa: E402


@dataclass
class Readiness:
    """The verdict, plus the baseline the checks already paid for."""
    ok: bool
    reason: Optional[str]
    baseline_output: str = ""
    baseline_rc: int = 0
    command_runs: int = 0

    def __bool__(self) -> bool:
        return self.ok


def check(project_dir: Path, command: Sequence[str],
          timeout: int = 120) -> Readiness:
    """Whether `command` in `project_dir` is something worth reducing.

    Returns a Readiness whose `reason` names the problem when it is not.
    On success `baseline_output` / `baseline_rc` carry the run the caller
    would otherwise have to make itself.

    Costs three or more runs of the command: one baseline, one repeat to
    catch flakiness, and one per top-level package for the positive
    control. That is the price of not discovering afterwards that a run
    measured nothing, and it is paid once per project rather than once
    per query.
    """
    project_dir = Path(project_dir).resolve()
    command = list(command)

    missing = _missing_node_ids(project_dir, command)
    if missing:
        return Readiness(False, f"the command names {missing[0]}, which this "
                                f"project does not contain")

    output, rc = _run(project_dir, command, timeout)
    runs = 1

    unhealthy = baseline_health(output, rc)
    if unhealthy:
        return Readiness(False, unhealthy, output, rc, runs)

    repeat_output, repeat_rc = _run(project_dir, command, timeout)
    runs += 1
    if repeat_rc != rc:
        return Readiness(False,
                         f"the command is not deterministic: exit {rc} then "
                         f"exit {repeat_rc}. Reduction against a flaky test "
                         f"records noise as behavior", output, rc, runs)
    if (Validator._normalize_output(output)
            != Validator._normalize_output(repeat_output)):
        return Readiness(False,
                         "the command produces different output on two "
                         "consecutive runs, so the oracle cannot tell a "
                         "broken candidate from an unlucky one",
                         output, rc, runs)

    vacuous, control_runs = _reduction_would_be_vacuous(
        project_dir, command, rc, timeout)
    runs += control_runs
    if vacuous:
        return Readiness(False, vacuous, output, rc, runs)

    return Readiness(True, None, output, rc, runs)


# --------------------------------------------------------------- checks

def baseline_health(output: str, rc: int) -> Optional[str]:
    """Reason the baseline is unusable, or None if it looks real.

    Execution fidelity only means something if the target test actually
    ran and passed before reduction. When it didn't, the harness happily
    adopts the failure itself as the behavior to preserve, and the
    reducer gets rewarded for deleting whatever isn't needed to keep
    reproducing it — which can be almost the entire repo.

    Both failure modes this catches were live in the manifest:
    requests-cookie_utils named a test that does not exist in v2.32.3
    (pytest exits 4, "no tests ran"), and the flask tasks errored during
    collection under a too-new pytest. Neither announced itself; both
    scored execution fidelity 1.
    """
    lowered = output.lower()

    if rc == 4:
        return "pytest usage error (rc=4) — bad node id?"
    if rc == 5:
        return "no tests collected (rc=5)"
    if "no tests ran" in lowered:
        return "no tests ran"
    if "error: not found:" in lowered:
        return "test node id not found"
    if re.search(r"\b\d+ errors? in [\d.]+s", lowered):
        return "collection/setup error before any assertion"
    if rc != 0:
        return f"target test did not pass (rc={rc})"
    return None


def _reduction_would_be_vacuous(project_dir: Path, command: List[str],
                                baseline_rc: int,
                                timeout: int) -> tuple:
    """Positive control: remove the code and require the test to notice.

    Asking where a package *resolves* is not the same question. An
    editable install can resolve into the tree at baseline and then
    silently fall through to another copy the moment the reducer deletes
    the package's files — which is exactly when it matters, and exactly
    the shape of failure that has cost this benchmark two audits.

    So instead of inspecting paths, reproduce the reducer's own most
    destructive move: rename the package directory aside, run the
    reproduction command, and require it to stop passing. A tree that
    still reproduces with its implementation missing is not being
    measured, and every reduction number from it is meaningless.

    Renaming rather than editing is deliberate. Appending `raise` to
    `__init__.py` would prove nothing: the file still exists, so the
    import never falls back, and the check would pass on precisely the
    setup it is meant to catch.

    Returns (reason or None, number of command runs spent).
    """
    packages = top_level_packages(project_dir)
    if not packages:
        return None, 0

    checked: List[str] = []
    runs = 0
    for pkg in packages:
        pkg_dir = next((root / pkg
                        for root in (project_dir, project_dir / "src")
                        if (root / pkg).is_dir()), None)
        if pkg_dir is None:
            continue
        hidden = pkg_dir.with_name(pkg_dir.name + ".automre_sabotage")
        pkg_dir.rename(hidden)
        try:
            purge_bytecode(project_dir)
            _, rc = _run(project_dir, command, timeout)
            runs += 1
            still_reproduces = (rc == baseline_rc)
        finally:
            hidden.rename(pkg_dir)
            purge_bytecode(project_dir)
        checked.append(pkg)
        if not still_reproduces:
            return None, runs   # removing the code broke the test: good

    if not checked:
        return None, runs
    return (f"removing {'/'.join(checked)} entirely did not change the "
            f"result — the command is not running the code in this "
            f"directory, so any reduction measured here is vacuous"), runs


# -------------------------------------------------------------- helpers

def top_level_packages(project_dir: Path) -> List[str]:
    """Importable package names shipped by the project under test."""
    names: List[str] = []
    for root in (project_dir, project_dir / "src"):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if (child.is_dir() and (child / "__init__.py").exists()
                    and not child.name.startswith((".", "_"))
                    and child.name not in ("tests", "test", "docs",
                                           "examples")):
                names.append(child.name)
    return names


def purge_bytecode(root: Path) -> None:
    """Drop every __pycache__ under root so the next run compiles sources.

    A stale .pyc matching on (mtime, size) let the oracle judge code that
    was no longer on disk. That is what the "nondeterminism" turned out
    to be, and it voided every number taken before it was found.
    """
    for cache_dir in Path(root).rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)


def _missing_node_ids(project_dir: Path, command: List[str]) -> List[str]:
    """Arguments that look like pytest node ids but name nothing here."""
    missing = []
    for arg in command:
        if arg.startswith("-"):
            continue
        head = arg.split("::")[0]
        if not head.endswith(".py"):
            continue
        if not node_id_exists(project_dir, arg):
            missing.append(arg)
    return missing


def _run(project_dir: Path, command: List[str], timeout: int) -> tuple:
    purge_bytecode(project_dir)
    try:
        proc = subprocess.run(command, cwd=project_dir, capture_output=True,
                              text=True, timeout=timeout)
        return (proc.stdout or "") + (proc.stderr or ""), proc.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1
    except OSError as exc:
        return f"could not run the command: {exc}", -1
