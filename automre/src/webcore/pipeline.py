"""The sequence that turns an extracted project into a reduced one.

Four steps, in an order that is not arbitrary:

    provision -> resolve the command -> gate -> reduce

Provisioning comes first because the gate has to run the command and the
command usually needs the project installed. Resolving comes next so that
every later run uses the job's own interpreter rather than whatever is on
PATH — getting that wrong means Phase 1 traces a different environment
than reductions are validated against, which has already produced a run
where 51 of 82 files were misclassified as unreachable. The gate comes
before the reducer because a command that is not really running the code
cannot be detected from the inside: every candidate truly does preserve
the behavior it was given.

This module is the engine both front doors drive — the FastAPI worker in
`web/worker/` and the in-sandbox runner — so that "what a reduction job
does" has one definition. It must not import `fastapi`, and must not
import anything from `web/`.
"""

from __future__ import annotations

import sys as _sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from multi_file import MultiFileDebugger  # noqa: E402
from provision import ProvisionError, provision  # noqa: E402
from provision import check as gate_check  # noqa: E402

from .archive import count_python  # noqa: E402
from .errors import JobError  # noqa: E402
from .state import JobState, ProgressProjector  # noqa: E402

# Left behind by the thousands of test runs a reduction makes, and not
# part of the reproducer anybody asked for.
_PACKAGE_EXCLUDE = {".pytest_cache", "__pycache__", ".git", ".mypy_cache",
                    ".ruff_cache", ".tox", ".venv", "venv"}

# Per-query ceiling. A query that runs this long is a candidate that does
# not terminate rather than a slow test; ten of them were once 76% of a
# task's entire query time.
DEFAULT_QUERY_TIMEOUT = 120


@dataclass
class PipelineOutcome:
    """What happened, in enough detail to report it without guessing."""
    interpreter: str
    command: List[str]
    readiness: Any = None           # provision.Readiness
    summary: Any = None             # MultiFileReductionResult, or None
    cancelled: bool = False

    @property
    def completed(self) -> bool:
        return self.summary is not None and not self.cancelled


# ------------------------------------------------------------- steps

def prepare_env(root: Path, venv_dir: Path) -> str:
    """A virtualenv with the project and its test dependencies in it.

    The reproduction command has to work before anything can be reduced,
    and for most real projects that means the package is importable. A
    failure here is reported as a failure to prepare rather than as a
    reduction result, because the difference matters: one is the user's
    environment, the other is the tool.

    `venv_dir` must sit outside the project tree, so the environment is
    never part of what the reducer walks. `provision` refuses otherwise.
    """
    try:
        spec = provision(root, venv_dir)
    except ProvisionError as exc:
        raise JobError(str(exc)) from exc
    return spec.python


def resolve_command(command: List[str], interpreter: str) -> List[str]:
    """Point a bare `pytest ...` at the job's own interpreter."""
    if not command:
        raise JobError("no reproduction command given")
    head = Path(command[0]).name.lower()
    if head in {"pytest", "py.test"}:
        return [interpreter, "-m", "pytest", *command[1:]]
    if head.startswith("python"):
        return [interpreter, *command[1:]]
    return list(command)


def check_ready(root: Path, command: List[str], timeout: int = 120):
    """Refuse to reduce a project the command is not really running.

    Without this a job naming a test that does not exist ran happily to
    "done": pytest exited 4, the oracle adopted *exiting 4* as the
    behavior to preserve, and the user downloaded a library reduced to
    blank lines with the interface reporting success.

    Returns the gate's verdict, which carries the baseline run and how
    long it took — the caller wants both rather than paying for them again.
    """
    verdict = gate_check(root, command, timeout=timeout)
    if not verdict.ok:
        raise JobError(f"the reproduction command is not usable: "
                       f"{verdict.reason}")
    return verdict


def package(out: Path, root: Path) -> Path:
    """Zip the reduced project, without the litter reduction leaves.

    Thousands of test runs happen in this directory, and each one writes
    `.pytest_cache` and `__pycache__`. Shipping those means the minimal
    reproducer a user downloads arrives with more directories of cache
    than files of code.
    """
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if any(part in _PACKAGE_EXCLUDE for part in path.parts):
                continue
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    return out


# ---------------------------------------------------------- the driver

def run_pipeline(root: Path, command: List[str], *,
                 venv_dir: Path,
                 state: JobState,
                 on_update: Optional[Callable[[JobState], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 deadline: Optional[float] = None,
                 interpreter: Optional[str] = None,
                 query_timeout: int = DEFAULT_QUERY_TIMEOUT,
                 ) -> PipelineOutcome:
    """Provision, gate and reduce `root` in place, updating `state`.

    `on_update` is called after every state change, including once per
    query. The worker's implementation is a no-op because its state object
    *is* the thing being polled; the sandbox runner's writes a file, and so
    has to throttle. Throttling belongs to the implementation, not here.

    `should_cancel` is polled between steps and at every progress event.
    On cancellation this returns normally with `cancelled=True` and leaves
    the tree in its last validated state, because a partly reduced project
    is a useful result and raising would discard it.

    Raises JobError for anything that is about the user's project.
    """
    cancelled = False

    def stop_requested() -> bool:
        if should_cancel is not None and should_cancel():
            return True
        return deadline is not None and time.time() > deadline

    def touch() -> None:
        if on_update is not None:
            on_update(state)

    state.original_files, state.original_lines = count_python(root)
    state.status = "preparing"
    state.message = "creating an environment and installing the project"
    touch()
    if stop_requested():
        return PipelineOutcome(interpreter or "", list(command), cancelled=True)

    interp = interpreter or prepare_env(root, venv_dir)
    resolved = resolve_command(command, interp)

    state.message = "checking the reproduction command"
    touch()
    if stop_requested():
        return PipelineOutcome(interp, resolved, cancelled=True)

    verdict = check_ready(root, resolved, timeout=query_timeout)
    # The gate already ran the command; keep what it measured so the state
    # can price the reduction before it starts.
    state.baseline_seconds = verdict.baseline_seconds
    state.status = "running"
    state.message = "starting reduction"
    touch()

    def on_progress(event: Dict[str, Any]) -> None:
        ProgressProjector.apply(state, event)
        touch()
        if stop_requested():
            # The reducer has no cancel of its own, and giving it one would
            # mean a kill switch inside the oracle loop. Raising from the
            # observer stops it at a query boundary, where the tree on disk
            # is a validated state rather than a half-written candidate.
            raise KeyboardInterrupt("cancelled")

    debugger = MultiFileDebugger(verbose=False, timeout=query_timeout,
                                 python=interp, progress=on_progress)
    summary = None
    try:
        summary = debugger.reduce_project(root, resolved)
    except KeyboardInterrupt:
        cancelled = True
    except ValueError as exc:
        # The refusals: an unprotected test-runner command, mainly.
        raise JobError(str(exc)) from exc

    if summary is not None:
        # Report the reducer's own accounting rather than recounting the
        # tree. protected_lines in particular cannot be recovered from the
        # output — it is the test file and its conftest, which are
        # identical before and after and so sit on both sides of the
        # headline figure, dragging it down for a reason that says nothing
        # about the reduction.
        state.protected_lines = summary.protected_line_count
        state.timed_out_queries = summary.timed_out_queries

    state.final_files, state.final_lines = count_python(root)
    touch()
    return PipelineOutcome(interp, resolved, readiness=verdict,
                           summary=summary, cancelled=cancelled)
