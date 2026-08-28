"""Job store and runner for the autoMRE web service.

One job is one reduction: a project archive plus a reproduction command
in, a reduced project archive out. Reductions take minutes to hours and
run thousands of subprocesses, so a job runs on a background thread and
the HTTP layer only ever reads a snapshot of its state.

The progress model is the part worth explaining. A reduction's unit of
work is the *query* — one run of the reproduction command — and the
number of queries a project needs is not knowable in advance, because it
depends on how much of the code turns out to be removable. So the
estimate is a calibrated guess (see MultiFileDebugger) and this module
turns it into a *range* rather than a countdown. An honest wide range is
more useful than a precise-looking number that is wrong.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "automre" / "src"))

from multi_file import MultiFileDebugger  # noqa: E402
# `discover` is re-exported for main.py, which asks this module for its
# imports so the sys.path bootstrap above stays in one file. It has to be
# named here to be re-exported, and once was not: the comment survived a
# refactor that dropped the name, and `uvicorn main:app` failed on the
# import line for weeks with a green test suite behind it, because
# nothing imported main.
from provision import ProvisionError, discover, provision  # noqa: E402
from provision import check as gate_check  # noqa: E402

# Upload guards. Reduction executes the code in the archive, so these are
# the first line of defence and not merely resource limits — see
# web/README.md, which does not pretend they are sufficient on their own.
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
MAX_FILES = 5000
# Wall-clock ceiling for one job. A reduction that has not converged by
# here is stopped and its partial tree is still returned, because a
# partly-reduced project is a useful result and a lost job is not.
MAX_JOB_SECONDS = 60 * 60

# How long a finished job keeps its disk, and how many finished jobs stay
# in the table at all. A job owns a temp directory holding a virtualenv,
# the extracted project and the result zip — of the order of a hundred
# megabytes — and nothing else on this machine deletes it. Without a
# retention policy the worker fills its own disk after a handful of
# reductions and then fails at `python -m venv`, reporting a problem with
# the machine for what is really a problem with this file.
RESULT_TTL_SECONDS = 60 * 60
MAX_REMEMBERED_JOBS = 200

# Left behind by the thousands of test runs a reduction makes, and not
# part of the reproducer anybody asked for.
_PACKAGE_EXCLUDE = {".pytest_cache", "__pycache__", ".git", ".mypy_cache",
                    ".ruff_cache", ".tox", ".venv", "venv"}

# How far the ETA range is spread either side of the point estimate. The
# per-line query factor is calibrated across repos that differ by a
# factor of four (requests 0.11 q/line, tomlkit 0.48), so a tight range
# would be a lie.
ETA_LOW = 0.55
ETA_HIGH = 2.0


class JobError(Exception):
    """A job could not be started, with a reason fit to show a user."""


@dataclass
class Job:
    job_id: str
    command: List[str]
    status: str = "queued"          # queued|preparing|running|done|failed|cancelled
    phase: str = "queued"
    message: str = "waiting to start"
    queries: int = 0
    estimated_queries: int = 0
    timed_out_queries: int = 0
    # Lines to examine, and lines examined. This is the honest progress
    # measure: the reducer walks the surviving files once and this is a
    # real fraction of that walk. Queries are shown too, but only as a
    # count of work done — the number a project *will* need varies
    # eightfold per line and cannot carry a progress bar.
    work_total: int = 0
    work_done: int = 0
    work_started_at: float = 0.0
    current_file: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Optional[str] = None

    original_files: int = 0
    original_lines: int = 0
    final_files: int = 0
    final_lines: int = 0
    protected_lines: int = 0

    work_dir: Optional[Path] = None
    result_zip: Optional[Path] = None
    # Set once the job's disk has been reclaimed, so a late download can
    # be told the result expired rather than that it never existed.
    expired: bool = False
    _cancel: threading.Event = field(default_factory=threading.Event)

    # -------------------------------------------------- derived views

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def fraction(self) -> Optional[float]:
        """How far through the file walk we are, or None before it starts."""
        if self.work_total <= 0:
            return None
        return min(1.0, self.work_done / self.work_total)

    def eta_range(self):
        """(low, high) seconds remaining, or None when it cannot be said.

        Extrapolates from lines-examined-per-second, measured over the
        file walk only — the phases before it are a fixed prelude whose
        cost says nothing about the rest.

        Stays None until a real fraction of the work is behind us. The
        first file is the largest (the reducer sorts that way), so an
        estimate drawn from it alone would be systematically pessimistic;
        and a wide range that appears late beats a confident one that
        appears immediately and is wrong.
        """
        if self.status != "running":
            return None
        frac = self.fraction
        if frac is None or frac <= 0.03:
            return None
        spent = time.time() - self.work_started_at if self.work_started_at else 0
        if spent <= 0:
            return None
        point = spent * (1 - frac) / frac
        return (point * ETA_LOW, point * ETA_HIGH)

    def snapshot(self) -> Dict:
        eta = self.eta_range()
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "queries": self.queries,
            "estimated_queries": self.estimated_queries,
            "timed_out_queries": self.timed_out_queries,
            "work_total": self.work_total,
            "work_done": self.work_done,
            "fraction": self.fraction,
            "current_file": self.current_file,
            "elapsed_seconds": round(self.elapsed, 1),
            "eta_low_seconds": round(eta[0]) if eta else None,
            "eta_high_seconds": round(eta[1]) if eta else None,
            "error": self.error,
            "original_files": self.original_files,
            "original_lines": self.original_lines,
            "final_files": self.final_files,
            "final_lines": self.final_lines,
            "protected_lines": self.protected_lines,
            "has_result": self.result_zip is not None,
            "expired": self.expired,
        }


# ------------------------------------------------------------ archives

def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack a zip, refusing anything that escapes `dest` or is huge.

    `zipfile.extractall` will happily write to `../../etc/whatever` if the
    archive says so. Every member is therefore resolved against the
    destination and rejected if it lands outside, and the uncompressed
    total is capped so a zip bomb cannot fill the disk.
    """
    total = 0
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if len(members) > MAX_FILES:
            raise JobError(f"archive holds {len(members)} entries; "
                           f"the limit is {MAX_FILES}")
        for info in members:
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise JobError("archive expands past "
                               f"{MAX_UNCOMPRESSED_BYTES // (1024*1024)} MB")
            target = (dest / info.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise JobError(f"archive entry escapes the extraction "
                               f"directory: {info.filename}")
        zf.extractall(dest)


def project_root(extracted: Path) -> Path:
    """The directory the project actually starts at.

    Zipping a folder usually produces a single top-level directory, and
    running the reproduction command one level above it would fail for a
    reason that has nothing to do with the user's project.
    """
    entries = [p for p in extracted.iterdir() if not p.name.startswith("__")]
    visible = [p for p in entries if not p.name.startswith(".")]
    if len(visible) == 1 and visible[0].is_dir():
        return visible[0]
    return extracted


def count_python(root: Path):
    files = [p for p in root.rglob("*.py")
             if not any(part in {".git", "__pycache__", ".venv", "venv"}
                        for part in p.parts)]
    lines = 0
    for f in files:
        try:
            lines += len(f.read_text().splitlines())
        except (OSError, UnicodeDecodeError):
            continue
    return len(files), lines


# ------------------------------------------------------------ registry

class JobRegistry:
    """In-memory job table.

    In memory on purpose: a job owns a temp directory on this machine's
    disk, so it cannot outlive the process that made it and there is
    nothing gained by pretending otherwise. Restarting the worker loses
    running jobs, which the UI reports rather than hides.
    """

    def __init__(self, max_concurrent: int = 1):
        # One at a time by default. A reduction saturates a core with
        # subprocesses, and running two halves the speed of both while
        # making every ETA wrong.
        self._sema = threading.Semaphore(max_concurrent)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in {"done", "failed", "cancelled"}:
            return False
        job._cancel.set()
        return True

    def sweep(self) -> None:
        """Reclaim the disk of jobs whose results have expired.

        Two tiers, because they answer different questions. A finished
        job loses its directory after RESULT_TTL_SECONDS but keeps its
        row, so a late poll gets "the result has expired" rather than
        "no such job" — the difference between an answer and a mystery.
        The row itself is dropped only once there are more than
        MAX_REMEMBERED_JOBS of them, which bounds the table on a worker
        that runs for weeks.

        Called when work arrives rather than on a timer: a sweeper thread
        would be a second thing to reason about for a saving that only
        matters at the moment a new job wants disk.
        """
        now = time.time()
        with self._lock:
            finished = sorted((j for j in self._jobs.values() if j.finished_at),
                              key=lambda j: j.finished_at)
            expired = [j for j in finished
                       if now - j.finished_at > RESULT_TTL_SECONDS]
            surplus = finished[:max(0, len(finished) - MAX_REMEMBERED_JOBS)]
            for job in surplus:
                self._jobs.pop(job.job_id, None)
            doomed = {j.job_id: (j, j.work_dir)
                      for j in expired + surplus if j.work_dir}.values()
            for job, _ in doomed:
                job.work_dir = None
                job.result_zip = None
                job.expired = True
        # Outside the lock: rmtree of a virtualenv is thousands of
        # unlinks, and polls should not queue behind it.
        for _, directory in doomed:
            shutil.rmtree(directory, ignore_errors=True)

    def submit(self, archive: Path, command: List[str],
               python: Optional[str] = None) -> Job:
        self.sweep()
        job = Job(job_id=uuid.uuid4().hex[:12], command=command)
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run, daemon=True,
                                  args=(job, archive, python))
        thread.start()
        return job

    # -------------------------------------------------------- internals

    def _run(self, job: Job, archive: Path, python: Optional[str]) -> None:
        job.started_at = time.time()
        tmp = Path(tempfile.mkdtemp(prefix="automre_job_"))
        job.work_dir = tmp
        try:
            self._sema.acquire()
            if job._cancel.is_set():
                job.status = "cancelled"
                return
            job.status = "preparing"
            job.message = "unpacking the project"
            extracted = tmp / "src"
            extracted.mkdir()
            safe_extract(archive, extracted)
            root = project_root(extracted)

            job.original_files, job.original_lines = count_python(root)

            interpreter = python or self._prepare_env(job, root)
            command = self._resolve_command(job.command, interpreter)
            self._check_ready(job, root, command)

            job.status = "running"
            job.message = "starting reduction"
            self._reduce(job, root, command, interpreter)

            if job._cancel.is_set():
                job.status = "cancelled"
                job.message = "cancelled; the partial result is still available"
            else:
                job.status = "done"

            job.final_files, job.final_lines = count_python(root)
            job.result_zip = self._package(tmp, root)

        except JobError as exc:
            job.status = "failed"
            job.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()
            try:
                self._sema.release()
            except ValueError:
                pass
            try:
                archive.unlink()
            except OSError:
                pass

    def _prepare_env(self, job: Job, root: Path) -> str:
        """A virtualenv with the project and its test dependencies in it.

        The reproduction command has to work before anything can be
        reduced, and for most real projects that means the package is
        importable. A failure here is reported as a failure to prepare
        rather than as a reduction result, because the difference matters:
        one is the user's environment, the other is the tool.

        The venv goes in the job directory, one level above the extracted
        project, so the environment is never part of the tree the reducer
        walks.
        """
        job.message = "creating an environment and installing the project"
        try:
            spec = provision(root, job.work_dir / "venv")
        except ProvisionError as exc:
            raise JobError(str(exc)) from exc
        return spec.python

    @staticmethod
    def _check_ready(job: Job, root: Path, command: List[str]) -> None:
        """Refuse to reduce a project the command is not really running.

        Without this a job naming a test that does not exist ran happily
        to "done": pytest exited 4, the oracle adopted *exiting 4* as the
        behavior to preserve, and the user downloaded a library reduced
        to blank lines with the interface reporting success. Reduction
        cannot detect that from the inside — every candidate really does
        preserve the behavior it was given — so it has to be caught
        before the first query.
        """
        job.message = "checking the reproduction command"
        verdict = gate_check(root, command, timeout=120)
        if not verdict.ok:
            raise JobError(f"the reproduction command is not usable: "
                           f"{verdict.reason}")

    def _reduce(self, job: Job, root: Path, command: List[str],
                interpreter: str) -> None:
        deadline = time.time() + MAX_JOB_SECONDS

        def on_progress(event: dict) -> None:
            job.phase = event.get("phase", job.phase)
            job.message = event.get("message", job.message)
            if "queries" in event:
                job.queries = event["queries"]
            if event.get("estimated_queries"):
                job.estimated_queries = event["estimated_queries"]
            if "timed_out_queries" in event:
                job.timed_out_queries = event["timed_out_queries"]
            if event.get("work_total"):
                if not job.work_started_at:
                    # The clock for the ETA starts when the file walk
                    # does, not when the job did. Everything before it —
                    # unpacking, pip install, the coverage trace — is a
                    # fixed prelude that would skew the rate.
                    job.work_started_at = time.time()
                job.work_total = event["work_total"]
            if "work_done" in event:
                job.work_done = event["work_done"]
            if event.get("current_file"):
                job.current_file = event["current_file"]
            if job._cancel.is_set() or time.time() > deadline:
                # The reducer has no cancel of its own, and giving it one
                # would mean a kill switch inside the oracle loop. Raising
                # from the observer stops it at a query boundary, where
                # the tree on disk is a validated state rather than a
                # half-written candidate.
                raise KeyboardInterrupt("cancelled")

        debugger = MultiFileDebugger(verbose=False, timeout=120,
                                     python=interpreter,
                                     progress=on_progress)
        try:
            summary = debugger.reduce_project(root, command)
        except KeyboardInterrupt:
            job._cancel.set()
            return
        except ValueError as exc:
            # The refusals: an unprotected test-runner command, mainly.
            raise JobError(str(exc)) from exc

        # Report the reducer's own accounting rather than recounting the
        # tree. protected_lines in particular cannot be recovered from
        # the output — it is the test file and its conftest, which are
        # identical before and after and so sit on both sides of the
        # headline figure, dragging it down for a reason that says
        # nothing about the reduction.
        job.protected_lines = summary.protected_line_count
        job.timed_out_queries = summary.timed_out_queries

    @staticmethod
    def _resolve_command(command: List[str], interpreter: str) -> List[str]:
        """Point a bare `pytest ...` at the job's own interpreter."""
        if not command:
            raise JobError("no reproduction command given")
        head = Path(command[0]).name.lower()
        if head in {"pytest", "py.test"}:
            return [interpreter, "-m", "pytest", *command[1:]]
        if head.startswith("python"):
            return [interpreter, *command[1:]]
        return list(command)

    @staticmethod
    def _package(tmp: Path, root: Path) -> Path:
        """Zip the reduced project, without the litter reduction leaves.

        Thousands of test runs happen in this directory, and each one
        writes `.pytest_cache` and `__pycache__`. Shipping those means
        the minimal reproducer a user downloads arrives with more
        directories of cache than files of code.
        """
        out = tmp / "reduced.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(root.rglob("*")):
                if any(part in _PACKAGE_EXCLUDE for part in path.parts):
                    continue
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())
        return out
