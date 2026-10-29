"""Job store for the autoMRE web service.

One job is one reduction: a project archive plus a reproduction command
in, a reduced project archive out. Reductions take minutes to hours and
run thousands of subprocesses, so a job runs on a background thread and
the HTTP layer only ever reads a snapshot of its state.

What a reduction job *does* lives in `automre/src/webcore/` and is shared
with the runner that executes inside a sandbox. What is left here is the
part that is genuinely about this process: an in-memory table, a thread
per job, a semaphore, and a retention policy for the disk those jobs own.
"""

from __future__ import annotations

# Kept deliberately: the end-to-end tests reach for `worker_jobs.shutil`
# to clean up after a job, so this is part of the module's surface even
# when the code below is the only other user. A linter calling it
# removable is wrong here.
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "automre" / "src"))

# `discover` is re-exported for main.py, which asks this module for its
# imports so the sys.path bootstrap above stays in one file. It has to be
# named here to be re-exported, and once was not: the comment survived a
# refactor that dropped the name, and `uvicorn main:app` failed on the
# import line for weeks with a green test suite behind it, because
# nothing imported main.
from provision import discover  # noqa: E402
from webcore import (  # noqa: E402
    MAX_ARCHIVE_BYTES,
    MAX_FILES,
    MAX_UNCOMPRESSED_BYTES,
    JobError,
    JobState,
    build_manifest,
    count_python,
    package,
    project_root,
    run_pipeline,
    safe_extract,
)

# Named explicitly because `discover` is re-exported rather than used
# here, so a linter calls it unused and the obvious fix is to delete it —
# which is exactly how `uvicorn main:app` came to fail on its import
# line. __all__ makes the re-export structural instead of a comment
# asking the next reader not to tidy it away.
__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_FILES",
    "MAX_UNCOMPRESSED_BYTES",
    "JobError",
    "JobRegistry",
    "count_python",
    "discover",
    "project_root",
    "safe_extract",
]

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


@dataclass
class Job(JobState):
    """A JobState plus the disk it owns on this machine.

    The split is the point: everything in JobState can be serialized and
    read back somewhere else, and everything added here cannot. A sandbox
    runner reports the base class through a file; only a process that
    holds the temp directory can answer `has_result`.
    """

    work_dir: Optional[Path] = None
    result_zip: Optional[Path] = None
    # Where the project ended up on disk, and where its untouched copy
    # went. Both are needed to answer "show me what was removed from this
    # file", which is the one question the result zip cannot answer on
    # its own.
    project_root: Optional[Path] = None
    # Held in memory rather than on disk because it is small and because
    # the retention sweep reclaims the directory while keeping the row:
    # a reader whose download expired can still be shown what the
    # reduction did.
    manifest: Optional[Dict] = None
    # Set once the job's disk has been reclaimed, so a late download can
    # be told the result expired rather than that it never existed.
    expired: bool = False
    _cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> Dict:
        snap = super().snapshot()
        snap["has_result"] = self.result_zip is not None
        snap["has_manifest"] = self.manifest is not None
        snap["expired"] = self.expired
        return snap


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
            job.project_root = root

            outcome = run_pipeline(
                root, job.command,
                venv_dir=tmp / "venv",
                state=job,
                # The job object *is* what the HTTP layer polls, so a
                # progress event has already landed by the time the
                # projector returns. Nothing to push anywhere.
                on_update=None,
                should_cancel=job._cancel.is_set,
                deadline=time.time() + MAX_JOB_SECONDS,
                interpreter=python,
                originals_dir=tmp / "before",
            )

            # The per-file account, written here rather than only in the
            # sandbox runner so both front doors answer /manifest and the
            # UI can be developed against a local worker.
            job.manifest = build_manifest(job.job_id, job, root, job.command,
                                          outcome.summary)

            if outcome.cancelled:
                job._cancel.set()
                job.status = "cancelled"
                job.message = "cancelled; the partial result is still available"
            else:
                job.status = "done"

            job.result_zip = package(tmp / "reduced.zip", root)

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
