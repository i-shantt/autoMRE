"""A finished job has to give its disk back.

Lives here rather than under `web/` so that `python3 -m pytest
automre/tests/` remains the one command that checks the repository.

Every job on the web worker owns a temp directory holding a virtualenv,
the extracted project and a result zip — of the order of a hundred
megabytes. Nothing deleted them. The failure that produces is not a leak
that someone notices: it is the seventh reduction dying inside `python -m
venv` with a disk-full error, which reads as a broken machine rather than
a missing eight lines in jobs.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "web" / "worker"))

import jobs as worker_jobs  # noqa: E402


def _finished_job(registry, tmp_path: Path, name: str, age: float):
    """A job that finished `age` seconds ago, with a directory on disk."""
    job = worker_jobs.Job(job_id=name, command=["pytest"])
    work = tmp_path / name
    work.mkdir()
    (work / "venv_stand_in.txt").write_text("x" * 1024)
    job.work_dir = work
    job.result_zip = work / "reduced.zip"
    job.result_zip.write_text("zip")
    job.status = "done"
    job.finished_at = time.time() - age
    registry._jobs[job.job_id] = job
    return job


def test_an_expired_job_loses_its_disk_but_keeps_its_row(tmp_path):
    registry = worker_jobs.JobRegistry()
    old = _finished_job(registry, tmp_path, "old",
                        worker_jobs.RESULT_TTL_SECONDS + 60)

    registry.sweep()

    assert not (tmp_path / "old").exists()
    # The row survives so a late download is told the result expired,
    # rather than being told the job never existed.
    assert registry.get("old") is old
    assert old.expired is True
    assert old.snapshot()["has_result"] is False


def test_a_fresh_job_keeps_everything(tmp_path):
    registry = worker_jobs.JobRegistry()
    fresh = _finished_job(registry, tmp_path, "fresh", 5)

    registry.sweep()

    assert (tmp_path / "fresh" / "reduced.zip").exists()
    assert fresh.expired is False


def test_a_running_job_is_never_swept(tmp_path):
    """`finished_at` is what marks a job done; a running one has none."""
    registry = worker_jobs.JobRegistry()
    live = _finished_job(registry, tmp_path, "live", 0)
    live.finished_at = 0.0
    live.status = "running"

    registry.sweep()

    assert (tmp_path / "live" / "reduced.zip").exists()
    assert live.work_dir is not None


def test_the_table_stops_growing(tmp_path, monkeypatch):
    """Rows are tiny, but a worker that runs for weeks still needs a bound."""
    monkeypatch.setattr(worker_jobs, "MAX_REMEMBERED_JOBS", 3)
    registry = worker_jobs.JobRegistry()
    for i in range(6):
        # Oldest first, so 0 and 1 and 2 are the ones to forget.
        _finished_job(registry, tmp_path, f"job{i}", age=100 - i)

    registry.sweep()

    remembered = sorted(registry._jobs)
    assert remembered == ["job3", "job4", "job5"]
    assert not (tmp_path / "job0").exists()
    assert (tmp_path / "job5" / "reduced.zip").exists()
