"""One job, all the way through: zip in, reduced project out.

The web worker had no test that ran a job. Its provisioning step is now
`provision()` rather than its own inline pip calls, and a refactor of the
step that builds the environment is exactly the kind that keeps every
unit test green while making the service do nothing.

Lives under automre/tests so `python3 -m pytest automre/tests/` stays the
one command that checks the repository.
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "web" / "worker"))

import jobs as worker_jobs  # noqa: E402

LIB = '''\
def add(a, b):
    return a + b


def unused_one():
    return "dead"


def unused_two():
    return "also dead"
'''

TEST = '''\
import mylib


def test_add():
    assert mylib.add(2, 3) == 5
'''


def _archive(tmp_path: Path) -> Path:
    """A zipped project with a single top-level directory, as uploads are."""
    src = tmp_path / "myproject"
    (src / "tests").mkdir(parents=True)
    (src / "mylib.py").write_text(LIB)
    (src / "tests" / "test_mylib.py").write_text(TEST)

    archive = tmp_path / "upload.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(tmp_path).as_posix())
    return archive


def _wait(job, timeout: float = 300.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.status in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.25)
    raise AssertionError(f"job did not finish; stuck at {job.status!r} "
                         f"({job.phase}: {job.message})")


def test_a_submitted_job_reduces_and_packages(tmp_path):
    registry = worker_jobs.JobRegistry()

    job = registry.submit(_archive(tmp_path),
                          ["pytest", "tests/test_mylib.py::test_add", "-x", "-q"])
    _wait(job)

    assert job.status == "done", job.error
    assert job.original_lines > job.final_lines, "nothing was removed"
    # The test file is the question, not the answer, so it survives whole.
    assert job.protected_lines > 0
    assert job.result_zip is not None and job.result_zip.exists()

    with zipfile.ZipFile(job.result_zip) as zf:
        names = zf.namelist()
        assert "tests/test_mylib.py" in names
        assert zf.read("tests/test_mylib.py").decode() == TEST

    worker_jobs.shutil.rmtree(job.work_dir, ignore_errors=True)


def test_a_job_whose_test_does_not_exist_fails_with_a_reason(tmp_path):
    """Not a traceback, and not a reduction of something that never ran."""
    registry = worker_jobs.JobRegistry()

    job = registry.submit(_archive(tmp_path),
                          ["pytest", "tests/test_mylib.py::test_imaginary"])
    _wait(job)

    assert job.status == "failed"
    assert job.error
    assert "Traceback" not in job.error

    worker_jobs.shutil.rmtree(job.work_dir, ignore_errors=True)
