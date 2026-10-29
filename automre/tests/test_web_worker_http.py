"""The HTTP layer, through the app rather than around it.

Every other test here drives `JobRegistry` directly, which left the
routes themselves unexercised — status codes, the refusal paths, and the
one endpoint that takes a filesystem path from the client.

Skipped unless the web stack is installed. fastapi is deliberately not in
the [dev] extra: the worker deploys as its own container and pins its
stack in web/worker/requirements.txt, so a core dev install should not
drag a web framework in. Install it with
`pip install -r web/worker/requirements.txt`.
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="web worker stack not installed")
from fastapi.testclient import TestClient  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "web" / "worker"))

import main as worker_main  # noqa: E402

from test_web_worker_end_to_end import _archive  # noqa: E402

COMMAND = "pytest tests/test_mylib.py::test_add -x -q"


@pytest.fixture
def client():
    return TestClient(worker_main.app)


@pytest.fixture
def finished_job(client, tmp_path):
    """A real reduction, submitted through the API and waited out."""
    archive = _archive(tmp_path)
    with archive.open("rb") as fh:
        res = client.post("/api/jobs",
                          files={"file": ("upload.zip", fh, "application/zip")},
                          data={"command": COMMAND})
    assert res.status_code == 202, res.text
    job_id = res.json()["job_id"]

    deadline = time.time() + 300
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in {"done", "failed", "cancelled"}:
            break
        time.sleep(0.25)
    assert status["status"] == "done", status.get("error")
    return job_id


def test_health_is_answerable_without_a_job(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_an_empty_command_is_refused_before_the_upload_is_kept(client, tmp_path):
    archive = _archive(tmp_path)
    with archive.open("rb") as fh:
        res = client.post("/api/jobs",
                          files={"file": ("upload.zip", fh, "application/zip")},
                          data={"command": "   "})
    assert res.status_code == 400
    assert "reproduction command" in res.json()["detail"]


def test_a_non_zip_upload_is_refused_with_a_reason(client, tmp_path):
    loose = tmp_path / "mylib.py"
    loose.write_text("x = 1\n")
    with loose.open("rb") as fh:
        res = client.post("/api/detect",
                          files={"file": ("mylib.py", fh, "text/x-python")})
    assert res.status_code == 400
    assert ".zip" in res.json()["detail"]


def test_detect_suggests_the_tests_that_are_actually_there(client, tmp_path):
    archive = _archive(tmp_path)
    with archive.open("rb") as fh:
        res = client.post("/api/detect",
                          files={"file": ("upload.zip", fh, "application/zip")})
    assert res.status_code == 200
    body = res.json()
    assert body["files"] == 2
    assert body["lines"] > 0
    assert "pytest tests/test_mylib.py::test_add" in body["suggested_commands"]


def test_an_unknown_job_says_the_worker_may_have_restarted(client):
    res = client.get("/api/jobs/doesnotexist")
    assert res.status_code == 404
    assert "restarted" in res.json()["detail"]


# ------------------------------------------------------------- manifest

def test_the_manifest_is_served_once_the_job_is_done(client, finished_job):
    res = client.get(f"/api/jobs/{finished_job}/manifest")
    assert res.status_code == 200
    manifest = res.json()
    assert manifest["root_prefix"] == "myproject"

    by_path = {f["path"]: f for f in manifest["files"]}
    assert by_path["mylib.py"]["status"] == "reduced"
    assert by_path["tests/test_mylib.py"]["status"] == "protected"


def test_a_manifest_asked_for_too_early_is_409_not_404(client, tmp_path):
    """409 and 404 answer different questions and a UI reacts differently.

    "Not yet" means keep polling; "no such job" means stop.
    """
    archive = _archive(tmp_path)
    with archive.open("rb") as fh:
        res = client.post("/api/jobs",
                          files={"file": ("upload.zip", fh, "application/zip")},
                          data={"command": COMMAND})
    job_id = res.json()["job_id"]
    got = client.get(f"/api/jobs/{job_id}/manifest")
    # Racy by nature: the job may already have finished on a fast machine.
    assert got.status_code in {200, 409}
    if got.status_code == 409:
        assert "no manifest yet" in got.json()["detail"]


# ----------------------------------------------------------- file content

def test_both_sides_of_a_reduced_file_are_served(client, finished_job):
    """The before copy is the point: the reducer rewrites in place.

    Without `snapshot_originals` the left-hand side of this diff would be
    the reduced file, and the viewer would show nothing changing.
    """
    before = client.get(f"/api/jobs/{finished_job}/file",
                        params={"path": "mylib.py", "side": "before"}).json()
    after = client.get(f"/api/jobs/{finished_job}/file",
                       params={"path": "mylib.py", "side": "after"}).json()

    assert before["exists"] and after["exists"]
    assert "unused_one" in before["text"], "the before copy was overwritten"
    assert "unused_one" not in after["text"], "nothing was removed"
    assert "def add" in after["text"], "the needed code went too"


def test_a_protected_file_is_identical_on_both_sides(client, finished_job):
    before = client.get(f"/api/jobs/{finished_job}/file",
                        params={"path": "tests/test_mylib.py",
                                "side": "before"}).json()
    after = client.get(f"/api/jobs/{finished_job}/file",
                       params={"path": "tests/test_mylib.py",
                               "side": "after"}).json()
    assert before["text"] == after["text"]


def test_a_path_that_escapes_the_project_is_refused(client, finished_job):
    """The one endpoint that takes a filesystem path from the client."""
    for attempt in ("../../../../etc/passwd",
                    "../venv/pyvenv.cfg",
                    "/etc/passwd"):
        res = client.get(f"/api/jobs/{finished_job}/file",
                         params={"path": attempt, "side": "after"})
        assert res.status_code == 400, f"{attempt} was not refused"
        assert "escapes" in res.json()["detail"]


def test_an_unknown_side_is_refused(client, finished_job):
    res = client.get(f"/api/jobs/{finished_job}/file",
                     params={"path": "mylib.py", "side": "sideways"})
    assert res.status_code == 400


def test_a_missing_file_is_a_result_not_an_error(client, finished_job):
    """A deleted file has no "after", and that is the answer.

    Reporting it as 404 would make the viewer show an error box where it
    should show an empty pane with "this file was deleted".
    """
    res = client.get(f"/api/jobs/{finished_job}/file",
                     params={"path": "never_existed.py", "side": "after"})
    assert res.status_code == 200
    assert res.json()["exists"] is False
    assert res.json()["text"] is None


def test_an_expired_job_says_expired_rather_than_missing(client, finished_job):
    """The distinction the retention policy exists to preserve."""
    job = worker_main.registry.get(finished_job)
    import jobs as worker_jobs
    worker_jobs.shutil.rmtree(job.work_dir, ignore_errors=True)
    job.work_dir = None
    job.result_zip = None
    job.expired = True

    for url in (f"/api/jobs/{finished_job}/file?path=mylib.py",
                f"/api/jobs/{finished_job}/result"):
        res = client.get(url)
        assert res.status_code == 410, url
        assert "expired" in res.json()["detail"]


def test_the_result_zip_downloads(client, finished_job):
    res = client.get(f"/api/jobs/{finished_job}/result")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    assert finished_job in res.headers.get("content-disposition", "")

    import io
    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
        assert "tests/test_mylib.py" in zf.namelist()
