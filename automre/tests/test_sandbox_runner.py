"""The in-sandbox runner must agree with the worker, byte for byte.

There are two front doors onto one reducer: the FastAPI worker, which
runs a job on a thread and keeps it in memory, and `web/runner/run_job.py`,
which runs inside a sandbox and reports through files. They share
`webcore`, and this is the test that keeps that true — it drives both
against the same fixture as test_web_worker_end_to_end.py and requires
the zip they produce to be identical.

Without it the two implementations agree on the day they were written and
drift afterwards, and the drift is invisible: each one passes its own
tests while the hosted service quietly reduces differently from the
self-hosted one.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "web" / "worker"))

import jobs as worker_jobs  # noqa: E402

from test_web_worker_end_to_end import TEST, _archive, _wait  # noqa: E402

RUNNER = _ROOT / "web" / "runner" / "run_job.py"
COMMAND = "pytest tests/test_mylib.py::test_add -x -q"


def _run_runner(tmp_path: Path, archive: Path, *, sink_name: str = "sink",
                extra: list | None = None) -> Path:
    sink = tmp_path / sink_name
    proc = subprocess.run(
        [sys.executable, str(RUNNER),
         "--job-id", "testjob",
         "--archive", str(archive),
         "--command", COMMAND,
         "--sink", str(sink),
         *(extra or [])],
        capture_output=True, text=True, timeout=600)
    assert sink.exists(), (
        f"runner wrote no sink (rc={proc.returncode})\n{proc.stderr[-2000:]}")
    return sink


def _final(sink: Path) -> dict:
    path = sink / "status-final.json"
    assert path.exists(), "no final status; the flush did not happen"
    return json.loads(path.read_text())


def test_the_runner_reduces_and_reports_done(tmp_path):
    sink = _run_runner(tmp_path, _archive(tmp_path))

    final = _final(sink)
    assert final["status"] == "done", final.get("error")
    assert final["original_lines"] > final["final_lines"], "nothing removed"
    assert final["protected_lines"] > 0
    assert (sink / "reduced.zip").exists()

    with zipfile.ZipFile(sink / "reduced.zip") as zf:
        assert "tests/test_mylib.py" in zf.namelist()
        # The test file is the question, not the answer.
        assert zf.read("tests/test_mylib.py").decode() == TEST


def test_the_runner_and_the_worker_produce_the_same_zip(tmp_path):
    """The claim that `webcore` is really one engine and not two.

    Same fixture, same command, two drivers. Comparing the member list
    and every member's bytes, because a difference in either is a
    difference in what the user downloads.
    """
    sink = _run_runner(tmp_path, _archive(tmp_path / "a"),
                       sink_name="from_runner")

    registry = worker_jobs.JobRegistry()
    job = registry.submit(_archive(tmp_path / "b"),
                          ["pytest", "tests/test_mylib.py::test_add",
                           "-x", "-q"])
    _wait(job)
    assert job.status == "done", job.error

    with zipfile.ZipFile(sink / "reduced.zip") as a, \
            zipfile.ZipFile(job.result_zip) as b:
        assert sorted(a.namelist()) == sorted(b.namelist())
        for name in sorted(a.namelist()):
            assert a.read(name) == b.read(name), f"{name} differs"

    # And the account of what happened, not just the artifact. Comparing
    # the per-file verdicts rather than the whole document, because the
    # stats block legitimately differs: two runs have two elapsed times.
    from_runner = json.loads((sink / "manifest.json").read_text())

    def verdicts(manifest):
        return {f["path"]: (f["status"], f["before_lines"], f["after_lines"])
                for f in manifest["files"]}

    assert verdicts(from_runner) == verdicts(job.manifest)
    assert from_runner["root_prefix"] == job.manifest["root_prefix"]

    worker_jobs.shutil.rmtree(job.work_dir, ignore_errors=True)


def test_the_manifest_says_what_happened_to_each_file(tmp_path):
    """The per-file account the service used to discard.

    The reducer reports `per_file_reduction` and `protected_files`; the
    web layer kept two numbers out of a dozen, so a user downloaded a zip
    and had to diff it against their own project to learn anything.
    """
    sink = _run_runner(tmp_path, _archive(tmp_path))

    manifest = json.loads((sink / "manifest.json").read_text())
    assert manifest["command"] == ["pytest", "tests/test_mylib.py::test_add",
                                  "-x", "-q"]
    # Without this a viewer cannot line the result up against the upload:
    # project_root() strips one wrapping directory.
    assert manifest["root_prefix"] == "myproject"

    by_path = {f["path"]: f for f in manifest["files"]}

    lib = by_path.get("mylib.py")
    assert lib is not None, sorted(by_path)
    assert lib["status"] == "reduced", lib
    assert lib["after_lines"] < lib["before_lines"]

    test_file = by_path.get("tests/test_mylib.py")
    assert test_file is not None, sorted(by_path)
    assert test_file["status"] == "protected", test_file
    assert test_file["before_lines"] == test_file["after_lines"]


def test_a_failing_job_still_writes_a_final_status(tmp_path):
    """A job that cannot start has to say so in the place readers look.

    The sandbox is gone moments after the process exits, so a failure
    that only went to stderr would reach the user as "lost" — which reads
    as our bug rather than as a reproduction command that names a test
    the project does not contain.
    """
    archive = _archive(tmp_path)
    sink = tmp_path / "sink"
    subprocess.run(
        [sys.executable, str(RUNNER),
         "--job-id", "nosuchtest",
         "--archive", str(archive),
         "--command", "pytest tests/test_mylib.py::test_imaginary -x -q",
         "--sink", str(sink)],
        capture_output=True, text=True, timeout=600)

    final = _final(sink)
    assert final["status"] == "failed"
    assert final["error"]
    assert "Traceback" not in final["error"]


def test_a_cancelled_job_keeps_what_it_reduced(tmp_path):
    """Cancel is not an error, and the partial tree is a real answer.

    The reducer only ever commits validated candidates, so the tree at a
    query boundary still reproduces the failure — just less reduced. The
    sentinel exists before the run starts here, which also pins that the
    check happens before provisioning rather than only at the first
    progress event.
    """
    cancel = tmp_path / "cancel"
    cancel.write_text("stop")

    sink = _run_runner(tmp_path, _archive(tmp_path),
                       extra=["--cancel-file", str(cancel)])

    final = _final(sink)
    assert final["status"] == "cancelled", final
    assert (sink / "reduced.zip").exists(), "cancel discarded the tree"


def test_running_out_of_budget_is_not_reported_as_a_cancellation(tmp_path):
    """Two different things that used to share one word.

    The worker raises the same KeyboardInterrupt for a user's cancel and
    for its own deadline, then reports both as "cancelled" — so a job we
    ran out of room for looks like the user's decision. A zero budget is
    the cheapest way to reach the deadline path.
    """
    sink = _run_runner(tmp_path, _archive(tmp_path),
                       extra=["--budget", "0"])

    final = _final(sink)
    assert final["status"] == "timed_out", final
    assert "time limit" in final["message"]
