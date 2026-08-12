"""Reading an instance list, and turning one into a task.

The field mapping is the part that quietly goes wrong: SWE-bench exports
FAIL_TO_PASS as a JSON-encoded string in some dumps and a real list in
others, and a mapping that silently produces an empty test list would
hand the runner a command that collects nothing — which is the exact
failure the readiness gate exists to catch, arriving one step earlier
than the gate can see it.

The end-to-end path is exercised against a git repository created here,
so these tests need no network and no GitHub.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "evaluation"))

import ingest_tasks as it  # noqa: E402

LIB = "def add(a, b):\n    return a + b\n"
TEST = "import mylib\n\n\ndef test_add():\n    assert mylib.add(1, 2) == 3\n"
BROKEN = "def add(a, b):\n    return a - b\n"


def _git_repo(tmp_path: Path) -> tuple:
    """A real repository with a commit where the test fails.

    Mirrors a SWE-bench instance: the commit is the one *before* the fix,
    so the named test does not pass there.
    """
    repo = tmp_path / "origin"
    (repo / "tests").mkdir(parents=True)
    (repo / "mylib.py").write_text(BROKEN)
    (repo / "tests" / "test_mylib.py").write_text(TEST)

    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True,
                   capture_output=True)
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    run("add", "-A")
    run("commit", "--quiet", "-m", "buggy")
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         check=True).stdout.strip()
    return repo, sha


def _diff_adding(repo: Path, relative: str, addition: str) -> str:
    """A real unified diff that appends `addition` to a file.

    Produced with git rather than written out here: a hand-counted @@
    header is one edit away from "corrupt patch", and the point of these
    tests is the applying, not the arithmetic.
    """
    target = repo / relative
    original = target.read_text()
    target.write_text(original + addition)
    patch = subprocess.run(["git", "-C", str(repo), "diff"],
                           capture_output=True, text=True,
                           check=True).stdout
    target.write_text(original)
    return patch


# ------------------------------------------------------------- mapping

def test_fail_to_pass_is_read_whether_list_or_json_string():
    as_list = {"repo": "psf/requests", "base_commit": "abc123",
               "FAIL_TO_PASS": ["tests/test_a.py::test_x"]}
    as_string = {"repo": "psf/requests", "base_commit": "abc123",
                 "FAIL_TO_PASS": '["tests/test_a.py::test_x"]'}

    assert it.to_task(as_list).test_command == it.to_task(as_string).test_command
    assert "tests/test_a.py::test_x" in it.to_task(as_string).test_command


def test_a_bare_owner_name_becomes_a_clone_url():
    task = it.to_task({"repo": "pallets/flask", "base_commit": "deadbeef",
                       "FAIL_TO_PASS": ["tests/test_x.py::test_y"]})

    assert task.repo == "https://github.com/pallets/flask"
    assert task.commit == "deadbeef"
    assert task.task_id == "flask-deadbeef"


def test_instance_id_is_kept_when_the_source_has_one():
    task = it.to_task({"instance_id": "psf__requests-1234",
                       "repo": "psf/requests", "base_commit": "abc",
                       "FAIL_TO_PASS": ["t.py::t"]})

    assert task.task_id == "psf__requests-1234"


def test_only_the_first_named_test_is_used():
    """Several FAIL_TO_PASS tests usually cover one bug; one is a task."""
    task = it.to_task({"repo": "a/b", "base_commit": "c",
                       "FAIL_TO_PASS": ["t.py::one", "t.py::two"]})

    assert task.test_command.count("t.py::one") == 1
    assert "t.py::two" not in task.test_command
    assert "2 test(s) named" in task.notes


@pytest.mark.parametrize("record", [
    {"base_commit": "c", "FAIL_TO_PASS": ["t::t"]},          # no repo
    {"repo": "a/b", "FAIL_TO_PASS": ["t::t"]},               # no commit
    {"repo": "a/b", "base_commit": "c"},                     # no test
    {"repo": "a/b", "base_commit": "c", "FAIL_TO_PASS": []},  # empty test
])
def test_an_unusable_record_is_refused_not_guessed_at(record):
    with pytest.raises(ValueError):
        it.to_task(record)


def test_load_instances_reads_json_and_jsonl(tmp_path):
    rows = [{"repo": "a/b", "base_commit": "1", "FAIL_TO_PASS": ["t::t"]},
            {"repo": "c/d", "base_commit": "2", "FAIL_TO_PASS": ["u::u"]}]

    as_json = tmp_path / "x.json"
    as_json.write_text(json.dumps(rows))
    as_jsonl = tmp_path / "x.jsonl"
    as_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert it.load_instances(as_json) == rows
    assert it.load_instances(as_jsonl) == rows


# ----------------------------------------------------------- end to end

def test_a_failing_test_at_the_base_commit_is_accepted(tmp_path):
    """The SWE-bench shape: the reproduction command is the failure.

    A gate that demanded the command pass would reject every instance in
    the dataset, which is why require_pass is off by default.
    """
    repo, sha = _git_repo(tmp_path)

    task = it.to_task({"repo": str(repo), "base_commit": sha,
                       "instance_id": "demo-1",
                       "FAIL_TO_PASS": ["tests/test_mylib.py::test_add"]})
    outcome = it.ingest_one(task, tmp_path / "ws",
                            cache_dir=tmp_path / "cache")

    assert outcome.accepted, outcome.reason
    assert outcome.lines > 0
    assert outcome.gate_runs >= 2


def test_an_instance_naming_a_missing_test_is_rejected_with_a_reason(tmp_path):
    repo, sha = _git_repo(tmp_path)

    task = it.to_task({"repo": str(repo), "base_commit": sha,
                       "instance_id": "demo-2",
                       "FAIL_TO_PASS": ["tests/test_mylib.py::test_ghost"]})
    outcome = it.ingest_one(task, tmp_path / "ws",
                            cache_dir=tmp_path / "cache")

    assert not outcome.accepted
    assert "does not contain" in outcome.reason


# --------------------------------------------------- building the command

def test_a_node_id_is_run_as_a_node_id(tmp_path):
    assert it._command_for(tmp_path, "tests/t.py::test_x")[1:] == [
        "-m", "pytest", "tests/t.py::test_x", "-x", "-q"]


def test_a_bare_function_name_is_selected_with_k(tmp_path):
    """Sympy names the function and nothing else — 19.6% of Verified."""
    assert it._command_for(tmp_path, "test_issue_24288")[1:] == [
        "-m", "pytest", "-k", "test_issue_24288", "-x", "-q"]


def test_a_repo_with_its_own_runner_is_driven_through_it(tmp_path):
    """Django's suite cannot run under a bare pytest.

    41.6% of SWE-bench Verified is written as a unittest label, and
    almost all of it is Django, whose settings, test database and app
    registry are set up by tests/runtests.py.
    """
    label = "test_x (migrations.test_writer.WriterTests.test_x)"
    assert it._command_for(tmp_path, label)[1:] == [
        "-m", "unittest", "migrations.test_writer.WriterTests.test_x"]

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "runtests.py").write_text("")
    assert it._command_for(tmp_path, label)[1:] == [
        "tests/runtests.py", "migrations.test_writer.WriterTests.test_x",
        "-v", "0"]


# ------------------------------------------------------------ test_patch

def test_a_test_that_only_the_patch_introduces_is_accepted(tmp_path):
    """The SWE-bench default: the commit predates the test.

    62.6% of SWE-bench Verified names a test the checkout does not
    contain until its test_patch is applied. Without this the gate
    rejects the instance for naming a test that is not there — which is
    true, and the wrong conclusion.
    """
    repo, sha = _git_repo(tmp_path)
    patch = _diff_adding(repo, "tests/test_mylib.py",
                         "\n\ndef test_added_by_patch():\n"
                         "    assert mylib.add(2, 2) == 4\n")

    task = it.to_task({"repo": str(repo), "base_commit": sha,
                       "instance_id": "demo-3", "test_patch": patch,
                       "FAIL_TO_PASS":
                           ["tests/test_mylib.py::test_added_by_patch"]})
    outcome = it.ingest_one(task, tmp_path / "ws",
                            cache_dir=tmp_path / "cache")

    assert outcome.accepted, outcome.reason


def test_the_patch_is_applied_again_on_a_second_run(tmp_path):
    """A cached checkout keeps the previous run's edits to tracked files.

    `git clean` removes untracked files only, so without a hard reset the
    second run finds the patch already applied and fails to apply it.
    """
    repo, sha = _git_repo(tmp_path)
    patch = _diff_adding(repo, "mylib.py", "# patched\n")
    record = {"repo": str(repo), "base_commit": sha, "instance_id": "demo-4",
              "test_patch": patch,
              "FAIL_TO_PASS": ["tests/test_mylib.py::test_add"]}
    cache = tmp_path / "cache"

    first = it.ingest_one(it.to_task(record), tmp_path / "ws1",
                          cache_dir=cache)
    second = it.ingest_one(it.to_task(record), tmp_path / "ws2",
                           cache_dir=cache)

    assert "did not apply" not in (second.reason or "")
    assert first.accepted == second.accepted


def test_a_patch_that_does_not_apply_is_refused_not_ignored(tmp_path):
    """Silently skipping it would blame the gate for the wrong thing."""
    repo, sha = _git_repo(tmp_path)
    task = it.to_task({
        "repo": str(repo), "base_commit": sha, "instance_id": "demo-5",
        "test_patch": "--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-x\n+y\n",
        "FAIL_TO_PASS": ["tests/test_mylib.py::test_add"]})
    outcome = it.ingest_one(task, tmp_path / "ws", cache_dir=tmp_path / "cache")

    assert not outcome.accepted
    assert "test_patch did not apply" in outcome.reason
