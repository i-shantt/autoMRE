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
