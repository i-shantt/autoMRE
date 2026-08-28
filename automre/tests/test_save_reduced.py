"""Keeping the reduced tree instead of deleting it with the temp dir.

The runner reduced a repository, measured it, and then let the
TemporaryDirectory take the result away — so every published number
described a tree nobody could look at afterwards. `--save-reduced`
copies it out; these tests cover the two ways that copy can be useless.

Sharp edge: the destination is copied *over* on a re-run. A reduction
that got smaller must not leave the previous, larger run's files behind
to be read as part of the new one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "evaluation"))

from gistify_runner import GistifyTask, _save_reduced  # noqa: E402


def _task() -> GistifyTask:
    return GistifyTask(
        task_id="acme__widget-1",
        repo="https://github.com/acme/widget",
        commit="0123456789abcdef",
        test_command=["python3", "-m", "pytest", "tests/test_w.py::test_x"],
        test_id="tests/test_w.py::test_x",
        test_patch="diff --git a/tests/test_w.py b/tests/test_w.py\n",
    )


def _tree(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "core.py").write_text("def f():\n    return 1\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_w.py").write_text("def test_x():\n    pass\n")
    # The three kinds of debris a reduction leaves behind.
    (root / "pkg" / "__pycache__").mkdir()
    (root / "pkg" / "__pycache__" / "core.cpython-313.pyc").write_bytes(b"\x00")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / "widget.egg-info").mkdir()
    (root / "widget.egg-info" / "PKG-INFO").write_text("Name: widget\n")
    return root


def test_saves_the_source_and_not_the_debris(tmp_path):
    work = _tree(tmp_path / "work")
    dest_root = tmp_path / "out"

    _save_reduced(work, dest_root, _task(), orig_lines=112008, final_lines=4)

    tree = dest_root / "acme__widget-1" / "tree"
    assert (tree / "pkg" / "core.py").read_text() == "def f():\n    return 1\n"
    assert (tree / "tests" / "test_w.py").exists()
    # A consumer walking this tree must not find compiled bytecode, the
    # repository's history, or install metadata and count them as source.
    assert not (tree / "pkg" / "__pycache__").exists()
    assert not (tree / ".git").exists()
    assert not (tree / "widget.egg-info").exists()


def test_metadata_can_rebuild_the_input_tree(tmp_path):
    """The original tree is not copied, so its coordinates must be here.

    Without commit and test_patch, the full-repository arm of any later
    comparison cannot be reconstructed, and the saved tree is a reduction
    of nothing in particular.
    """
    work = _tree(tmp_path / "work")
    dest_root = tmp_path / "out"

    _save_reduced(work, dest_root, _task(), orig_lines=112008, final_lines=4)

    meta = json.loads(
        (dest_root / "acme__widget-1" / "meta.json").read_text())
    assert meta["commit"] == "0123456789abcdef"
    assert meta["repo"] == "https://github.com/acme/widget"
    assert meta["test_command"][-1] == "tests/test_w.py::test_x"
    assert meta["original_lines"] == 112008
    assert meta["final_lines"] == 4


def test_a_rerun_replaces_rather_than_merges(tmp_path):
    work = _tree(tmp_path / "work")
    dest_root = tmp_path / "out"
    _save_reduced(work, dest_root, _task(), orig_lines=10, final_lines=4)

    # Second run reduces further: core.py is gone entirely.
    (work / "pkg" / "core.py").unlink()
    _save_reduced(work, dest_root, _task(), orig_lines=10, final_lines=2)

    tree = dest_root / "acme__widget-1" / "tree"
    assert not (tree / "pkg" / "core.py").exists()


# ------------------------------------------------------- surviving a kill

def test_a_result_survives_the_json_round_trip():
    """--resume rebuilds finished tasks out of the results file.

    Ten tasks are 3.3 hours and the file used to be written once, at the
    end. A kill at task nine threw away all nine, and the workaround was
    a shell script outside the repository. Resuming is only worth
    anything if a row read back is the row that was written, so that is
    what this pins.
    """
    from dataclasses import asdict
    from gistify_runner import GistifyResult

    original = GistifyResult(
        task_id="requests-super_len_partial",
        execution_fidelity=1,
        original_files=41, final_files=1,
        original_lines=8992, final_lines=1166,
        single_file_output=True,
        total_queries=1184, timed_out_queries=3,
        time_seconds=612.5, undecodable_files=2)

    assert GistifyResult(**asdict(original)) == original


def test_a_partial_run_says_so_in_its_config(tmp_path):
    """A resumed-but-unfinished file must not read as a full run.

    This benchmark has voided its own numbers seven times, and a results
    file holding nine of ten tasks looks exactly like one holding ten
    unless something in it says otherwise.
    """
    import json
    from dataclasses import asdict
    from gistify_runner import GistifyResult, summarize

    done = [GistifyResult(task_id="a", execution_fidelity=1,
                          original_files=1, final_files=1,
                          original_lines=10, final_lines=5,
                          single_file_output=True, total_queries=7,
                          time_seconds=1.0)]
    payload = {
        "config": {"task_ids": ["a", "b"],
                   "complete": len(done) == 2},
        "summary": summarize(done),
        "runs": [asdict(r) for r in done],
    }
    out = tmp_path / "results.json"
    out.write_text(json.dumps(payload))

    read_back = json.loads(out.read_text())
    assert read_back["config"]["complete"] is False
    assert {r["task_id"] for r in read_back["runs"]} == {"a"}
