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
