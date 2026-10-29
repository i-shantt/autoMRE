"""The per-file verdicts, and the one that cannot be read off a counter.

A manifest row says what happened to a file. Most of them follow from
what the reducer reports, but "did this file change?" does not — and
getting it from the line counts produced a UI that said a file was
untouched while showing a diff of it being gutted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from webcore import JobState, build_manifest  # noqa: E402

ORIGINAL = '''\
"""A module with one needed function and one that is not."""


def needed(a, b):
    return a + b


def not_needed(values):
    total = 0
    for v in values:
        total += v
    return total
'''

# What the reducer leaves behind: the docstring is gone, the body of
# not_needed is gone, and a `pass` plus the blank lines around the cuts
# bring the file back to exactly the length it started at. Contrived to
# land on equality, but the shape is real — it was found on a 37-line
# module that came out at 37.
STUBBED = '''\



def needed(a, b):
    return a + b


def not_needed(values):



    pass
'''


def _summary(root: Path, per_file):
    return SimpleNamespace(
        protected_files=[],
        unreachable_deleted=[],
        imported_deleted=[],
        inlined_away=[],
        per_file_reduction=per_file,
        undecodable_files=[],
        outside_project_files=[],
        reduction_errors={},
    )


def _state():
    return JobState(job_id="j", command=["pytest", "t.py::k"], status="done")


def test_a_stubbed_file_is_reported_as_reduced_not_unchanged(tmp_path):
    """The bug this exists to prevent, at the size it was found.

    Removing a function body deletes four lines and adds a `pass` plus
    the blank lines around it, so the count can land exactly where it
    started. A 37-line module reported 37 -> 37 and the interface
    labelled it "Kept whole" directly beside a diff showing its docstring
    and bodies being removed.
    """
    root = tmp_path / "proj"
    before = tmp_path / "before"
    (root).mkdir()
    (before).mkdir()
    (root / "mylib.py").write_text(STUBBED)
    (before / "mylib.py").write_text(ORIGINAL)

    lines = len(ORIGINAL.splitlines())
    assert len(STUBBED.splitlines()) == lines, "fixture must be same-length"

    manifest = build_manifest(
        "j", _state(), root, ["pytest", "t.py::k"],
        _summary(root, {root / "mylib.py": (lines, lines)}),
        originals_dir=before)

    row = next(f for f in manifest["files"] if f["path"] == "mylib.py")
    assert row["status"] == "reduced", row
    # Flagged so the viewer can explain why "shrunk" shows equal counts
    # instead of looking like a contradiction.
    assert row["same_line_count"] is True


def test_a_genuinely_untouched_file_is_still_unchanged(tmp_path):
    """The other direction, or the fix would just relabel everything."""
    root = tmp_path / "proj"
    before = tmp_path / "before"
    root.mkdir()
    before.mkdir()
    (root / "mylib.py").write_text(ORIGINAL)
    (before / "mylib.py").write_text(ORIGINAL)

    lines = len(ORIGINAL.splitlines())
    manifest = build_manifest(
        "j", _state(), root, ["pytest", "t.py::k"],
        _summary(root, {root / "mylib.py": (lines, lines)}),
        originals_dir=before)

    row = next(f for f in manifest["files"] if f["path"] == "mylib.py")
    assert row["status"] == "unchanged", row
    assert "same_line_count" not in row


def test_without_the_originals_it_falls_back_to_the_counts(tmp_path):
    """No before copy is a degraded answer, not a crash.

    The snapshot is taken after the gate, so a job refused before then
    has nothing to compare against and the counts are all there is.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mylib.py").write_text(STUBBED)
    lines = len(ORIGINAL.splitlines())

    manifest = build_manifest(
        "j", _state(), root, ["pytest", "t.py::k"],
        _summary(root, {root / "mylib.py": (lines, lines)}))

    row = next(f for f in manifest["files"] if f["path"] == "mylib.py")
    assert row["status"] == "unchanged"


def test_a_shorter_file_needs_no_comparison(tmp_path):
    """The common case stays cheap: fewer lines is already an answer."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mylib.py").write_text("x = 1\n")

    manifest = build_manifest(
        "j", _state(), root, ["pytest", "t.py::k"],
        _summary(root, {root / "mylib.py": (120, 8)}))

    row = next(f for f in manifest["files"] if f["path"] == "mylib.py")
    assert row["status"] == "reduced"
    assert "same_line_count" not in row


def test_a_manifest_is_still_written_when_there_is_no_summary(tmp_path):
    """A job cancelled during provisioning never produced one.

    Writing nothing would read to the caller as a lost job rather than
    as one that stopped early.
    """
    root = tmp_path / "proj"
    root.mkdir()
    manifest = build_manifest("j", _state(), root, ["pytest", "t.py::k"], None)
    assert manifest["files"] == []
    assert manifest["root_prefix"] == "proj"


def test_the_deleted_list_is_what_gets_truncated(tmp_path):
    """A cap must not cost the survivors, which are the result.

    On a real reduction most files are deleted and that list says little;
    the handful left is what someone came to look at.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "kept.py").write_text("x = 1\n")
    per_file = {root / "kept.py": (10, 2)}
    summary = _summary(root, per_file)
    summary.unreachable_deleted = [root / f"gone{i}.py" for i in range(50)]

    manifest = build_manifest("j", _state(), root, ["pytest", "t.py::k"],
                              summary, max_files=10)

    assert manifest["truncated"] is True
    assert len(manifest["files"]) == 10
    paths = {f["path"] for f in manifest["files"]}
    assert "kept.py" in paths, "the survivor was dropped for deleted files"
