"""Reading a model's reply and putting its edits into a file.

This is the part of Stage 3 with the most room to be quietly wrong. A
parser that drops an edit reports a model that produced nothing; one
that applies an edit to the wrong place reports a model that broke the
repository. Both look like results.

The reduced arm has a failure mode of its own, covered at the bottom:
its SEARCH block is copied from a tree the reducer has cut lines out of,
so an anchor spanning a deletion cannot match the original file. That
has to be *refused and counted*, never approximated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "evaluation" / "stage3"))

from score_patches import apply_edits, parse_edits  # noqa: E402

FILE = (
    "class Widget:\n"
    "    def size(self):\n"
    "        return self.w * self.h\n"
    "\n"
    "    def name(self):\n"
    "        return self._name\n"
)


def _block(path, search, replace, heading="### "):
    return (f"{heading}{path}\n"
            f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n")


def test_parses_the_format_the_prompt_asks_for():
    edits, err = parse_edits(
        "Here is the fix.\n\n"
        + _block("pkg/widget.py", "    def size(self):",
                 "    def size(self) -> int:"))
    assert err is None
    assert len(edits) == 1
    assert edits[0].path == "pkg/widget.py"
    assert edits[0].replace == "    def size(self) -> int:"


def test_a_second_edit_inherits_the_last_named_file():
    """Models write one heading and then several blocks under it."""
    text = (_block("pkg/widget.py", "a = 1", "a = 2")
            + "<<<<<<< SEARCH\nb = 1\n=======\nb = 2\n>>>>>>> REPLACE\n")
    edits, err = parse_edits(text)
    assert err is None
    assert [e.path for e in edits] == ["pkg/widget.py"] * 2


def test_path_without_the_heading_marker():
    edits, _ = parse_edits(_block("pkg/widget.py", "a = 1", "a = 2",
                                  heading=""))
    assert edits and edits[0].path == "pkg/widget.py"


def test_prose_with_no_blocks_is_a_parse_failure_not_an_empty_patch():
    edits, err = parse_edits(
        "The bug is in Widget.size, which multiplies the wrong fields.")
    assert edits == []
    assert err


def test_applies_and_leaves_the_rest_of_the_file_alone(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "widget.py").write_text(FILE)
    edits, _ = parse_edits(_block(
        "pkg/widget.py",
        "        return self.w * self.h",
        "        return self.w * self.h * self.d"))

    originals, status = apply_edits(tmp_path, edits)

    assert status == "applied"
    text = (tmp_path / "pkg" / "widget.py").read_text()
    assert "self.w * self.h * self.d" in text
    assert "    def name(self):" in text
    assert originals["pkg/widget.py"] == FILE


def test_an_ambiguous_anchor_is_refused(tmp_path):
    """Two matches means the edit lands somewhere by coin flip."""
    (tmp_path / "w.py").write_text("x = 1\ny = 2\nx = 1\n")
    edits, _ = parse_edits(_block("w.py", "x = 1", "x = 3"))

    _, status = apply_edits(tmp_path, edits)

    assert status == "SEARCH block is ambiguous"
    assert (tmp_path / "w.py").read_text() == "x = 1\ny = 2\nx = 1\n"


def test_a_failed_second_edit_rolls_the_first_one_back(tmp_path):
    """A half-applied patch would be scored as the model's whole answer."""
    (tmp_path / "w.py").write_text(FILE)
    text = (_block("w.py", "    def size(self):", "    def area(self):")
            + _block("w.py", "def never_present(self):", "nope"))

    edits, _ = parse_edits(text)
    _, status = apply_edits(tmp_path, edits)

    assert status == "SEARCH block not found in file"
    assert (tmp_path / "w.py").read_text() == FILE


def test_an_anchor_spanning_a_reduced_away_line_does_not_apply(tmp_path):
    """The reduced arm's own failure mode, made explicit.

    In the reduced tree `size` and `name` are adjacent, because the
    reducer deleted what sat between them. A SEARCH block copied from
    there spans a gap that the original file fills, so it must be
    reported as not-applying rather than fuzzily matched — an arm whose
    edits land approximately is not an arm anyone can score.
    """
    (tmp_path / "w.py").write_text(FILE)
    from_reduced = "    def size(self):\n    def name(self):"
    edits, _ = parse_edits(_block("w.py", from_reduced, "    def sz(self):"))

    _, status = apply_edits(tmp_path, edits)

    assert status == "SEARCH block not found in file"


# ---------------------------------------------- the previously-passing set

def test_ids_pytest_cannot_collect_are_dropped_not_fatal(tmp_path):
    """SWE-bench's PASS_TO_PASS is not all runnable as written.

    pylint's list contains a parametrized id truncated at the comma
    inside its own parameter:

        tests/config/test_config.py::test_p[foo,

    Handed to pytest it fails the whole batch, which made the
    *ground-truth* patch look like it broke previously-passing tests —
    and a rig that reports that scores every arm as a regression.
    """
    import sys as _sys
    from score_patches import p2p_command

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('a', ['foo', 'foo,bar'])\n"
        "def test_p(a):\n    assert a\n\n"
        "def test_plain():\n    assert True\n")

    command, dropped = p2p_command(
        tmp_path,
        ["tests/test_x.py::test_plain",
         "tests/test_x.py::test_p[foo]",
         "tests/test_x.py::test_p[foo,"],
        _sys.executable, timeout=120)

    assert dropped == ["tests/test_x.py::test_p[foo,"]
    assert "tests/test_x.py::test_plain" in command
    assert "tests/test_x.py::test_p[foo]" in command


def test_django_labels_route_through_the_project_runner(tmp_path):
    from score_patches import p2p_command

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "runtests.py").write_text("")

    command, dropped = p2p_command(
        tmp_path,
        ["test_a (apps.tests.AppsTests.test_a)",
         "test_b (apps.tests.AppsTests.test_b)"],
        "python3", timeout=120)

    assert command == ["python3", "tests/runtests.py",
                       "apps.tests.AppsTests.test_a",
                       "apps.tests.AppsTests.test_b", "-v", "0"]
    assert dropped == []


def test_a_failure_in_a_later_file_rolls_back_the_earlier_one(tmp_path):
    """Rollback has to span files, not just edits within one file.

    pylint's ground-truth patch touches three files, so a model
    answering that instance emits edits across several. If the first
    file were left modified when the third fails to apply, the tree
    handed to the test runner would be half the model's answer and
    would be scored as all of it.
    """
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 1\n")
    text = (_block("a.py", "x = 1", "x = 2")
            + _block("b.py", "not present anywhere", "y = 2"))

    edits, _ = parse_edits(text)
    _, status = apply_edits(tmp_path, edits)

    assert status == "SEARCH block not found in file"
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
    assert (tmp_path / "b.py").read_text() == "y = 1\n"
