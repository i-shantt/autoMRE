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
    # bytes, because that is what rollback has to write back
    assert originals["pkg/widget.py"] == FILE.encode("utf-8")


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
    from score_patches import p2p_plan

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('a', ['foo', 'foo,bar'])\n"
        "def test_p(a):\n    assert a\n\n"
        "def test_plain():\n    assert True\n")

    plan = p2p_plan(
        tmp_path,
        ["tests/test_x.py::test_plain",
         "tests/test_x.py::test_p[foo]",
         "tests/test_x.py::test_p[foo,"],
        _sys.executable, timeout=120)

    assert plan.dropped == ["tests/test_x.py::test_p[foo,"]
    assert plan.n_tests == 2
    assert "tests/test_x.py::test_plain" in plan.commands[0]
    assert "tests/test_x.py::test_p[foo]" in plan.commands[0]


def test_django_labels_route_through_the_project_runner(tmp_path):
    from score_patches import p2p_plan

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "runtests.py").write_text("")

    plan = p2p_plan(
        tmp_path,
        ["test_a (apps.tests.AppsTests.test_a)",
         "test_b (apps.tests.AppsTests.test_b)"],
        "python3", timeout=120)

    assert plan.commands == [["python3", "tests/runtests.py",
                              "apps.tests.AppsTests.test_a",
                              "apps.tests.AppsTests.test_b", "-v", "0"]]
    assert plan.n_tests == 2
    assert plan.dropped == []


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


def test_labels_and_node_ids_both_survive(tmp_path):
    """django-17084 lists 93 unittest labels beside 14 bare names.

    One command could hold only one shape, so a single bare name
    resolving through the index was enough to take the pytest branch and
    drop all 93 labels — with nothing recorded to say they had gone. It
    escaped notice only because none of that instance's bare names
    happened to resolve.
    """
    import sys as _sys
    from score_patches import p2p_plan

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "runtests.py").write_text("")
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_solo():\n    assert True\n")

    plan = p2p_plan(
        tmp_path,
        ["test_a (apps.tests.AppsTests.test_a)", "test_solo"],
        _sys.executable, timeout=120)

    assert len(plan.commands) == 2
    assert plan.commands[0][1] == "tests/runtests.py"
    assert "tests/test_x.py::test_solo" in plan.commands[1]
    assert plan.n_tests == 2


def test_a_name_that_resolves_to_nothing_is_recorded(tmp_path):
    """SWE-bench prints a unittest docstring in place of the method name.

    34 of django-17029's 43 PASS_TO_PASS entries are prose. They matched
    no test, were skipped by a bare `if len(found) == 1`, and never
    reached the dropped list — so the control reported a clean
    previously-passing set of 43 while running 9.
    """
    import sys as _sys
    from score_patches import p2p_plan

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_real():\n    assert True\n")

    plan = p2p_plan(
        tmp_path,
        ["test_real", "If the __path__ attr is empty, use __file__."],
        _sys.executable, timeout=120)

    assert plan.dropped == ["If the __path__ attr is empty, use __file__."]
    assert plan.n_tests == 1


def test_labels_without_a_django_runner_are_dropped_not_run(tmp_path):
    from score_patches import p2p_plan

    plan = p2p_plan(tmp_path, ["test_a (pkg.tests.T.test_a)"],
                    "python3", timeout=120)

    assert plan.commands == []
    assert plan.dropped == ["test_a (pkg.tests.T.test_a)"]


# ------------------------------------------------- editing the graded test

def test_an_edit_to_the_graded_test_file_is_refused(tmp_path):
    """The prompt says not to edit tests; nothing made that true.

    It also shows the failing test's source verbatim, so the anchor is
    in front of the model. Three of the eighty generations on disk aim
    an edit at the exact FAIL_TO_PASS file, and one deleted assertion
    there would have been scored as a fix.
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_writer.py").write_text("assert x == 1\n")

    edits, _ = parse_edits(
        _block("tests/test_writer.py", "assert x == 1", "assert True"))
    _, status = apply_edits(tmp_path, edits)

    assert status.startswith("edit targets a test file")
    assert (tmp_path / "tests" / "test_writer.py").read_text() == \
        "assert x == 1\n"


def test_a_refusal_rolls_back_the_edits_before_it(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "test_a.py").write_text("y = 1\n")

    edits, _ = parse_edits(_block("pkg/a.py", "x = 1", "x = 2")
                           + _block("pkg/test_a.py", "y = 1", "y = 2"))
    _, status = apply_edits(tmp_path, edits)

    assert status.startswith("edit targets a test file")
    assert (tmp_path / "pkg" / "a.py").read_text() == "x = 1\n"


def test_a_source_file_beside_the_tests_is_still_a_test_file(tmp_path):
    """django-17087's model reached for tests/postgres_tests/models.py.

    No test_patch names it and its filename is innocent, but the graded
    test imports it, so an edit there moves the number just the same.
    """
    from score_patches import is_test_path

    assert is_test_path("tests/postgres_tests/models.py")
    assert is_test_path("sympy/core/tests/test_numbers.py")
    assert is_test_path("pkg/conftest.py")
    assert not is_test_path("pylint/config/argument.py")
    assert not is_test_path("django/db/models/fields/__init__.py")


# --------------------------------------------------- bytes, not decoded text

def test_a_rolled_back_file_keeps_its_own_line_endings(tmp_path):
    """Restore used to write back what `read_text` had already changed.

    Universal newlines turn CRLF into LF on the way in, so the "original"
    written back on rollback was not the original. The tree is prepared
    once and reused for all fifteen samples of an instance, so the damage
    outlives the sample that caused it.
    """
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")

    edits, _ = parse_edits(_block("a.py", "nope", "still nope"))
    _, status = apply_edits(tmp_path, edits)

    assert status == "SEARCH block not found in file"
    assert (tmp_path / "a.py").read_bytes() == b"x = 1\r\ny = 2\r\n"


def test_a_file_that_is_not_utf8_is_refused_not_mangled(tmp_path):
    (tmp_path / "a.py").write_bytes(b"# \xff\xfe\nx = 1\n")

    edits, _ = parse_edits(_block("a.py", "x = 1", "x = 2"))
    _, status = apply_edits(tmp_path, edits)

    assert status.startswith("file is not utf-8")
    assert (tmp_path / "a.py").read_bytes() == b"# \xff\xfe\nx = 1\n"


# ---------------------------------------------------- the positive control

def test_the_gold_patch_survives_the_round_trip():
    """The control the README claimed, as something that runs.

    Every ground-truth patch is re-expressed in the model's own reply
    format and read back by the same parser a sample goes through. If
    this fails, a row of zeros cannot be blamed on the model.
    """
    import json as _json
    from score_patches import gold_as_reply, is_test_path

    path = _ROOT / "evaluation" / "stage3" / "instances.json"
    for record in _json.loads(path.read_text())["instances"]:
        reply, skipped = gold_as_reply(record["patch"])
        edits, err = parse_edits(reply)
        gold = {line[len("diff --git a/"):].split(" ")[0]
                for line in record["patch"].splitlines()
                if line.startswith("diff --git ")}

        assert err is None, record["instance_id"]
        assert skipped == [], record["instance_id"]
        assert {e.path for e in edits} == gold, record["instance_id"]
        # And the test-file guard must not refuse the right answer.
        assert not any(is_test_path(e.path) for e in edits), \
            record["instance_id"]


def test_a_block_with_no_filename_is_its_own_failure():
    edits, err = parse_edits(
        "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n")

    assert edits == []
    assert err == "SEARCH/REPLACE block with no file path"
