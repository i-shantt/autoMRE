"""Deciding what each arm gets to see.

The comparison in Stage 3 is only as honest as this file. Both arms are
supposed to differ in one thing — which tree the budget is filled from —
so anything that could quietly give one of them more room, or a
different ranking, or a recall figure computed against the wrong set,
turns the experiment into a number rather than an answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "evaluation" / "stage3"))

from build_contexts import (BM25, code_tokens, collect_python,  # noqa: E402
                            extract_test_source, failing_test_name,
                            gold_files, pack, render, token_costs)


def _words(text):
    """Token cost stand-in: whitespace words, so tests need no model."""
    return len(text.split())


def _batch(texts):
    return [_words(t) for t in texts]


# ------------------------------------------------------------- ranking

def test_identifier_parts_are_searchable():
    """The issue says "clear cache"; the code says clear_cache."""
    tokens = code_tokens("def clearCache(self): self.foo_bar = 1")
    assert "clearcache" in tokens and "clear" in tokens and "cache" in tokens
    assert "foo_bar" in tokens and "foo" in tokens and "bar" in tokens


def test_bm25_puts_the_file_the_issue_talks_about_first():
    docs = {
        "a.py": "def unrelated():\n    return 1\n",
        "b.py": "def clear_cache(self):\n    self._cache.clear()\n",
        "c.py": "import os\n",
    }
    ranked = [name for name, _ in
              BM25(docs).rank("clear_cache does not clear the cache")]
    assert ranked[0] == "b.py"


def test_ranking_ties_break_on_path_not_dict_order():
    """Two runs must produce the same context or they are incomparable."""
    docs = {"z.py": "x = 1\n", "a.py": "x = 1\n", "m.py": "x = 1\n"}
    ranked = [name for name, _ in BM25(docs).rank("nothing matches this")]
    assert ranked == ["a.py", "m.py", "z.py"]


# ------------------------------------------------------------- packing

def test_pack_stops_at_the_budget():
    files = {f"f{i}.py": "word " * 100 for i in range(10)}
    costs = token_costs(files, _batch)
    ctx = pack(sorted(files), files, budget=250, costs=costs)
    assert ctx.tokens <= 250
    assert 0 < len(ctx.included) < 10


def test_a_file_too_big_to_fit_does_not_end_the_packing():
    """A giant top-ranked file must not cost the arm every later file."""
    files = {"huge.py": "word " * 10_000, "small.py": "word " * 5}
    costs = token_costs(files, _batch)
    ctx = pack(["huge.py", "small.py"], files, budget=100, costs=costs)
    assert ctx.included == ["small.py"]


def test_files_are_included_whole():
    files = {"a.py": "line1\nline2\nline3\n"}
    costs = token_costs(files, _batch)
    ctx = pack(["a.py"], files, budget=1000, costs=costs)
    assert "line1" in ctx.text and "line3" in ctx.text
    assert ctx.text == render("a.py", files["a.py"])


def test_cost_accounting_includes_the_wrapper():
    """Budgets are spent on what is sent, headers and fences included."""
    files = {"a.py": "x = 1\n"}
    costs = token_costs(files, _batch)
    assert costs["a.py"] == _words(render("a.py", "x = 1\n"))
    assert costs["a.py"] > _words("x = 1\n")


# --------------------------------------------------------- the trees

def test_collect_python_skips_bytecode_and_metadata(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "core.py").write_text("x = 2\n")
    (tmp_path / "thing.egg-info").mkdir()
    (tmp_path / "thing.egg-info" / "setup.py").write_text("x = 3\n")

    files = collect_python(tmp_path)

    assert set(files) == {"pkg/core.py"}


def test_undecodable_files_are_skipped_not_fatal(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_bytes(b"x = '\xff\xfe'\n")

    files = collect_python(tmp_path)

    assert set(files) == {"ok.py"}


# ----------------------------------------------- the failing test itself

def test_the_three_identifier_shapes_all_yield_the_function():
    """pytest node ids, django labels and sympy's bare names."""
    assert failing_test_name(
        "tests/config/test_config.py::test_csv_regex_error"
    ) == "test_csv_regex_error"
    assert failing_test_name(
        "tests/_core/test_plot.py::TestLegend::test_legend_has_no_offset"
    ) == "test_legend_has_no_offset"
    assert failing_test_name(
        "test_clear_cache (apps.tests.AppsTests.test_clear_cache)"
    ) == "test_clear_cache"
    assert failing_test_name("test_issue_24543") == "test_issue_24543"


def test_only_the_named_test_is_extracted():
    """Not the whole file: seaborn's test file is 2,000 lines.

    An eighth of the budget spent on tests nobody asked about is an
    eighth the arm does not spend on the code holding the bug.
    """
    files = {"tests/t.py": (
        "def test_other():\n"
        "    assert something_else()\n"
        "\n"
        "class TestLegend:\n"
        "    def test_wanted(self):\n"
        "        assert plot().legend is None\n")}

    src = extract_test_source(files, ["tests/t.py::TestLegend::test_wanted"])

    assert "test_wanted" in src and "plot().legend" in src
    assert "something_else" not in src


def test_gold_files_reads_the_diff_headers():
    patch = (
        "diff --git a/django/apps/registry.py b/django/apps/registry.py\n"
        "--- a/django/apps/registry.py\n+++ b/django/apps/registry.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/django/apps/config.py b/django/apps/config.py\n"
        "--- a/django/apps/config.py\n+++ b/django/apps/config.py\n")
    assert gold_files(patch) == ["django/apps/config.py",
                                 "django/apps/registry.py"]
