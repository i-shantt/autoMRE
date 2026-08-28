"""A coverage prune must not hollow a block out into a syntax error.

Phase 4a bets a whole file's uncovered units on one query, and refuses
to spend that query when the pruned source does not parse. So a single
unparseable removal costs no query — it costs the entire file's prune.
Every other legitimate removal is discarded with it, and HDD-E goes on
to rediscover those units one query at a time, which is exactly the cost
the pruner exists to avoid.

Emptying a block is the easy way to get there, and coverage makes it
likely rather than rare: a class's `def` lines run at import even when
no method is ever called, so a wholly dead class is never taken whole —
it is descended into and hollowed out method by method until `class
Helper:` has nothing under it. An `if` whose condition ran false leaves
the same shape.

These are unit tests on the pruner because that is where the shape is
decided; the end-to-end claim is that reduction of such a file does not
silently fall back to per-unit discovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from multi_file.coverage_pruner import CoveragePruner  # noqa: E402


def _prune(source, executed):
    result = CoveragePruner().prune_source(source, set(executed))
    compile(result.pruned_source, "<pruned>", "exec")   # the whole point
    return result


def test_a_class_whose_methods_are_all_dead_still_parses():
    source = (
        "class Helper:\n"
        "    def unused_a(self):\n"
        "        return 1\n"
        "\n"
        "    def unused_b(self):\n"
        "        return 2\n"
        "\n"
        "print('live')\n"
    )
    # Coverage records the class statement and both `def` lines at import
    # time; neither body ever ran.
    result = _prune(source, {1, 2, 5, 8})

    assert result.n_removed == 1          # one method goes, one stays
    assert "print('live')" in result.pruned_source


def test_an_if_whose_body_never_ran_still_parses():
    source = ("flag = False\n"
              "if flag:\n"
              "    do_a()\n"
              "    do_b()\n"
              "print('live')\n")

    result = _prune(source, {1, 2, 5})

    assert result.n_removed == 1


def test_a_live_method_keeps_the_class_open_so_every_dead_one_goes():
    """The block is not at risk here, so nothing is held back."""
    source = ("class Helper:\n"
              "    def used(self):\n"
              "        return 1\n"
              "\n"
              "    def dead(self):\n"
              "        return 2\n")

    result = _prune(source, {1, 2, 3, 5})

    assert result.n_removed == 1
    assert "def used" in result.pruned_source
    assert "def dead" not in result.pruned_source


def test_a_comment_is_not_a_body():
    """Keeping a comment where a statement was is still a syntax error."""
    source = ("class Helper:\n"
              "    # explains the method below\n"
              "    def dead(self):\n"
              "        return 2\n"
              "\n"
              "print('live')\n")

    result = _prune(source, {1, 3, 6})

    assert result.n_removed == 0
    assert "def dead" in result.pruned_source


def test_emptying_a_module_is_allowed():
    """A file with nothing in it parses; only blocks need a body."""
    result = _prune("def a():\n    return 1\n", {1})

    assert result.n_removed == 1
    assert result.pruned_source.strip() == ""
