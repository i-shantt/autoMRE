"""Removal spans must line up with the source on non-ASCII files.

tree-sitter reports offsets into the UTF-8 *bytes* of a file. Everything
downstream here slices a Python `str`, which is indexed by *characters*.
The two agree exactly as long as the file is ASCII and diverge from the
first character that is not, by one index per extra byte.

This is not a corner case dressed up as a bug. `psf/requests` ships
`src/requests/certs.py` with a single em dash in its module docstring,
and that one character is enough to push every later span out of
alignment: the import statement slices as `om certifi import where`, and
the last statement in the file starts past the end of the string, so
"removing" it removes nothing at all.

Both failure modes cost something. A mis-sliced candidate is garbage,
the syntax check rejects it, and the unit it came from can never be
removed — reduction quietly gets worse on exactly the files that have a
stray Unicode character in a comment. A no-op candidate is worse: it
validates (nothing changed, so the bug still reproduces), the pass
records a removal, the source does not shrink, and the outer loop runs
again on identical input until `max_iterations` stops it. Measured on
the benchmark, that spent 100 of the 1319 queries of a `psf/requests`
task re-asking one question.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))

from parser import PythonParser, remap_executed_lines  # noqa: E402
from reducer import _apply_removals  # noqa: E402
from multi_file.coverage_pruner import CoveragePruner  # noqa: E402


# An em dash in the docstring, then ordinary statements after it. Shaped
# after requests/certs.py, which is where this was found.
EM_DASH_SOURCE = '''\
"""Module doc — with an em dash in it."""
from os import path

KEEP = 1
DROP = 2
'''


def _units(source):
    parser = PythonParser()
    tree = parser.parse_source(source)
    return parser.get_flat_units(parser.extract_units(tree, source, set()))


def test_source_really_is_misaligned():
    """Guard the guard: this file must actually have byte/char drift."""
    assert len(EM_DASH_SOURCE.encode("utf-8")) > len(EM_DASH_SOURCE)


def test_every_unit_slices_to_its_own_text():
    """A unit's span must select that unit, not something two bytes off."""
    for unit in _units(EM_DASH_SOURCE):
        sliced = EM_DASH_SOURCE[unit.start_char:unit.end_char]
        assert sliced == unit.text, (
            f"{unit.node_type} span {unit.start_char}:{unit.end_char} "
            f"slices {sliced!r}, expected {unit.text!r}")


def test_removing_a_unit_removes_exactly_that_unit():
    for unit in _units(EM_DASH_SOURCE):
        span = (unit.start_char, unit.end_char)
        after = _apply_removals(EM_DASH_SOURCE, [span])
        assert after != EM_DASH_SOURCE, (
            f"removing {unit.node_type} at {span} changed nothing; a "
            "candidate identical to its input validates, is recorded as a "
            "removal, and sends the pass round again on the same source")
        assert len(after) == len(EM_DASH_SOURCE) - (
            unit.end_char - unit.start_char)


def test_no_unit_span_runs_past_the_end_of_the_source():
    for unit in _units(EM_DASH_SOURCE):
        assert unit.end_char <= len(EM_DASH_SOURCE), (
            f"{unit.node_type} ends at {unit.end_char} in a source of "
            f"{len(EM_DASH_SOURCE)} characters")


def test_removing_the_last_statement_leaves_the_rest_intact():
    """The concrete regression: DROP goes, KEEP stays, syntax survives."""
    last = [u for u in _units(EM_DASH_SOURCE)
            if u.text.startswith("DROP")]
    assert last, "expected a DROP = 2 statement among the units"
    after = _apply_removals(EM_DASH_SOURCE,
                            [(last[0].start_char, last[0].end_char)])
    assert "DROP" not in after
    assert "KEEP = 1" in after
    assert "from os import path" in after
    compile(after, "<candidate>", "exec")


def test_candidates_drop_spans_that_would_remove_nothing():
    """Defence in depth: a no-op span must never reach the query loop.

    The offsets are fixed, so the parser no longer produces these. This
    pins the guard that stops the *next* span bug from costing a hundred
    queries instead of merely being wrong, by feeding the reducer units
    the parser would not currently build.
    """
    from reducer import HybridDeltaDebugger  # noqa: WPS433
    from parser import CodeUnit

    source = "A = 1\nB = 2\n"
    empty = CodeUnit("expression_statement", 1, 1, 6, 6, "")
    past_end = CodeUnit("expression_statement", 2, 2, 99, 120, "")
    real = CodeUnit("expression_statement", 1, 1, 0, 5, "A = 1")

    debugger = HybridDeltaDebugger()
    debugger._get_flat_units = lambda src, trace: [empty, past_end, real]

    kept = debugger._candidates(source, trace=None)
    assert [(u.start_char, u.end_char) for u in kept] == [(0, 5)], (
        "a zero-width span and one starting past the end of the source "
        "both remove nothing and must not be offered as candidates")


def test_coverage_pruner_cuts_on_character_boundaries():
    """The Phase 4a pruner slices the same source with the same offsets."""
    executed = {2}  # only the import ran
    result = CoveragePruner().prune_source(EM_DASH_SOURCE, executed)
    assert result.any_removed
    compile(result.pruned_source, "<pruned>", "exec")
    assert "from os import path" in result.pruned_source


def test_coverage_remap_counts_characters_not_bytes():
    """Coverage held across a removal must survive the em dash too.

    Two cases, because they check different halves of the walk. A unit
    span stops short of its trailing newline, so the line goes blank and
    nothing after it moves; a span that swallows the newline really does
    renumber the rest of the file. Getting either wrong on a non-ASCII
    file means Phase 4b prioritises against lines that are not there.
    """
    keep = [u for u in _units(EM_DASH_SOURCE) if u.text.startswith("KEEP")][0]

    # (a) unit span: line 4 is emptied, so line 5 keeps its number.
    remapped = remap_executed_lines(
        EM_DASH_SOURCE, [(keep.start_char, keep.end_char)], {2, 5})
    assert remapped == {2, 5}

    # (b) same cut extended over the newline: line 5 becomes line 4.
    remapped = remap_executed_lines(
        EM_DASH_SOURCE, [(keep.start_char, keep.end_char + 1)], {2, 5})
    assert remapped == {2, 4}, (
        "a removal that takes the newline must renumber what follows; "
        f"got {sorted(remapped)}")
