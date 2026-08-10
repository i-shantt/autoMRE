"""Unit tests for the pieces that decide how much can be removed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from parser import (  # noqa: E402
    PythonParser,
    remap_executed_lines,
)
from reducer import (  # noqa: E402
    _apply_removals,
    _collapse_blank_lines,
    _is_parseable,
    _largest_removable_subset,
    _overlaps,
)
from multi_file.coverage_pruner import CoveragePruner  # noqa: E402
from multi_file.multi_file_validator import MultiFileValidator  # noqa: E402
from validator import Validator  # noqa: E402


NESTED = '''\
def outer():
    a = 1
    b = 2
    print(a)
    return a + b


class C:
    def m(self):
        x = 1
        return x
'''


def test_units_inside_function_bodies_are_discoverable():
    """The bug that capped minimization for the project's whole history.

    A function's body hangs off a structural `block` child. Dropping
    non-removable children severed the path to every statement inside
    every function, so HDD-E could delete a whole function but never a
    line within one.
    """
    p = PythonParser()
    units = p.get_flat_units(p.extract_units(p.parse_source(NESTED),
                                             NESTED, set()))
    kinds = {u.node_type for u in units}

    assert "function_definition" in kinds
    assert "class_definition" in kinds
    # The point of the fix:
    assert "return_statement" in kinds, "statements in bodies are invisible"
    assert any(u.node_type in ("assignment", "expression_statement")
               for u in units), "assignments in bodies are invisible"
    # The method inside the class must be reachable too.
    assert sum(1 for u in units if u.node_type == "function_definition") >= 2


DECORATED = '''\
import functools


@functools.lru_cache
def cached():
    return 2


class C:
    @property
    def prop(self):
        return 3

    def plain(self):
        return 4
'''


def _flat(source: str):
    p = PythonParser()
    return p.get_flat_units(p.extract_units(p.parse_source(source),
                                            source, set()))


def test_a_decorated_definition_is_a_removable_unit():
    """Decorated definitions were invisible, then unremovable.

    tree-sitter wraps a decorated def in a `decorated_definition` and puts
    the decorators there, outside the `function_definition`. Two separate
    consequences followed. At module level, `extract_units` dropped the
    wrapper for not being removable and took everything inside it with it,
    so the function was never a candidate at all. Inside a class the inner
    definition was reachable, but removing it cut between the decorators
    and their target, so every attempt was a syntax error and the method
    could never go.
    """
    units = _flat(DECORATED)
    kinds = [u.node_type for u in units]

    assert kinds.count("decorated_definition") == 2, (
        "the module-level decorated function and the decorated method "
        "must both be candidates")
    # Exactly one candidate per decorated definition: proposing the bare
    # inner definition as well would spend a rejection on every pass.
    starts = {(u.start_char, u.end_char) for u in units
              if u.node_type == "decorated_definition"}
    for unit in units:
        if unit.node_type == "function_definition":
            assert (unit.start_char, unit.end_char) not in starts


def test_removing_a_decorated_definition_takes_its_decorators():
    checked = 0
    for unit in _flat(DECORATED):
        if unit.node_type != "decorated_definition":
            continue
        checked += 1
        candidate = _apply_removals(DECORATED,
                                    [(unit.start_char, unit.end_char)])
        assert _is_parseable(candidate), (
            f"removing the decorated unit at L{unit.start_line} stranded "
            "its decorators, so the reducer can never remove it")
        assert unit.text not in candidate, (
            "the unit's own text survived its removal")

    # Without this the test passes by having nothing to check, which is
    # exactly the state the fix was written to leave behind.
    assert checked == 2, f"expected 2 decorated units, checked {checked}"


def test_coverage_prune_never_strands_a_decorator():
    """The expensive half of the same bug.

    A prune strips every uncovered unit in a file in one shot, so one
    uncovered decorated function made the whole file's pruned source a
    syntax error. Downstream that is indistinguishable from "coverage
    lied": the query fails, Phase 4a rolls back, and every unit in the
    file is rediscovered one query at a time.
    """
    source = '''\
import functools


def used():
    return 1


@functools.lru_cache
def never_called():
    return 2


x = used()
'''
    # Import-time lines plus the body of `used`. The decorator line runs
    # at import too, which is exactly why coverage of the *body* is what
    # decides whether a decorated function is dead.
    executed = {1, 4, 5, 8, 9, 13}

    result = CoveragePruner().prune_source(source, executed)

    assert result.any_removed, "the dead decorated function was not pruned"
    assert _is_parseable(result.pruned_source), (
        "the prune left a decorator with nothing under it")
    assert "lru_cache" not in result.pruned_source
    assert "def used" in result.pruned_source


# --------------------------------------------------------------------
# The per-query time limit
#
# A hang is not a slow query. On tomlkit-write_backslash the median query
# is 0.137 s and ten queries out of 2,699 hit the 120 s ceiling — 76% of
# all query time spent waiting for candidates that were never going to
# finish. The ceiling is now a ceiling and the working limit comes from
# what this project's queries actually cost.


def _validator(timeout, *observed):
    v = MultiFileValidator(["true"], Path("."), timeout=timeout)
    v._recent.extend(observed)
    return v


def test_the_limit_is_the_user_ceiling_until_a_query_has_been_timed():
    assert _validator(120).query_timeout == 120.0


def test_the_limit_is_floored_for_a_fast_project():
    """20 x 0.137 s is 2.7 s, which is too tight to be worth the risk."""
    v = _validator(120, 0.137, 0.14, 0.13)
    assert v.query_timeout == MultiFileValidator.TIMEOUT_FLOOR
    assert v.query_timeout < 120.0, (
        "a sub-second project still waits the full ceiling on a hang")


def test_the_limit_tracks_a_slow_project():
    assert _validator(600, 1.0, 3.0, 0.5).query_timeout == 60.0


def test_the_user_timeout_remains_a_hard_ceiling():
    """`--timeout 5` must mean at most 5s, not 20x whatever we measured."""
    assert _validator(5, 3.0).query_timeout == 5.0


def test_one_slow_query_does_not_raise_the_limit_forever():
    """A running maximum would ratchet and quietly restore the ceiling.

    One cold import or machine hiccup should not buy every later hang two
    more minutes of waiting, so the estimate is taken over a window of
    recent queries and the outlier ages out.
    """
    v = _validator(120, 4.0)
    assert v.query_timeout == 80.0, "the outlier should count while recent"

    v._recent.extend([0.1] * MultiFileValidator.TIMEOUT_WINDOW)
    assert v.query_timeout == MultiFileValidator.TIMEOUT_FLOOR, (
        "the outlier never aged out of the estimate")


def test_remap_preserves_line_content_across_removal():
    src = "import os\nimport sys\n\n\ndef dead():\n    return 1\n\n\ndef live():\n    return 2\n"
    lines = src.splitlines()
    executed = {1, 2, 9, 10}

    # remove `def dead(): return 1`
    start = src.index("def dead()")
    end = src.index("\n\n\ndef live()")
    remapped = remap_executed_lines(src, [(start, end)], executed)

    pruned = _apply_removals(src, [(start, end)])
    pruned_lines = pruned.splitlines()

    # every surviving executed line must point at identical text
    survivors = sorted(remapped)
    originals = [l for i, l in enumerate(lines, 1) if i in executed]
    mapped_text = [pruned_lines[i - 1] for i in survivors]
    for text in mapped_text:
        assert text in originals


def test_remap_is_identity_without_removals():
    src = "a = 1\nb = 2\n"
    assert remap_executed_lines(src, [], {1, 2}) == {1, 2}


@pytest.mark.parametrize("span,spans,expected", [
    ((0, 5), [(5, 10)], False),      # adjacent, not overlapping
    ((0, 6), [(5, 10)], True),       # one byte of overlap
    ((6, 8), [(5, 10)], True),       # fully contained
    ((0, 5), [], False),
])
def test_overlap_detection(span, spans, expected):
    assert _overlaps(span, spans) is expected


def test_apply_removals_is_order_independent():
    src = "AAAA BBBB CCCC"
    spans = [(0, 4), (10, 14)]
    assert _apply_removals(src, spans) == _apply_removals(src,
                                                          list(reversed(spans)))


def test_collapse_blank_lines_keeps_code_and_parses():
    src = "import os\n\n\n\n\ndef f():\n    return 1\n\n\n\n"
    out = _collapse_blank_lines(src)
    assert _is_parseable(out)
    assert "import os" in out and "def f():" in out
    assert "\n\n\n" not in out


def test_collapse_blank_lines_refuses_to_break_syntax():
    """Blank lines inside a triple-quoted string are content, not padding."""
    src = 'x = """\n\nkeep\n\n"""\n'
    out = _collapse_blank_lines(src)
    assert _is_parseable(out)


def test_syntax_precheck_rejects_emptied_block():
    assert _is_parseable("def f():\n    return 1\n")
    assert not _is_parseable("def f():\n")


# ------------------------------------------------------- normalization

def test_normalizer_ignores_traceback_line_numbers():
    """Reduction shifts every line; the same failure must still match."""
    a = ('Traceback (most recent call last):\n'
         '  File "/tmp/x/main.py", line 12, in <module>\n'
         '    boom()\n'
         'ValueError: bad')
    b = a.replace("line 12", "line 3")
    assert (Validator._normalize_output(a) ==
            Validator._normalize_output(b))


def test_normalizer_ignores_object_addresses():
    a = "<Foo object at 0x10a3b2f50>"
    b = "<Foo object at 0x7ffe12345678>"
    assert (Validator._normalize_output(a) ==
            Validator._normalize_output(b))


def test_normalizer_does_not_conflate_different_errors():
    """The guard that makes output_match a sound oracle.

    Loosening position must not loosen identity — these two are the exact
    pair the reducer previously swapped one for the other.
    """
    a = 'TypeError: can only concatenate str (not "list") to str'
    b = "TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'"
    assert (Validator._normalize_output(a) !=
            Validator._normalize_output(b))


def test_normalizer_ignores_pytest_durations():
    assert (Validator._normalize_output("5 passed in 0.08s") ==
            Validator._normalize_output("5 passed in 1.42s"))


# --------------------------------------------------------------- halving

def _counting_probe(removable):
    """An `ok` that accepts only subsets of `removable`, counting calls."""
    allowed = set(removable)
    calls = []

    def ok(spans):
        calls.append(list(spans))
        return set(spans) <= allowed

    return ok, calls


def test_halving_finds_bulk_in_log_probes():
    """The case the batch pass exists for: everything can go at once."""
    spans = [(i, i + 1) for i in range(64)]
    ok, calls = _counting_probe(spans)

    assert _largest_removable_subset(spans, ok, min_chunk=8) == spans
    assert len(calls) == 1


def test_halving_does_not_pay_2n_probes_when_nothing_is_removable():
    """The regression this guards against.

    With an unbounded descent an unremovable set costs one probe per node
    of a binary tree over the spans — ~2n, where the per-unit pass that
    runs next asks n. The batch pass then made the pipeline slower while
    finding nothing. Bounded, it must give up well short of n.
    """
    spans = [(i, i + 1) for i in range(64)]
    ok, calls = _counting_probe([])

    assert _largest_removable_subset(spans, ok, min_chunk=8) == []
    assert len(calls) < len(spans), (
        f"{len(calls)} probes to learn nothing is removable from "
        f"{len(spans)} spans; the per-unit pass would cost {len(spans)}")


def test_halving_leaves_sub_chunk_isolation_to_the_per_unit_pass():
    """Bounding the descent trades granularity away on purpose.

    A lone removable span inside a chunk of 8 is not found here. That is
    the intended split of labor, not a soundness bug: the per-unit pass
    runs next over whatever survives and will find it.
    """
    spans = [(i, i + 1) for i in range(64)]
    ok, _ = _counting_probe([spans[3]])

    assert _largest_removable_subset(spans, ok, min_chunk=8) == []
    # Unbounded, the same set is isolated exactly -- which is why the
    # comment pass, having no fallback behind it, keeps min_chunk=1.
    ok_unbounded, _ = _counting_probe([spans[3]])
    assert _largest_removable_subset(spans, ok_unbounded) == [spans[3]]


def test_halving_confirms_the_union_of_two_removable_halves():
    """Two halves each safe alone need not be safe together."""
    spans = [(0, 1), (2, 3)]
    calls = []

    def ok(subset):
        calls.append(list(subset))
        # Either alone is fine; both at once breaks.
        return len(subset) < 2

    assert _largest_removable_subset(spans, ok) in ([(0, 1)], [(2, 3)])
    assert [(0, 1), (2, 3)] in calls, "union was claimed without a probe"


# --------------------------------------------------------------- tracing

class _RecordingTracer:
    """Stands in for ExecutionTracer and counts what it was asked to do."""

    def __init__(self):
        self.file_traces = 0
        self.command_traces = 0

    def trace_python_file(self, file_path, args=None):
        self.file_traces += 1
        return self._empty()

    def trace_command(self, command, cwd=None, env=None):
        self.command_traces += 1
        return self._empty()

    @staticmethod
    def _empty():
        from tracer import ExecutionTrace
        return ExecutionTrace(executed_lines={}, output="", return_code=0,
                              success=True)


class _SeededValidator(Validator):
    """A validator that already knows the behavior to preserve."""

    def __init__(self):
        super().__init__()
        self.set_original_behavior("original output", 0)

    def validate(self, source_code, command=None, cwd=None):
        from validator import ValidationResult
        return ValidationResult(is_valid=False, output="", return_code=1,
                                matches_original=False, error_type_match=False,
                                error_message_similarity=0.0)


def test_no_standalone_trace_when_the_caller_supplies_coverage_and_oracle():
    """Phase 4b hands down both; re-deriving them costs a subprocess.

    The trace's coverage loses to `self._executed`, and the behavior it
    sets is not what ProjectFileValidator compares against, so every
    file under reduction was paying for two answers that nothing read.
    """
    from reducer import HybridDeltaDebugger

    tracer = _RecordingTracer()
    reducer = HybridDeltaDebugger(tracer=tracer, validator=_SeededValidator())
    result = reducer.reduce(source_code="A = 1\nB = 2\n",
                            test_command=None,
                            executed_lines={1})

    assert tracer.file_traces == 0 and tracer.command_traces == 0
    assert result.success


def test_standalone_trace_still_runs_when_there_is_no_oracle_yet():
    """The single-file CLI path has neither, and still needs the trace."""
    from reducer import HybridDeltaDebugger

    tracer = _RecordingTracer()
    reducer = HybridDeltaDebugger(tracer=tracer, validator=Validator())
    reducer.reduce(source_code="A = 1\n", test_command=None)

    assert tracer.file_traces == 1, (
        "without a seeded oracle the trace is the only thing that "
        "establishes the behavior to preserve")
