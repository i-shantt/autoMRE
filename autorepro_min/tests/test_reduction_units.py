"""Unit tests for the pieces that decide how much can be removed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))

from parser import (  # noqa: E402
    PythonParser,
    remap_executed_lines,
)
from reducer import (  # noqa: E402
    _apply_removals,
    _collapse_blank_lines,
    _is_parseable,
    _overlaps,
)
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
