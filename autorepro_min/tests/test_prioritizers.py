"""Tests for the Prioritizer strategy layer.

These tests intentionally avoid any real model download — the LLM path is
exercised through a FakeBackend that returns predefined rankings. That
lets us verify: (1) heuristic behavior is unchanged, (2) LLM rankings are
applied when valid, (3) invalid rankings fall back cleanly, (4) backend
exceptions fall back cleanly, (5) unranked units are still tried
(nothing gets silently dropped).
"""

from __future__ import annotations

from typing import List

import pytest

from parser import CodeUnit
from ml.llm_backend import parse_ranking
from ml.prioritizers import (
    ErrorContext,
    HeuristicPrioritizer,
    LLMPrioritizer,
    build_prioritizer,
)


def _make_units() -> List[CodeUnit]:
    """Five toy units with varied exec counts and sizes."""
    return [
        CodeUnit("function_definition", 1, 3, 0, 30, "def hot_small():\n    ...",
                 execution_count=3),   # id 0 — small hot
        CodeUnit("function_definition", 5, 20, 40, 300, "def hot_big():\n    ...",
                 execution_count=8),   # id 1 — big hot
        CodeUnit("function_definition", 22, 25, 310, 400, "def cold_small():\n    ...",
                 execution_count=0),   # id 2 — small cold
        CodeUnit("class_definition", 27, 60, 410, 900,
                 "class ColdBig:\n    ...",
                 execution_count=0),   # id 3 — big cold
        CodeUnit("import_statement", 62, 62, 910, 940, "import unused",
                 execution_count=0),   # id 4 — tiny cold
    ]


class FakeBackend:
    """Stand-in for HFBackend used in tests. Records rank() calls."""

    def __init__(self, ranking=None, raises=False):
        self.ranking = ranking
        self.raises = raises
        self.calls = 0

    def rank(self, context, units):
        self.calls += 1
        if self.raises:
            raise RuntimeError("simulated backend failure")
        return self.ranking


# ---------- HeuristicPrioritizer --------------------------------------------

def test_heuristic_ranks_cold_before_hot():
    p = HeuristicPrioritizer()
    ordered = p.prioritize(_make_units())
    # First three should all be cold (execution_count == 0)
    assert [u.execution_count for u in ordered[:3]] == [0, 0, 0]
    # Last two should be hot
    assert all(u.execution_count > 0 for u in ordered[3:])


def test_heuristic_prefers_bigger_within_cold():
    p = HeuristicPrioritizer()
    ordered = p.prioritize(_make_units())
    cold = [u for u in ordered if u.execution_count == 0]
    # Sizes: cold_small=4 lines, ColdBig=34, unused=1 -> sort desc by size.
    assert cold[0].size >= cold[1].size >= cold[2].size


def test_heuristic_ignores_context():
    p = HeuristicPrioritizer()
    ordered_a = p.prioritize(_make_units())
    ordered_b = p.prioritize(_make_units(),
                             ErrorContext(error_type="TypeError"))
    assert [u.text for u in ordered_a] == [u.text for u in ordered_b]


# ---------- LLMPrioritizer --------------------------------------------------

def test_llm_applies_valid_ranking():
    # Ask the LLM to try units in this order: 3, 0, 4, 1, 2
    fake = FakeBackend(ranking=[3, 0, 4, 1, 2])
    p = LLMPrioritizer(backend=fake)
    units = _make_units()
    ordered = p.prioritize(units, ErrorContext())
    assert [units.index(u) for u in ordered] == [3, 0, 4, 1, 2]
    assert fake.calls == 1


def test_llm_backfills_missing_indices():
    # Model only ranks 3 of 5. Remaining 2 must still be tried, in order.
    fake = FakeBackend(ranking=[4, 2])
    p = LLMPrioritizer(backend=fake)
    units = _make_units()
    ordered = p.prioritize(units, ErrorContext())
    idxs = [units.index(u) for u in ordered]
    assert idxs[:2] == [4, 2]
    assert set(idxs) == {0, 1, 2, 3, 4}


def test_llm_ignores_out_of_range_and_duplicate_indices():
    fake = FakeBackend(ranking=[7, -1, 3, 3, 0])  # 7, -1, dup 3 all dropped
    p = LLMPrioritizer(backend=fake)
    units = _make_units()
    ordered = p.prioritize(units, ErrorContext())
    idxs = [units.index(u) for u in ordered]
    assert idxs[0] == 3
    assert idxs[1] == 0
    assert set(idxs) == {0, 1, 2, 3, 4}


def test_llm_falls_back_when_backend_raises():
    fake = FakeBackend(raises=True)
    p = LLMPrioritizer(backend=fake)
    units = _make_units()
    ordered = p.prioritize(units, ErrorContext())
    # Same result as heuristic — cold first.
    heur = HeuristicPrioritizer().prioritize(units)
    assert [u.text for u in ordered] == [u.text for u in heur]


def test_llm_falls_back_when_ranking_is_garbage():
    # All indices out of range -> ordered is empty -> fallback.
    fake = FakeBackend(ranking=[99, 100, 101])
    p = LLMPrioritizer(backend=fake)
    units = _make_units()
    ordered = p.prioritize(units, ErrorContext())
    heur = HeuristicPrioritizer().prioritize(units)
    assert [u.text for u in ordered] == [u.text for u in heur]


def test_llm_handles_empty_unit_list():
    fake = FakeBackend(ranking=[0, 1, 2])
    p = LLMPrioritizer(backend=fake)
    assert p.prioritize([], ErrorContext()) == []
    assert fake.calls == 0  # never bothers the backend


# ---------- parse_ranking (LLM response tolerance) --------------------------

@pytest.mark.parametrize("raw,expected", [
    ("[3, 0, 4, 1, 2]",                     [3, 0, 4, 1, 2]),
    ("  [1,2,3]  ",                         [1, 2, 3]),
    ('```json\n[0, 4, 2]\n```',             [0, 4, 2]),
    ('Sure! Here you go: [1, 0]. Enjoy.',   [1, 0]),
    ('[1,2,3,]',                            [1, 2, 3]),
    ('["1", "2", "3"]',                     [1, 2, 3]),
])
def test_parse_ranking_tolerant(raw, expected):
    assert parse_ranking(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "no json here at all",
    "[not, valid, json]",
])
def test_parse_ranking_returns_none_on_garbage(raw):
    assert parse_ranking(raw) is None


# ---------- build_prioritizer factory --------------------------------------

def test_build_prioritizer_defaults_to_heuristic():
    p = build_prioritizer()
    assert isinstance(p, HeuristicPrioritizer)


def test_build_prioritizer_llm_without_transformers_falls_back():
    # torch/transformers likely aren't installed in the test env; the
    # factory should degrade gracefully rather than raise.
    p = build_prioritizer(kind="llm", model="tiny")
    # Either an LLMPrioritizer (if the env DOES have transformers) or a
    # HeuristicPrioritizer (fallback). Both are acceptable — the contract
    # is "never raise from the factory".
    assert isinstance(p, (HeuristicPrioritizer, LLMPrioritizer))


def test_build_prioritizer_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_prioritizer(kind="mystery-strategy")


# ---------- ErrorContext summarization -------------------------------------

def test_error_context_summarize_prefers_structured_fields():
    ctx = ErrorContext(
        error_type="TypeError",
        error_message="object of type 'int' has no len()",
        stack_trace="File main.py, line 5\n  return len(x)",
        raw_output="ignored when structured fields exist",
    )
    text = ctx.summarize()
    assert "TypeError" in text
    assert "len()" in text
    assert "line 5" in text
    assert "ignored" not in text


def test_error_context_summarize_truncates_long_traces():
    ctx = ErrorContext(stack_trace="X" * 10000)
    text = ctx.summarize(max_chars=100)
    assert len(text) < 500  # header + 100 trailing chars, no explosion


def test_error_context_summarize_empty():
    ctx = ErrorContext()
    text = ctx.summarize()
    assert "no error context" in text.lower()
