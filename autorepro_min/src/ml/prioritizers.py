"""
AutoRepro-Min: Prioritizer Strategy Layer

A `Prioritizer` decides the order in which candidate `CodeUnit`s are tried
for removal during delta debugging. Query count — the dominant cost — is
directly a function of prioritization quality: put the actually-irrelevant
units first and every one succeeds, put the load-bearing units first and
every one fails before you eventually find something removable.

Two implementations ship today:

* `HeuristicPrioritizer` — wraps the existing cold-code-first sort in
  `PythonParser.prioritize_units`. Zero dependencies, deterministic.
* `LLMPrioritizer` — asks an open-source code LLM to rank candidates given
  the reproduction error context. Falls back to `HeuristicPrioritizer` if
  the backend isn't installed, fails to load, or returns garbage.

A future `LearnedPrioritizer` (fine-tuned code encoder) is planned as a
follow-up.
"""

from __future__ import annotations

import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from parser import CodeUnit, PythonParser


@dataclass
class ErrorContext:
    """The signal available to a prioritizer about *why* the code fails.

    Populated from the original reproduction run and passed unchanged into
    every prioritization call within a reduction session.
    """
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    raw_output: Optional[str] = None
    return_code: int = 0

    def summarize(self, max_chars: int = 2000) -> str:
        """Compact human-readable summary suitable for LLM context."""
        parts: List[str] = []
        if self.error_type:
            parts.append(f"Error type: {self.error_type}")
        if self.error_message:
            parts.append(f"Error message: {self.error_message}")
        if self.stack_trace:
            trace = self.stack_trace.strip()
            if len(trace) > max_chars:
                trace = trace[-max_chars:]  # keep the innermost frames
            parts.append(f"Stack trace:\n{trace}")
        if not parts and self.raw_output:
            snippet = self.raw_output.strip()
            if len(snippet) > max_chars:
                snippet = snippet[-max_chars:]
            parts.append(f"Output:\n{snippet}")
        return "\n\n".join(parts) or "(no error context available)"


class Prioritizer(Protocol):
    """Strategy interface: rank removable units for delta debugging."""

    def prioritize(self, units: List[CodeUnit],
                   context: Optional[ErrorContext] = None) -> List[CodeUnit]:
        ...


class HeuristicPrioritizer:
    """Cold-code-first / larger-first sort (the pre-ML baseline).

    Ignores `context` — this is what HDD-E used before the ML additions and
    what the tool falls back to whenever an LLM path fails.
    """

    def __init__(self, parser: Optional[PythonParser] = None):
        self._parser = parser or PythonParser()

    def prioritize(self, units: List[CodeUnit],
                   context: Optional[ErrorContext] = None) -> List[CodeUnit]:
        return self._parser.prioritize_units(units)


class LLMPrioritizer:
    """Rank candidates with an open-source code LLM.

    Delegates the actual model call to a `backend` object exposing
    `rank(context: ErrorContext, units: List[CodeUnit]) -> List[int]`
    where the returned list contains unit indices in preferred-removal
    order. Any exception, missing dependency, or malformed ranking causes
    a clean fallback to the heuristic — the tool must never break because
    the ML side broke.
    """

    def __init__(self, backend, fallback: Optional[Prioritizer] = None,
                 verbose: bool = False):
        self._backend = backend
        self._fallback = fallback or HeuristicPrioritizer()
        self._verbose = verbose

    def prioritize(self, units: List[CodeUnit],
                   context: Optional[ErrorContext] = None) -> List[CodeUnit]:
        if not units:
            return units
        try:
            ranking = self._backend.rank(context or ErrorContext(), units)
        except Exception as exc:
            if self._verbose:
                print(f"[LLMPrioritizer] backend error: {exc}; "
                      "falling back to heuristic")
            return self._fallback.prioritize(units, context)

        ordered = self._reorder(units, ranking)
        if ordered is None:
            if self._verbose:
                print("[LLMPrioritizer] backend returned invalid ranking; "
                      "falling back to heuristic")
            return self._fallback.prioritize(units, context)
        return ordered

    @staticmethod
    def _reorder(units: List[CodeUnit],
                 ranking: List[int]) -> Optional[List[CodeUnit]]:
        """Apply an index ranking; append missing indices in original order.

        Returns None if the model's ranking contained ZERO valid indices —
        that's the "total garbage" case where we should abandon the LLM
        output and let the caller fall back to the heuristic. Partial
        rankings (some valid, some not) are honored and the missing units
        are backfilled in original order so nothing is silently dropped.
        """
        n = len(units)
        seen = set()
        ordered: List[CodeUnit] = []
        for idx in ranking:
            if not isinstance(idx, int) or idx < 0 or idx >= n or idx in seen:
                continue
            seen.add(idx)
            ordered.append(units[idx])

        if not seen:
            # Zero usable indices — treat as backend failure.
            return None

        for i, u in enumerate(units):
            if i not in seen:
                ordered.append(u)
        return ordered


def build_prioritizer(kind: str = "heuristic", *, model: Optional[str] = None,
                      verbose: bool = False) -> Prioritizer:
    """Factory used by the CLI and MultiFileDebugger.

    kind : "heuristic" (default) or "llm".
    model: name from ml.model_registry.MODEL_TIERS (only used when kind="llm").
    """
    kind = (kind or "heuristic").lower()
    if kind == "heuristic":
        return HeuristicPrioritizer()
    if kind == "llm":
        # Import lazily so the default install never touches torch/transformers.
        from .llm_backend import build_llm_backend
        backend = build_llm_backend(model=model, verbose=verbose)
        if backend is None:
            if verbose:
                print("[build_prioritizer] LLM backend unavailable; "
                      "using heuristic")
            return HeuristicPrioritizer()
        return LLMPrioritizer(backend=backend, verbose=verbose)
    raise ValueError(f"unknown prioritizer kind: {kind!r}")
