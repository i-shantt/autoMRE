"""
AutoRepro-Min: Multi-File Reduction
Coverage-Based Pruner

Bulk-removes top-level definitions from a file whose lines were never
executed in the recorded coverage trace. Runs as a preprocessing step
before per-file HDD-E in Phase 4 of MultiFileDebugger.

Motivation:
    HDD-E discovers uncovered dead code by trying to remove each
    top-level unit individually and validating. On a file with 20
    top-level defs where 15 are cold, that's 15 wasted queries the
    delta debugger has to make just to learn what coverage already
    knows for free.

    The pruner does one query per file instead: strip every uncovered
    top-level unit at once, validate. If the removal preserved the
    bug, the debugger inherits a much smaller starting point. If it
    broke the bug (e.g. a decorator side effect that coverage.py
    missed), we roll back and let HDD-E figure it out the slow way.

Design notes:
  * Top-level only. Nested pruning (removing cold branches inside a
    covered function) is left to HDD-E — the payoff/risk ratio at
    that granularity is worse and the delta debugger handles it fine.
  * Respects PythonParser.REMOVABLE_TYPES via extract_units — we don't
    strip structural nodes we couldn't remove anyway.
  * A file with an empty executed_lines set is skipped entirely: we
    have no evidence to prune from, and pruning everything would
    definitely break any file that survived Phase 2/3.
"""

from __future__ import annotations

import sys as _sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from parser import CodeUnit, PythonParser, format_code_without_units


@dataclass
class PruneResult:
    """Outcome of a single-file coverage-based prune."""
    pruned_source: str
    removed_units: List[CodeUnit]
    original_line_count: int
    pruned_line_count: int

    @property
    def n_removed(self) -> int:
        return len(self.removed_units)

    @property
    def any_removed(self) -> bool:
        return bool(self.removed_units)


class CoveragePruner:
    """Removes top-level units with zero executed lines."""

    def __init__(self, parser: Optional[PythonParser] = None):
        self.parser = parser or PythonParser()

    def prune_source(self, source: str,
                     executed_lines: Set[int]) -> PruneResult:
        """Strip every uncovered top-level unit from `source`.

        Args:
            source: File contents.
            executed_lines: 1-indexed line numbers that ran under
                coverage. If empty, no pruning happens.
        """
        original_line_count = len(source.splitlines())

        if not executed_lines or not source.strip():
            return PruneResult(source, [], original_line_count,
                               original_line_count)

        tree = self.parser.parse_source(source)
        top_units = self.parser.extract_units(tree, source,
                                              execution_lines=executed_lines)

        uncovered = [u for u in top_units
                     if u.can_remove and not self._touches(u, executed_lines)]

        if not uncovered:
            return PruneResult(source, [], original_line_count,
                               original_line_count)

        pruned = format_code_without_units(source, uncovered)
        return PruneResult(pruned, uncovered, original_line_count,
                           len(pruned.splitlines()))

    def prune_file(self, file_path: Path,
                   executed_lines: Set[int]) -> PruneResult:
        """Convenience: read `file_path`, prune, write result back.

        The caller is responsible for validation and rollback — this
        just mutates the file on disk.
        """
        source = file_path.read_text()
        result = self.prune_source(source, executed_lines)
        if result.any_removed:
            file_path.write_text(result.pruned_source)
        return result

    @staticmethod
    def _touches(unit: CodeUnit, executed_lines: Set[int]) -> bool:
        """Does any line inside `unit` appear in `executed_lines`?"""
        for line in range(unit.start_line, unit.end_line + 1):
            if line in executed_lines:
                return True
        return False
