"""
AutoRepro-Min: Multi-File Reduction
Coverage-Based Pruner

Bulk-removes definitions from a file whose lines were never executed
in the recorded coverage trace. Runs as a preprocessing step before
per-file HDD-E in Phase 4 of MultiFileDebugger.

Motivation:
    HDD-E discovers uncovered dead code by trying to remove each
    unit individually and validating. On a test file with a class
    holding one used test and 20 unused ones, that's 20 wasted
    queries the delta debugger burns learning what coverage already
    knows for free.

    The pruner does one query per file instead: strip every unit
    whose lines are absent from the coverage trace, validate the
    whole project once. If validation passes, we skipped many
    per-unit queries. If it fails (decorator side effect, dynamic
    dispatch, metaclass wiring that coverage missed), roll back and
    let HDD-E figure it out the slow way.

Traversal:
    Walks the tree-sitter tree recursively. At each node, if the
    node type is removable AND the node has zero executed lines,
    it's marked for removal and we stop descending. Otherwise we
    keep walking into children — a class with one live method and
    twenty dead ones has coverage at the class level, so we
    descend into it and prune the individual dead methods.

    Removals produced this way never overlap (once we take a node
    we don't recurse), so byte-level slicing is safe.
"""

from __future__ import annotations

import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from tree_sitter import Node

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from parser import PythonParser


@dataclass
class RemovedRange:
    """One byte-range removed from the source, tagged with its node type."""
    node_type: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


@dataclass
class PruneResult:
    """Outcome of a single-file coverage-based prune."""
    pruned_source: str
    removed: List[RemovedRange] = field(default_factory=list)
    original_line_count: int = 0
    pruned_line_count: int = 0

    @property
    def n_removed(self) -> int:
        return len(self.removed)

    @property
    def any_removed(self) -> bool:
        return bool(self.removed)


class CoveragePruner:
    """Removes every unit with zero executed lines, at any nesting level."""

    def __init__(self, parser: Optional[PythonParser] = None):
        self.parser = parser or PythonParser()

    def prune_source(self, source: str,
                     executed_lines: Set[int]) -> PruneResult:
        """Strip every uncovered unit (at any level) from `source`.

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
        to_remove = self._find_prunable(tree.root_node, executed_lines)

        if not to_remove:
            return PruneResult(source, [], original_line_count,
                               original_line_count)

        # Sort by start_byte desc so removals don't shift earlier
        # offsets. Because _find_prunable never returns a node whose
        # ancestor is also being removed, ranges never overlap.
        to_remove.sort(key=lambda r: r.start_byte, reverse=True)

        pruned = source
        for r in to_remove:
            pruned = pruned[:r.start_byte] + pruned[r.end_byte:]

        return PruneResult(pruned, to_remove, original_line_count,
                           len(pruned.splitlines()))

    def prune_file(self, file_path: Path,
                   executed_lines: Set[int]) -> PruneResult:
        """Read `file_path`, prune, write result back. Caller must
        still validate/rollback."""
        source = file_path.read_text()
        result = self.prune_source(source, executed_lines)
        if result.any_removed:
            file_path.write_text(result.pruned_source)
        return result

    # -------------------------------------------------------- internals

    def _find_prunable(self, node: Node,
                       executed_lines: Set[int]) -> List[RemovedRange]:
        """Recursive walk yielding non-overlapping removable ranges.

        Rules:
          * function_definition is a LEAF: whole function or nothing.
            Coverage is checked against the function BODY only,
            because the `def` line runs at import time even when
            the function is never called — using it would mark every
            top-level function as covered and defeat the pruner.
          * Any other removable node with zero covered lines is
            taken whole.
          * Partially-covered classes and modules are descended into
            so we can prune individual uncovered methods.
        """
        node_type = node.type

        if node_type == "function_definition":
            body = self._function_body(node)
            body_touched = (self._touches(body, executed_lines)
                            if body is not None
                            else self._touches(node, executed_lines))
            if body_touched:
                return []
            return [self._to_range(node)]

        if (node_type in self.parser.REMOVABLE_TYPES
                and not self._touches(node, executed_lines)):
            return [self._to_range(node)]

        collected: List[RemovedRange] = []
        for child in node.children:
            collected.extend(self._find_prunable(child, executed_lines))
        return collected

    @staticmethod
    def _function_body(node: Node) -> Optional[Node]:
        """Return the `block` child of a function_definition, if any."""
        for child in node.children:
            if child.type == "block":
                return child
        return None

    @staticmethod
    def _touches(node: Node, executed_lines: Set[int]) -> bool:
        start = node.start_point[0] + 1
        end = node.end_point[0] + 1
        for line in range(start, end + 1):
            if line in executed_lines:
                return True
        return False

    @staticmethod
    def _to_range(node: Node) -> RemovedRange:
        return RemovedRange(
            node_type=node.type,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
        )
