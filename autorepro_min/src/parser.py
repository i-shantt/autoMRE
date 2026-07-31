"""
AutoRepro-Min: Automated Bug Reproduction Minimization
AST Parser Module

This module provides Python AST parsing capabilities using tree-sitter
to identify candidate removable units for delta debugging.

Based on:
- Zeller & Hildebrandt (2002): ddmin algorithm
- Misherghi & Su (2006): Hierarchical Delta Debugging
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tree_sitter import Language, Node, Parser, Tree
import tree_sitter_python as tspython


def byte_to_char_offsets(source: str) -> Optional[List[int]]:
    """Table translating UTF-8 byte offsets into `str` indices.

    tree-sitter reports every node position as an offset into the file's
    UTF-8 *bytes*. Everything downstream slices a Python `str`, which is
    indexed by *characters*. The two agree exactly while the source is
    ASCII and drift apart from the first character that is not, by one
    index per extra byte.

    Returns None for pure-ASCII sources, where the identity mapping is
    correct and building a table would be waste; callers treat None as
    "offsets already line up". That is the common case — 109 of the 118
    files in the two benchmark repos.
    """
    encoded = source.encode("utf-8")
    if len(encoded) == len(source):
        return None

    table: List[int] = []
    for index, char in enumerate(source):
        table.extend([index] * len(char.encode("utf-8")))
    table.append(len(source))
    return table


def to_char_offset(table: Optional[List[int]], offset: int) -> int:
    """One byte offset in `str` indices, given a table from above."""
    if table is None:
        return offset
    if offset >= len(table):
        return table[-1]
    return table[offset]


@dataclass
class CodeUnit:
    """Represents a removable unit of code for delta debugging.

    `start_char`/`end_char` index the source as a `str`, not as bytes.
    They were once named for bytes and carried tree-sitter's byte
    offsets straight into `str` slicing, which silently cut in the wrong
    place on any file containing a non-ASCII character. See
    `byte_to_char_offsets` and tests/test_non_ascii_offsets.py.
    """
    node_type: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    text: str
    children: List[CodeUnit] = field(default_factory=list)
    execution_count: int = 0  # From coverage data
    can_remove: bool = True
    # The tree-sitter node this unit came from. Carried so the learned
    # oracle can read structural context (parent decorators, depth,
    # siblings) that the flattened CodeUnit drops. Excluded from equality
    # and repr: it is provenance, not identity, and Node has no useful
    # printed form.
    ts_node: Optional[Node] = field(default=None, compare=False, repr=False)

    @property
    def size(self) -> int:
        """Return the size of this unit in lines."""
        return self.end_line - self.start_line + 1

    @property
    def is_hot(self) -> bool:
        """Return True if this unit was executed (based on coverage)."""
        return self.execution_count > 0

    def __repr__(self) -> str:
        return f"CodeUnit({self.node_type}, L{self.start_line}-{self.end_line}, exec={self.execution_count})"


class PythonParser:
    """
    Python AST parser using tree-sitter.

    Identifies removable code units at multiple granularities:
    - Module-level: classes, functions, imports
    - Function-level: statements, docstrings
    - Statement-level: expressions
    """

    # Node types that can be removed (leaf units for HDD)
    REMOVABLE_TYPES = {
        'function_definition',
        'class_definition',
        'import_statement',
        'import_from_statement',
        'expression_statement',
        'if_statement',
        'for_statement',
        'while_statement',
        'try_statement',
        'with_statement',
        'return_statement',
        'pass_statement',
        'break_statement',
        'continue_statement',
        'raise_statement',
        'assert_statement',
        'assignment',
        'augmented_assignment',
    }

    # Node types that should not be removed individually (structural)
    STRUCTURAL_TYPES = {
        'module',
        'block',
        'elif_clause',
        'else_clause',
        'except_clause',
        'finally_clause',
    }

    def __init__(self):
        """Initialize the parser with Python grammar."""
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)

    def parse_file(self, file_path: Path | str) -> Tuple[Tree, str]:
        """
        Parse a Python file and return the parse tree.

        Args:
            file_path: Path to the Python file

        Returns:
            Tuple of (parse tree, source code)
        """
        file_path = Path(file_path)
        source = file_path.read_text()
        tree = self.parser.parse(source.encode('utf-8'))
        return tree, source

    def parse_source(self, source: str) -> Tree:
        """
        Parse Python source code and return the parse tree.

        Args:
            source: Python source code string

        Returns:
            Parse tree
        """
        return self.parser.parse(source.encode('utf-8'))

    def extract_units(self, tree: Tree, source: str,
                      execution_lines: Optional[Set[int]] = None) -> List[CodeUnit]:
        """
        Extract removable code units from a parse tree.

        Args:
            tree: Parse tree from tree-sitter
            source: Source code string
            execution_lines: Set of line numbers that were executed

        Returns:
            List of CodeUnit objects representing removable units
        """
        execution_lines = execution_lines or set()
        root = tree.root_node
        units = []
        # Built once per parse and threaded through the recursion; the
        # ASCII fast path makes it None and costs nothing.
        offsets = byte_to_char_offsets(source)

        for child in root.children:
            unit = self._node_to_unit(child, source, execution_lines, offsets)
            if unit and unit.can_remove:
                units.append(unit)

        return units

    def _node_to_unit(self, node: Node, source: str,
                      execution_lines: Set[int],
                      offsets: Optional[List[int]] = None
                      ) -> Optional[CodeUnit]:
        """
        Convert a tree-sitter node to a CodeUnit.

        Args:
            node: Tree-sitter node
            source: Source code string
            execution_lines: Set of executed line numbers

        Returns:
            CodeUnit or None if node cannot be a removable unit
        """
        start_char = to_char_offset(offsets, node.start_byte)
        end_char = to_char_offset(offsets, node.end_byte)

        if node.type in self.STRUCTURAL_TYPES:
            # Structural nodes - extract their children instead
            children = []
            for child in node.children:
                child_unit = self._node_to_unit(child, source, execution_lines,
                                                offsets)
                if child_unit:
                    children.append(child_unit)

            if children:
                # Return the structural node with children
                return CodeUnit(
                    node_type=node.type,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_char=start_char,
                    end_char=end_char,
                    text=source[start_char:end_char],
                    children=children,
                    execution_count=self._count_execution(node, execution_lines),
                    can_remove=False,  # Can't remove structural nodes directly
                    ts_node=node
                )
            return None

        if node.type not in self.REMOVABLE_TYPES:
            # Unknown node type - check if it has removable children
            children = []
            for child in node.children:
                child_unit = self._node_to_unit(child, source, execution_lines,
                                                offsets)
                if child_unit:
                    children.append(child_unit)

            if children:
                return CodeUnit(
                    node_type=node.type,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_char=start_char,
                    end_char=end_char,
                    text=source[start_char:end_char],
                    children=children,
                    execution_count=self._count_execution(node, execution_lines),
                    can_remove=False,
                    ts_node=node
                )
            return None

        # This is a removable unit. Keep descending so nested removable
        # units stay reachable.
        #
        # Structural children must be kept even though they can't be
        # removed themselves. A function_definition's children are
        # `def`, the name, the parameters, `:` and a `block`; the block
        # is structural, so dropping non-removable children here severed
        # the only path to every statement in the body. The effect was
        # that HDD-E saw whole top-level definitions and nothing else —
        # it could delete an entire function but never a line inside
        # one, and never a method inside a class.
        children = []
        for child in node.children:
            child_unit = self._node_to_unit(child, source, execution_lines,
                                                offsets)
            if child_unit is not None:
                children.append(child_unit)

        return CodeUnit(
            node_type=node.type,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_char=start_char,
            end_char=end_char,
            text=source[start_char:end_char],
            children=children,
            execution_count=self._count_execution(node, execution_lines),
            can_remove=True,
            ts_node=node
        )

    def _count_execution(self, node: Node, execution_lines: Set[int]) -> int:
        """
        Count how many lines in this node were executed.

        Args:
            node: Tree-sitter node
            execution_lines: Set of executed line numbers

        Returns:
            Count of executed lines in this node
        """
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        count = 0
        for line in range(start_line, end_line + 1):
            if line in execution_lines:
                count += 1
        return count

    def get_flat_units(self, units: List[CodeUnit]) -> List[CodeUnit]:
        """
        Flatten hierarchical units into a single list.

        Args:
            units: List of hierarchical CodeUnit objects

        Returns:
            Flattened list of all removable units
        """
        flat = []
        for unit in units:
            if unit.can_remove:
                flat.append(unit)
            flat.extend(self.get_flat_units(unit.children))
        return flat

    def prioritize_units(self, units: List[CodeUnit]) -> List[CodeUnit]:
        """
        Prioritize units for reduction based on execution count and size.

        Strategy (inspired by WDD - Zhou et al., 2024):
        1. Prefer removing cold code (not executed) first
        2. Among cold code, prefer larger units
        3. Among hot code, prefer larger units with lower execution count

        Args:
            units: List of CodeUnit objects

        Returns:
            Sorted list of units by priority
        """
        def priority_key(unit: CodeUnit) -> Tuple:
            # Cold code (not executed) gets highest priority
            is_cold = unit.execution_count == 0
            # Larger units are tried first (coarse to fine)
            size = unit.size
            # Among hot code, lower execution = higher priority
            exec_count = unit.execution_count

            # Return tuple for sorting: (cold first, larger first, lower exec first)
            return (not is_cold, -size, exec_count)

        return sorted(units, key=priority_key)


def extract_line_ranges(units: List[CodeUnit]) -> Set[int]:
    """
    Extract all line numbers covered by the given units.

    Args:
        units: List of CodeUnit objects

    Returns:
        Set of line numbers
    """
    lines = set()
    for unit in units:
        lines.update(range(unit.start_line, unit.end_line + 1))
    return lines


def remap_executed_lines(source: str,
                         removed_char_ranges: List[Tuple[int, int]],
                         executed: Set[int]) -> Set[int]:
    """Translate coverage line numbers across a character-range removal.

    Deleting bytes shifts every line number after the cut, so any
    coverage set held across a removal silently starts describing the
    wrong lines. This walks the pre-removal source once and returns the
    executed set expressed in the post-removal numbering. Lines that were
    removed outright drop out.

    Removals slice [start, end), stopping short of the unit's trailing
    newline, so a removal usually leaves a blank line behind rather than
    collapsing the line. The character-level walk handles that — and
    partial-line cuts — exactly, where subtracting line spans would not.

    Offsets are `str` indices, matching CodeUnit.start_char. Feeding it
    tree-sitter's byte offsets would misplace every cut in a file with
    a non-ASCII character in it.
    """
    if not removed_char_ranges or not executed:
        return set(executed)

    kept = bytearray(b"\x01") * len(source)
    for start, end in removed_char_ranges:
        for i in range(max(0, start), min(end, len(source))):
            kept[i] = 0

    # Record each original line's new number at its first surviving byte.
    mapping: Dict[int, int] = {}
    old_line = 1
    new_line = 1
    for i, ch in enumerate(source):
        if kept[i] and old_line not in mapping:
            mapping[old_line] = new_line
        if ch == '\n':
            old_line += 1
            if kept[i]:
                new_line += 1

    return {mapping[line] for line in executed if line in mapping}


def format_code_without_units(source: str, units_to_remove: List[CodeUnit]) -> str:
    """
    Remove the specified units from the source code.

    Args:
        source: Original source code
        units_to_remove: List of units to remove

    Returns:
        Source code with units removed
    """
    if not units_to_remove:
        return source

    # Sort by position (reverse order to maintain byte indices)
    units_to_remove = sorted(units_to_remove,
                            key=lambda u: u.start_char,
                            reverse=True)

    result = source
    for unit in units_to_remove:
        result = result[:unit.start_char] + result[unit.end_char:]

    return result


# For backward compatibility and direct use
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parser.py <python_file>")
        sys.exit(1)

    parser = PythonParser()
    tree, source = parser.parse_file(sys.argv[1])
    units = parser.extract_units(tree, source)

    print(f"Found {len(units)} top-level removable units:")
    for unit in units:
        print(f"  {unit}")
        if unit.children:
            print(f"    {len(unit.children)} nested units")
