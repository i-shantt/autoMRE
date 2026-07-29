"""
AutoRepro-Min: Learned Removability Oracle
Feature Extraction

Turns one tree-sitter node into a fixed-width numeric vector describing
"what kind of code is this, and how does it sit in its file". The oracle
consumes these vectors to predict whether removing the unit will keep the
bug reproducing.

Design notes
------------
The features deliberately cover the failure modes coverage.py cannot see,
because those are exactly where CoveragePruner loses its bet (12 rollbacks
per benchmark run):

  * Decorator machinery. A @pytest.fixture that is never "executed" in the
    line-coverage sense is still load-bearing — collecting the test module
    evaluates the decorator and registers the fixture. Coverage reports the
    body as cold; deleting it breaks collection. So decorator kind gets its
    own indicators rather than being folded into node type.
  * Import-time vs call-time execution. A function's `def` line runs when
    the module is imported even if the function is never called, which is
    why `coverage_ratio` (whole node) and `body_coverage_ratio` (the block
    child only) are separate features. CoveragePruner already relies on
    this distinction; here we hand both numbers to the model and let it
    decide.
  * Name reachability. A symbol referenced elsewhere in the same file is
    far more likely to be load-bearing than a dead one, and that is cheap
    to measure without a full import graph.

Vector layout is pinned by FEATURE_NAMES. Order must stay stable — a
pickled model is meaningless if the columns shift underneath it — so
to_vector() is built from that tuple and asserts its own width.

Line numbers are 1-indexed here to match coverage.py; tree-sitter's
start_point/end_point are 0-indexed, so every conversion adds 1. This
mirrors CoveragePruner._touches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from tree_sitter import Node


# Node types common enough in real Python to be worth their own indicator.
# Anything outside this list falls into the `node_type_other` bucket.
NODE_TYPES: Tuple[str, ...] = (
    "function_definition",
    "class_definition",
    "decorated_definition",
    "import_statement",
    "import_from_statement",
    "expression_statement",
    "assignment",
    "augmented_assignment",
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "return_statement",
)

# Decorator families that change removability. The pytest ones dominate the
# rollback cases; the descriptor ones (classmethod/staticmethod/property)
# matter because removing them changes attribute access on a live class.
DECORATOR_KINDS: Tuple[str, ...] = (
    "pytest_fixture",
    "pytest_mark",
    "pytest_parametrize",
    "classmethod",
    "staticmethod",
    "property",
    "other",
)

# Substrings suggesting the unit does something at import time that a
# line-coverage trace won't attribute back to it.
SIDE_EFFECT_TOKENS: Tuple[str, ...] = (
    "register", "atexit", "signal.", "importlib", "__import__",
    "setattr", "globals()", "locals()", "sys.path", "os.environ",
    "logging.", "warnings.", "socket", "subprocess", "open(",
)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _one_hot_names(prefix: str, values: Sequence[str]) -> List[str]:
    return [f"{prefix}{v}" for v in values]


# Canonical column order. Changing this invalidates any pickled model, so
# treat it as a schema: append, never reorder or delete.
FEATURE_NAMES: Tuple[str, ...] = tuple(
    _one_hot_names("node_type_", NODE_TYPES + ("other",))
    + [
        # size
        "n_lines",
        "n_bytes",
        # coverage
        "n_covered_lines",
        "coverage_ratio",
        "body_coverage_ratio",
        "is_fully_uncovered",
    ]
    + _one_hot_names("dec_", DECORATOR_KINDS)
    + [
        "n_decorators",
        # file context
        "is_test_file",
        "is_conftest",
        "is_init_file",
        # structural position
        "depth_in_ast",
        "is_module_top_level",
        "sibling_count",
        "n_child_units",
        # naming / reachability
        "references_in_file",
        "has_side_effect_token",
        "is_dunder",
        "is_private",
    ]
)


@dataclass
class UnitFeatures:
    """Numeric description of one candidate removal.

    Kept purely numeric — identifying metadata (task, repo, file) rides
    alongside in the training row, not in the vector.
    """

    node_type: str = "other"
    n_lines: int = 0
    n_bytes: int = 0
    n_covered_lines: int = 0
    coverage_ratio: float = 0.0
    body_coverage_ratio: float = 0.0
    is_fully_uncovered: int = 0
    decorator_kinds: Set[str] = field(default_factory=set)
    n_decorators: int = 0
    is_test_file: int = 0
    is_conftest: int = 0
    is_init_file: int = 0
    depth_in_ast: int = 0
    is_module_top_level: int = 0
    sibling_count: int = 0
    n_child_units: int = 0
    references_in_file: int = 0
    has_side_effect_token: int = 0
    is_dunder: int = 0
    is_private: int = 0

    def to_vector(self) -> List[float]:
        """Flatten to FEATURE_NAMES order."""
        values: List[float] = []

        for t in NODE_TYPES:
            values.append(1.0 if self.node_type == t else 0.0)
        values.append(0.0 if self.node_type in NODE_TYPES else 1.0)

        values.extend([
            float(self.n_lines),
            float(self.n_bytes),
            float(self.n_covered_lines),
            float(self.coverage_ratio),
            float(self.body_coverage_ratio),
            float(self.is_fully_uncovered),
        ])

        for kind in DECORATOR_KINDS:
            values.append(1.0 if kind in self.decorator_kinds else 0.0)

        values.extend([
            float(self.n_decorators),
            float(self.is_test_file),
            float(self.is_conftest),
            float(self.is_init_file),
            float(self.depth_in_ast),
            float(self.is_module_top_level),
            float(self.sibling_count),
            float(self.n_child_units),
            float(self.references_in_file),
            float(self.has_side_effect_token),
            float(self.is_dunder),
            float(self.is_private),
        ])

        assert len(values) == len(FEATURE_NAMES), (
            f"feature width drift: built {len(values)}, "
            f"FEATURE_NAMES has {len(FEATURE_NAMES)}"
        )
        return values

    def as_dict(self) -> dict:
        """Column-name -> value, for JSONL training rows."""
        return dict(zip(FEATURE_NAMES, self.to_vector()))


def extract_features(node: Node,
                     source: str,
                     executed_lines: Set[int],
                     file_path: Optional[Path | str] = None) -> UnitFeatures:
    """Describe `node` as a removal candidate.

    Args:
        node: tree-sitter node for the unit being considered.
        source: full text of the file the node came from.
        executed_lines: 1-indexed lines that ran under coverage. Empty set
            is allowed and simply zeroes the coverage features.
        file_path: path the source came from, used only for the test /
            conftest / __init__ indicators.
    """
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    n_lines = max(1, end_line - start_line + 1)

    covered = _count_covered(node, executed_lines)
    body = _child_of_type(node, "block")
    body_ratio = (
        _covered_ratio(body, executed_lines) if body is not None
        else _covered_ratio(node, executed_lines)
    )

    decorators = _decorator_kinds(node)
    name = _definition_name(node)

    return UnitFeatures(
        node_type=node.type,
        n_lines=n_lines,
        n_bytes=node.end_byte - node.start_byte,
        n_covered_lines=covered,
        coverage_ratio=covered / n_lines,
        body_coverage_ratio=body_ratio,
        is_fully_uncovered=1 if covered == 0 else 0,
        decorator_kinds=decorators,
        n_decorators=_count_decorators(node),
        is_test_file=_is_test_file(file_path),
        is_conftest=_is_named(file_path, "conftest.py"),
        is_init_file=_is_named(file_path, "__init__.py"),
        depth_in_ast=_depth(node),
        is_module_top_level=1 if _is_top_level(node) else 0,
        sibling_count=_sibling_count(node),
        n_child_units=len(node.named_children),
        references_in_file=_count_references(name, source, node),
        has_side_effect_token=_has_side_effect_token(source, node),
        is_dunder=1 if name and name.startswith("__") and name.endswith("__") else 0,
        is_private=1 if name and name.startswith("_") and not name.startswith("__") else 0,
    )


# ------------------------------------------------------------- internals


def _count_covered(node: Node, executed_lines: Set[int]) -> int:
    if not executed_lines:
        return 0
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    return sum(1 for line in range(start, end + 1) if line in executed_lines)


def _covered_ratio(node: Node, executed_lines: Set[int]) -> float:
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    span = max(1, end - start + 1)
    return _count_covered(node, executed_lines) / span


def _child_of_type(node: Node, type_name: str) -> Optional[Node]:
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _decorator_nodes(node: Node) -> List[Node]:
    """Decorators attached to `node`.

    tree-sitter-python wraps a decorated function or class in a
    `decorated_definition` parent holding the `decorator` children, so the
    decorators are never children of the function_definition itself. Handle
    being handed either the wrapper or the inner definition.
    """
    holder: Optional[Node] = None
    if node.type == "decorated_definition":
        holder = node
    elif node.parent is not None and node.parent.type == "decorated_definition":
        holder = node.parent

    if holder is None:
        return []
    return [c for c in holder.children if c.type == "decorator"]


def _count_decorators(node: Node) -> int:
    return len(_decorator_nodes(node))


def _decorator_kinds(node: Node) -> Set[str]:
    kinds: Set[str] = set()
    for dec in _decorator_nodes(node):
        text = dec.text.decode("utf-8", errors="replace") if dec.text else ""
        kinds.add(_classify_decorator(text))
    return kinds


def _classify_decorator(text: str) -> str:
    stripped = text.lstrip("@").strip()
    if "parametrize" in stripped:
        return "pytest_parametrize"
    if "fixture" in stripped:
        return "pytest_fixture"
    if stripped.startswith("pytest.mark") or ".mark." in stripped:
        return "pytest_mark"
    # Match the bare descriptor names, not merely a substring — `property`
    # would otherwise swallow anything containing that word.
    head = stripped.split("(")[0].strip()
    if head in ("classmethod", "staticmethod", "property"):
        return head
    return "other"


def _definition_name(node: Node) -> Optional[str]:
    """Name bound by a def/class, if this node binds one."""
    target = node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                target = child
                break

    if target.type not in ("function_definition", "class_definition"):
        return None

    name_node = target.child_by_field_name("name")
    if name_node is None or name_node.text is None:
        return None
    return name_node.text.decode("utf-8", errors="replace")


def _count_references(name: Optional[str], source: str, node: Node) -> int:
    """Occurrences of `name` elsewhere in the file.

    Excludes the node's own byte range so a definition doesn't count as a
    reference to itself. Purely lexical — good enough as a cheap proxy for
    "is this symbol used", and it costs no import resolution.
    """
    if not name:
        return 0
    outside = source[:node.start_byte] + source[node.end_byte:]
    return len(re.findall(rf"\b{re.escape(name)}\b", outside))


def _has_side_effect_token(source: str, node: Node) -> int:
    text = source[node.start_byte:node.end_byte]
    return 1 if any(tok in text for tok in SIDE_EFFECT_TOKENS) else 0


def _depth(node: Node) -> int:
    depth = 0
    cur = node.parent
    while cur is not None:
        depth += 1
        cur = cur.parent
    return depth


def _is_top_level(node: Node) -> bool:
    parent = node.parent
    if parent is None:
        return False
    if parent.type == "module":
        return True
    # A decorated top-level def has `module` as its grandparent.
    return parent.type == "decorated_definition" and (
        parent.parent is not None and parent.parent.type == "module")


def _sibling_count(node: Node) -> int:
    parent = node.parent
    if parent is None:
        return 0
    return max(0, len(parent.named_children) - 1)


def _is_test_file(file_path: Optional[Path | str]) -> int:
    if file_path is None:
        return 0
    name = Path(file_path).name
    return 1 if (name.startswith("test_") or name.endswith("_test.py")) else 0


def _is_named(file_path: Optional[Path | str], target: str) -> int:
    if file_path is None:
        return 0
    return 1 if Path(file_path).name == target else 0


if __name__ == "__main__":
    import sys as _sys

    _SRC_DIR = Path(__file__).resolve().parent.parent
    if str(_SRC_DIR) not in _sys.path:
        _sys.path.insert(0, str(_SRC_DIR))
    from parser import PythonParser  # noqa: E402

    if len(_sys.argv) < 2:
        print("Usage: python features.py <python_file>")
        raise SystemExit(1)

    path = Path(_sys.argv[1])
    text = path.read_text()
    tree = PythonParser().parse_source(text)

    print(f"{len(FEATURE_NAMES)} features per unit\n")
    for top in tree.root_node.named_children:
        feats = extract_features(top, text, set(), path)
        label = _definition_name(top) or top.type
        print(f"{label:<40} {top.type:<24} "
              f"lines={feats.n_lines:<4} refs={feats.references_in_file:<3} "
              f"decs={sorted(feats.decorator_kinds)}")
