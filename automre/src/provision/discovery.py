"""Find the tests a project offers, without running it.

The counterpart to `environment.provision`: an environment is only useful
once you know which command to run in it. SWE-Hub calls this the Test
Agent — entrypoint discovery — and it is the second half of what stops
autoMRE from being pointed at a repository it has never seen.

Read out of the source rather than by running pytest, for two reasons.
Collection on a project whose dependencies are not yet installed usually
errors, so the answer would be an error message rather than a list; and
this has to answer in the time of an upload rather than the time of a
test run.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
              ".tox", "build", "dist", ".eggs"}


def discover(root: Path, limit: int = 40) -> List[str]:
    """pytest node ids worth suggesting, cheapest signal first.

    Returns ids of the form `path/to/test_x.py::test_name` and
    `path/to/test_x.py::TestClass::test_name`, in path order, capped at
    `limit` because the caller is a person choosing one.
    """
    found: List[str] = []
    for node_id in _walk(root):
        found.append(node_id)
        if len(found) >= limit:
            break
    return found


def locate(root: Path, test_name: str) -> List[str]:
    """Every node id in the tree whose test function is `test_name`.

    Some task sources name the function and nothing else — 19.6% of
    SWE-bench Verified, all of it sympy. `pytest -k <name>` finds it, but
    only by collecting the entire suite, and the run then reports counts
    for eleven thousand unrelated tests. Two sympy instances were refused
    for exactly that: the warning tally moved between consecutive runs
    (1429, then 1431) with nothing about the target test changing, so the
    command was not repeatable and the oracle could not use it.

    Resolving the name to a real node id fixes the cause rather than the
    symptom, and collects one file instead of the suite.
    """
    return [node_id for node_id in _walk(root)
            if node_id.rsplit("::", 1)[-1] == test_name]


def _walk(root: Path):
    """Every test node id in the tree, in path order."""
    root = Path(root)
    for path in sorted(root.rglob("test_*.py")) + sorted(root.rglob("*_test.py")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name.startswith("test"):
                yield f"{rel}::{node.name}"
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and sub.name.startswith("test"):
                        yield f"{rel}::{node.name}::{sub.name}"


def node_id_exists(root: Path, node_id: str) -> bool:
    """Whether a pytest node id names something this project actually has.

    A named test that does not exist is the failure mode that made
    `requests-cookie_utils` score a perfect execution fidelity while
    running nothing: pytest exits 4, the harness adopts *that* as the
    behavior to preserve, and the reducer is rewarded for deleting
    everything not needed to keep exiting 4.

    Checked structurally so it costs nothing and works before the
    environment exists. A node id whose file is present but whose test is
    generated at runtime (parametrised ids, fixtures that build classes)
    reads as present, which is the right way to be wrong here: the
    readiness gate runs the command for real straight afterwards.
    """
    file_part, _, rest = node_id.partition("::")
    path = root / file_part
    if not path.is_file():
        return False
    if not rest:
        return True

    wanted = rest.split("::")[-1].split("[")[0]
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return True   # unreadable is the gate's problem, not ours
    return any(getattr(node, "name", None) == wanted
               for node in ast.walk(tree))
