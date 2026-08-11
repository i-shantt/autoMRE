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
    root = Path(root)
    found: List[str] = []

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
                found.append(f"{rel}::{node.name}")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and sub.name.startswith("test"):
                        found.append(f"{rel}::{node.name}::{sub.name}")
            if len(found) >= limit:
                return found
    return found


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
