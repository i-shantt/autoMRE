"""
AutoRepro-Min: Benchmark Dataset

Small curated set of multi-file Python bug projects used to compare
prioritizers. Each entry is a directory under
`autorepro_min/examples/multi_file/` that reproduces a specific error
when its reproduction command runs.

This is intentionally not BugsInPy today. BugsInPy needs per-bug Python
version management, dependency installs, and test-framework
orchestration — a big rabbit hole. The harness in `bugsinpy_runner.py`
supports pointing at BugsInPy checkouts if the user has done that
setup; the dataset below is the built-in "always works" benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


_EXAMPLES = (Path(__file__).resolve().parent.parent /
             "autorepro_min" / "examples" / "multi_file")


@dataclass(frozen=True)
class BenchmarkBug:
    bug_id: str
    project_dir: Path
    reproduction_command: List[str]
    expected_error_type: str
    description: str


BUILT_IN_BUGS: List[BenchmarkBug] = [
    BenchmarkBug(
        bug_id="project1_cross_file_type_error",
        project_dir=_EXAMPLES / "project1_cross_file_type_error",
        reproduction_command=["python3", "main.py"],
        expected_error_type="TypeError",
        description=(
            "main.py imports utils.py, calls len() on an int. Plus "
            "unused_a.py and unused_b.py that are never imported."),
    ),
    BenchmarkBug(
        bug_id="project2_dep_chain",
        project_dir=_EXAMPLES / "project2_dep_chain",
        reproduction_command=["python3", "main.py"],
        expected_error_type="TypeError",
        description=(
            "Four-deep import chain main -> a -> b -> c, TypeError in c. "
            "Plus one unused side.py."),
    ),
    BenchmarkBug(
        bug_id="project3_side_effects",
        project_dir=_EXAMPLES / "project3_side_effects",
        reproduction_command=["python3", "main.py"],
        expected_error_type="ZeroDivisionError",
        description=(
            "config.py sets an env var that bug.py reads; main.py "
            "imports both. Tests that config.py's side effects are "
            "preserved (not inlined away)."),
    ),
]


def load_dataset() -> List[BenchmarkBug]:
    """Return the built-in benchmark dataset, skipping any missing."""
    return [b for b in BUILT_IN_BUGS if b.project_dir.exists()]
