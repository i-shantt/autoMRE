"""
AutoRepro-Min: Multi-File Reduction
Dependency Analyzer (Phase 1 — Cross-File Reconnaissance)

Responsibilities:
  1. Discover: enumerate all .py files under a project directory.
  2. Trace:    run the reproduction command with coverage.py to capture
               every executed line, per file.
  3. Graph:    parse each file's imports to build a project-local
               dependency graph (file -> files it imports).
  4. Classify: label every file as Executed, Imported-only, or
               Unreachable.
"""

from __future__ import annotations

import ast
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import sys as _sys
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from tracer import ExecutionTrace, ExecutionTracer


class FileClass(enum.Enum):
    """Classification of a project file after coverage tracing."""
    EXECUTED = "executed"        # at least one line ran
    IMPORTED_ONLY = "imported"   # imported by another file, no lines ran
    UNREACHABLE = "unreachable"  # not imported, never ran


@dataclass
class ProjectAnalysis:
    """Result of Phase 1 analysis for a project."""
    project_dir: Path
    all_files: List[Path]
    executed_lines: Dict[Path, Set[int]]
    dep_graph: Dict[Path, Set[Path]]         # file -> files it imports
    reverse_graph: Dict[Path, Set[Path]]     # file -> files importing it
    classification: Dict[Path, FileClass]
    original_trace: ExecutionTrace

    @property
    def executed_files(self) -> Set[Path]:
        return {f for f, c in self.classification.items()
                if c is FileClass.EXECUTED}

    @property
    def imported_only_files(self) -> Set[Path]:
        return {f for f, c in self.classification.items()
                if c is FileClass.IMPORTED_ONLY}

    @property
    def unreachable_files(self) -> Set[Path]:
        return {f for f, c in self.classification.items()
                if c is FileClass.UNREACHABLE}


class DependencyAnalyzer:
    """Discovers files, traces execution, and builds a dep graph."""

    def __init__(self, tracer: Optional[ExecutionTracer] = None,
                 exclude_dirs: Optional[Set[str]] = None):
        self.tracer = tracer or ExecutionTracer()
        # Common directories we should never treat as source.
        self.exclude_dirs = exclude_dirs or {
            "__pycache__", ".git", ".hg", ".svn", ".venv", "venv",
            "env", "node_modules", ".pytest_cache", ".mypy_cache",
            ".tox", "dist", "build", ".eggs",
        }

    # ------------------------------------------------------------ Phase 1

    def analyze(self, project_dir: Path,
                test_command: Optional[List[str]] = None) -> ProjectAnalysis:
        """Run the full Phase-1 analysis and return a ProjectAnalysis."""
        project_dir = Path(project_dir).resolve()

        all_files = self._discover(project_dir)
        trace = self._trace(project_dir, test_command)
        executed_lines = self._localize_trace(trace, project_dir, all_files)
        dep_graph = self._build_dep_graph(project_dir, all_files)
        reverse_graph = self._reverse(dep_graph, all_files)
        classification = self._classify(all_files, executed_lines, reverse_graph)

        return ProjectAnalysis(
            project_dir=project_dir,
            all_files=all_files,
            executed_lines=executed_lines,
            dep_graph=dep_graph,
            reverse_graph=reverse_graph,
            classification=classification,
            original_trace=trace,
        )

    # ---------------------------------------------------- 1a) discovery

    def _discover(self, project_dir: Path) -> List[Path]:
        found: List[Path] = []
        for path in project_dir.rglob("*.py"):
            if any(part in self.exclude_dirs for part in path.parts):
                continue
            found.append(path.resolve())
        return sorted(found)

    # ---------------------------------------------------- 1b) tracing

    def _trace(self, project_dir: Path,
               test_command: Optional[List[str]]) -> ExecutionTrace:
        command = list(test_command or ["main.py"])
        # `coverage run` IS the Python interpreter — passing `python3 foo.py`
        # makes coverage try to execute a script literally named "python3".
        # Strip a leading interpreter name/path if present.
        if command:
            head = Path(command[0]).name.lower()
            interp_names = {"python", "python3", "python2",
                            Path(_sys.executable).name.lower()}
            if head in interp_names or head.startswith("python"):
                command = command[1:]
        if not command:
            command = ["main.py"]
        return self.tracer.trace_command(command, cwd=project_dir)

    def _localize_trace(self, trace: ExecutionTrace, project_dir: Path,
                        all_files: List[Path]) -> Dict[Path, Set[int]]:
        """Restrict coverage results to files inside the project."""
        project_files = {f.resolve() for f in all_files}
        localized: Dict[Path, Set[int]] = {}
        for raw_path, lines in trace.executed_lines.items():
            resolved = Path(raw_path).resolve()
            if resolved in project_files and lines:
                localized[resolved] = set(lines)
        return localized

    # ---------------------------------------------------- 1c) dep graph

    def _build_dep_graph(self, project_dir: Path,
                         all_files: List[Path]) -> Dict[Path, Set[Path]]:
        module_index = self._index_modules(project_dir, all_files)
        graph: Dict[Path, Set[Path]] = {f: set() for f in all_files}

        for file_path in all_files:
            try:
                source = file_path.read_text()
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

            for module_name in self._imports_from(tree, file_path, project_dir):
                target = self._resolve_module(module_name, module_index)
                if target and target != file_path:
                    graph[file_path].add(target)
        return graph

    def _index_modules(self, project_dir: Path,
                       all_files: List[Path]) -> Dict[str, Path]:
        """Map dotted module names ('pkg.sub.mod') -> file path."""
        index: Dict[str, Path] = {}
        for file_path in all_files:
            rel = file_path.relative_to(project_dir)
            parts = list(rel.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            dotted = ".".join(parts)
            index[dotted] = file_path
        return index

    def _imports_from(self, tree: ast.AST, file_path: Path,
                      project_dir: Path) -> List[str]:
        """Yield dotted module names referenced by import statements."""
        names: List[str] = []
        rel_parts = list(file_path.relative_to(project_dir)
                         .with_suffix("").parts)
        # For a package __init__.py the package equals the containing dir.
        pkg_parts = (rel_parts[:-1]
                     if rel_parts and rel_parts[-1] != "__init__"
                     else rel_parts[:-1])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    # Relative import: walk up `level` packages.
                    if node.level > len(pkg_parts):
                        continue
                    prefix_parts = pkg_parts[: len(pkg_parts) - node.level + 1]
                    dotted = ".".join(prefix_parts + ([base] if base else []))
                else:
                    dotted = base
                if not dotted:
                    continue
                names.append(dotted)
                # `from pkg import mod` may reference pkg/mod.py directly.
                for alias in node.names:
                    if alias.name != "*":
                        names.append(f"{dotted}.{alias.name}")
        return names

    def _resolve_module(self, dotted: str,
                        module_index: Dict[str, Path]) -> Optional[Path]:
        if dotted in module_index:
            return module_index[dotted]
        # Progressively strip trailing components — `pkg.mod.symbol` may
        # actually name a symbol inside module `pkg.mod`.
        parts = dotted.split(".")
        while len(parts) > 1:
            parts.pop()
            candidate = ".".join(parts)
            if candidate in module_index:
                return module_index[candidate]
        return None

    def _reverse(self, graph: Dict[Path, Set[Path]],
                 all_files: List[Path]) -> Dict[Path, Set[Path]]:
        rev: Dict[Path, Set[Path]] = {f: set() for f in all_files}
        for src, targets in graph.items():
            for tgt in targets:
                rev.setdefault(tgt, set()).add(src)
        return rev

    # ---------------------------------------------------- 1d) classify

    def _classify(self, all_files: List[Path],
                  executed_lines: Dict[Path, Set[int]],
                  reverse_graph: Dict[Path, Set[Path]]
                  ) -> Dict[Path, FileClass]:
        result: Dict[Path, FileClass] = {}
        for f in all_files:
            if executed_lines.get(f):
                result[f] = FileClass.EXECUTED
            elif reverse_graph.get(f):
                result[f] = FileClass.IMPORTED_ONLY
            else:
                result[f] = FileClass.UNREACHABLE
        return result


def topological_order(files: List[Path],
                      dep_graph: Dict[Path, Set[Path]]) -> List[Path]:
    """Return files in dependency order (leaves last).

    Ties and cycles are broken by path sort — sufficient for our use since
    the caller reverses this to process leaves first during inlining.
    """
    remaining = {f: {d for d in dep_graph.get(f, set()) if d in set(files)}
                 for f in files}
    ordered: List[Path] = []
    while remaining:
        ready = sorted(f for f, deps in remaining.items() if not deps)
        if not ready:
            # Cycle: emit whatever's left in a stable order and bail.
            ordered.extend(sorted(remaining.keys()))
            break
        for f in ready:
            ordered.append(f)
            remaining.pop(f)
            for deps in remaining.values():
                deps.discard(f)
    return ordered
