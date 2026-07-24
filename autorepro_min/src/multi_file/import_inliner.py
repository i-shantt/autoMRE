"""
AutoRepro-Min: Multi-File Reduction
Import Inliner (Phase 3 — Selective Inlining)

Given a target file F and the set of files that import it, copy the
top-level definitions that importers actually use into each importer,
rewriting the `from F import ...` statements accordingly.

Scope (kept intentionally narrow so we only inline when it's safe):
  - Handles `from <mod> import name [, name...]` (single-file target).
  - Skips `from <mod> import *`.
  - Skips `import <mod>` / `import <mod> as alias` — the module object
    itself is used, which we can't cleanly inline.
  - Only inlines top-level FunctionDef, AsyncFunctionDef, ClassDef, and
    simple assignments (Assign / AnnAssign) from the target file.
  - Refuses to inline if the target file has any top-level side-effect
    statements (calls, prints, etc.) beyond the whitelisted defs — those
    side-effects would run at import time, and dropping them changes
    behavior.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# AST node types safe to leave at the top level of an inlinable module.
_TOP_LEVEL_ALLOWED = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.ImportFrom,   # forwarded imports the target itself uses
    ast.Import,
    ast.Expr,         # module docstring only (checked below)
)


@dataclass
class InlineResult:
    """Result of an inlining attempt."""
    success: bool
    reason: str = ""
    modified_files: Dict[Path, str] = field(default_factory=dict)
    # importer path -> new file content (after inlining)


class ImportInliner:
    """Inlines a target file's used definitions into its importers."""

    def can_inline(self, target_file: Path) -> Tuple[bool, str]:
        """Cheap up-front check: does the target look safe to inline?"""
        try:
            tree = ast.parse(target_file.read_text())
        except (SyntaxError, OSError, UnicodeDecodeError) as exc:
            return False, f"unparseable target: {exc}"

        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(
                    node.value, ast.Constant) and isinstance(
                    node.value.value, str):
                # Module docstring — fine.
                continue
            if not isinstance(node, _TOP_LEVEL_ALLOWED):
                return False, (
                    f"top-level {type(node).__name__} has side-effects; "
                    "refusing to inline")
        return True, ""

    def inline(self, target_file: Path, importers: List[Path],
               project_dir: Path) -> InlineResult:
        """Inline target_file's used defs into every importer.

        Returns InlineResult with the modified importer contents.  The
        caller is responsible for writing them to disk (typically under a
        snapshot so the mutation can be rolled back).
        """
        ok, reason = self.can_inline(target_file)
        if not ok:
            return InlineResult(success=False, reason=reason)

        target_module = self._module_dotted(target_file, project_dir)
        if target_module is None:
            return InlineResult(success=False,
                                reason="target not under project_dir")

        target_source = target_file.read_text()
        target_tree = ast.parse(target_source)
        top_defs = self._collect_top_level_defs(target_tree, target_source)
        # Forward imports used by the target module itself so inlined defs
        # can still resolve their free names.
        forward_imports = self._collect_forward_imports(target_tree,
                                                        target_source)

        modified: Dict[Path, str] = {}
        for importer in importers:
            try:
                imp_source = importer.read_text()
                imp_tree = ast.parse(imp_source)
            except (SyntaxError, OSError, UnicodeDecodeError) as exc:
                return InlineResult(
                    success=False,
                    reason=f"cannot parse importer {importer}: {exc}")

            new_source = self._rewrite_importer(
                imp_source, imp_tree, target_module, top_defs,
                forward_imports)
            if new_source is None:
                return InlineResult(
                    success=False,
                    reason=(f"importer {importer.name} uses target in a way "
                            "we can't inline (e.g. `import` or `import *`)"))
            modified[importer] = new_source

        return InlineResult(success=True, modified_files=modified)

    # ---------------------------------------------------- helpers

    def _module_dotted(self, path: Path,
                       project_dir: Path) -> Optional[str]:
        try:
            rel = path.resolve().relative_to(project_dir.resolve())
        except ValueError:
            return None
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else None

    def _collect_top_level_defs(self, tree: ast.Module,
                                source: str) -> Dict[str, str]:
        """Map exported name -> source text for its top-level definition."""
        defs: Dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                defs[node.name] = ast.get_source_segment(source, node) or ""
            elif isinstance(node, ast.Assign):
                snippet = ast.get_source_segment(source, node) or ""
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        defs[tgt.id] = snippet
            elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name):
                defs[node.target.id] = ast.get_source_segment(source, node) or ""
        return defs

    def _transitively_close(self, seeds: Set[str],
                            top_defs: Dict[str, str]) -> Set[str]:
        """Expand seed names with peer top-level defs they reference.

        Parses each def's source, walks its AST for Name references, and
        adds any that resolve to another top-level def in the target
        module. Iterates to a fixed point so chains close correctly.
        """
        closed: Set[str] = set(seeds)
        frontier = set(seeds)
        while frontier:
            next_frontier: Set[str] = set()
            for name in frontier:
                snippet = top_defs.get(name, "")
                if not snippet:
                    continue
                try:
                    sub_tree = ast.parse(snippet)
                except SyntaxError:
                    continue
                for node in ast.walk(sub_tree):
                    if isinstance(node, ast.Name) and node.id in top_defs \
                            and node.id not in closed:
                        closed.add(node.id)
                        next_frontier.add(node.id)
            frontier = next_frontier
        return closed

    def _collect_forward_imports(self, tree: ast.Module,
                                 source: str) -> List[str]:
        """Return source lines for the target module's own imports."""
        out: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                snippet = ast.get_source_segment(source, node)
                if snippet:
                    out.append(snippet)
        return out

    def _rewrite_importer(self, source: str, tree: ast.Module,
                          target_module: str, top_defs: Dict[str, str],
                          forward_imports: List[str]) -> Optional[str]:
        """Replace `from target import ...` lines with inlined defs.

        Returns None if the importer uses the target in an un-inlineable
        way (e.g. `import target`, `from target import *`).
        """
        source_lines = source.splitlines(keepends=True)
        edits: List[Tuple[int, int, str]] = []  # (start, end, replacement)
        inlined_names: Set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == target_module or \
                       alias.name.startswith(target_module + "."):
                        return None  # can't inline plain `import mod`
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module != target_module:
                    continue
                if any(a.name == "*" for a in node.names):
                    return None
                for a in node.names:
                    if a.name not in top_defs:
                        # Importing a symbol we didn't collect (probably a
                        # side-effect binding). Refuse the whole inlining.
                        return None
                    inlined_names.add(a.name)
                start = node.lineno - 1
                end = (node.end_lineno or node.lineno)
                edits.append((start, end, ""))

        if not inlined_names:
            # Nothing to do for this importer — nothing to change.
            return source

        # Also inline any top-level defs the inlined names transitively
        # reference within the same target module. Without this, chained
        # inlines break: e.g. inlining `call_b` from b.py that internally
        # calls a peer `call_c` would leave `call_c` undefined.
        inlined_names = self._transitively_close(inlined_names, top_defs)

        # Build the inlined block: forward imports, then the definitions
        # (deterministic order to keep output stable).
        block_parts: List[str] = []
        if forward_imports:
            block_parts.extend(forward_imports)
        for name in sorted(inlined_names):
            block_parts.append(top_defs[name])
        inlined_block = "\n\n".join(block_parts).rstrip() + "\n\n"

        # Apply edits from bottom to top so line indices stay valid.
        # We drop the import lines entirely and prepend the inlined block
        # at the position of the first removed import.
        edits.sort(key=lambda e: e[0])
        first_start = edits[0][0]

        remaining: List[str] = []
        skip_ranges = [(s, e) for s, e, _ in edits]
        for i, line in enumerate(source_lines):
            if any(s <= i < e for s, e in skip_ranges):
                continue
            remaining.append(line)

        # Splice inlined_block in at the position formerly held by the
        # first removed import.
        skipped_before_first = sum(
            1 for i in range(first_start)
            if any(s <= i < e for s, e in skip_ranges))
        insert_at = first_start - skipped_before_first
        remaining.insert(insert_at, inlined_block)
        return "".join(remaining)
