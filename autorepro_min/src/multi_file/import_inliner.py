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
    """Inlines a target file's used definitions into its importers.

    Args:
        aggressive: when True, skip the "target has side-effect statements"
            refusal. In aggressive mode we attempt the inline anyway and
            let the whole-project validator roll back if the reduction
            actually breaks the bug. Trades safety for coverage — useful
            for benchmarks like Gistify that reward single-file output
            (Self-Containment metric).
    """

    def __init__(self, aggressive: bool = False):
        self.aggressive = aggressive

    def can_inline(self, target_file: Path) -> Tuple[bool, str]:
        """Cheap up-front check: does the target look safe to inline?

        In aggressive mode this always returns True for parseable targets
        — the caller (MultiFileDebugger) will verify empirically via the
        whole-project validator.
        """
        try:
            tree = ast.parse(target_file.read_text())
        except (SyntaxError, OSError, UnicodeDecodeError) as exc:
            return False, f"unparseable target: {exc}"

        if self.aggressive:
            return True, ""

        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(
                    node.value, ast.Constant) and isinstance(
                    node.value.value, str):
                # Module docstring — fine.
                continue
            if not isinstance(node, _TOP_LEVEL_ALLOWED):
                return False, (
                    f"top-level {type(node).__name__} has side-effects; "
                    "refusing to inline (use aggressive=True to try anyway)")
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
                forward_imports, target_full_source=target_source)
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

    def _uses_attribute(self, tree: ast.Module, name: str) -> bool:
        """Does the importer access `name.something` anywhere?"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and \
                    isinstance(node.value, ast.Name) and \
                    node.value.id == name:
                return True
        return False

    def _strip_main_guard(self, source: str) -> str:
        """Remove `if __name__ == '__main__': ...` blocks from a source
        we're about to inline as a module preamble — that guard only
        fires when the file runs as a script, not when it's imported.
        Leaving it in a preamble would cause it to execute in the
        importer's context, which is wrong.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        drops: List[Tuple[int, int]] = []
        for node in tree.body:
            if isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Compare) and
                        isinstance(test.left, ast.Name) and
                        test.left.id == "__name__" and
                        len(test.comparators) == 1 and
                        isinstance(test.comparators[0], ast.Constant) and
                        test.comparators[0].value == "__main__"):
                    drops.append((node.lineno - 1,
                                  node.end_lineno or node.lineno))
        if not drops:
            return source
        lines = source.splitlines(keepends=True)
        keep = [ln for i, ln in enumerate(lines)
                if not any(s <= i < e for s, e in drops)]
        return "".join(keep)

    def _rewrite_importer(self, source: str, tree: ast.Module,
                          target_module: str, top_defs: Dict[str, str],
                          forward_imports: List[str],
                          target_full_source: str = "") -> Optional[str]:
        """Replace `from target import ...` lines with inlined defs.

        In aggressive mode, un-inlineable patterns (`import target`,
        `from target import *`, unknown-name imports) fall back to
        prepending the ENTIRE target module source to the importer and
        stripping the import lines. This preserves module-level side
        effects at the cost of any naming conflicts.

        Returns None (aggressive: only if unrecoverable, e.g. `target.X`
        attribute access we can't rewrite; non-aggressive: on any of the
        above patterns).
        """
        source_lines = source.splitlines(keepends=True)
        edits: List[Tuple[int, int, str]] = []  # (start, end, replacement)
        inlined_names: Set[str] = set()
        fallback_dump = False        # aggressive: dump full target source

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == target_module or \
                       alias.name.startswith(target_module + "."):
                        if not self.aggressive:
                            return None  # can't inline plain `import mod`
                        # Aggressive: check the importer doesn't call
                        # target.X anywhere. If it does, we'd need a
                        # rewriter — skip. If it doesn't (side-effect
                        # import), dumping the target source in place
                        # preserves semantics.
                        bound_name = alias.asname or alias.name.split(".")[0]
                        if self._uses_attribute(tree, bound_name):
                            return None
                        fallback_dump = True
                        start = node.lineno - 1
                        end = (node.end_lineno or node.lineno)
                        edits.append((start, end, ""))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module != target_module:
                    continue
                if any(a.name == "*" for a in node.names):
                    if not self.aggressive:
                        return None
                    fallback_dump = True
                    start = node.lineno - 1
                    end = (node.end_lineno or node.lineno)
                    edits.append((start, end, ""))
                    continue
                for a in node.names:
                    if a.name not in top_defs:
                        if not self.aggressive:
                            # Symbol we didn't collect (side-effect
                            # binding). Refuse the whole inlining.
                            return None
                        # Aggressive: fall back to dumping full source.
                        fallback_dump = True
                    else:
                        inlined_names.add(a.name)
                start = node.lineno - 1
                end = (node.end_lineno or node.lineno)
                edits.append((start, end, ""))

        if not edits:
            # No import of the target found in this importer — no work.
            return source
        if not inlined_names and not fallback_dump:
            # Nothing to do for this importer — nothing to change.
            return source

        # Also inline any top-level defs the inlined names transitively
        # reference within the same target module. Without this, chained
        # inlines break: e.g. inlining `call_b` from b.py that internally
        # calls a peer `call_c` would leave `call_c` undefined.
        inlined_names = self._transitively_close(inlined_names, top_defs)

        # Build the inlined block. In aggressive fallback mode we dump
        # the whole target source (minus its `if __name__ == '__main__'`
        # block, which shouldn't run when the module is imported); in
        # surgical mode we assemble the specific inlined defs.
        if fallback_dump and target_full_source:
            inlined_block = self._strip_main_guard(target_full_source) \
                            .rstrip() + "\n\n"
        else:
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
