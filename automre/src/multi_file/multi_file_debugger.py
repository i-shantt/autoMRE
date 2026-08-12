"""
autoMRE: Multi-File Reduction
Orchestrator (MF-HDD-E)

Wires together the four phases of multi-file reduction:

  Phase 1 — Cross-file analysis           (DependencyAnalyzer)
  Phase 2 — Module-level reduction        (delete unreachable / try-delete
                                          imported-only files)
  Phase 3 — Selective inlining            (ImportInliner)
  Phase 4 — Intra-file HDD-E reduction    (HybridDeltaDebugger driven by
                                          ProjectFileValidator)

The debugger mutates the project directory in place. Callers that want
to preserve the original should copy the project first (the CLI does).
"""

from __future__ import annotations

import shutil
import sys as _sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from reducer import HybridDeltaDebugger, _is_parseable

from .coverage_pruner import CoveragePruner, remap_executed_lines
from .dependency_analyzer import (
    DependencyAnalyzer,
    topological_order,
)
from .import_inliner import ImportInliner
from .multi_file_validator import MultiFileValidator, ProjectFileValidator


@dataclass
class MultiFileReductionResult:
    """Summary of a multi-file reduction run."""
    project_dir: Path
    original_file_count: int
    final_file_count: int
    original_line_count: int
    final_line_count: int
    unreachable_deleted: List[Path] = field(default_factory=list)
    imported_deleted: List[Path] = field(default_factory=list)
    inlined_away: List[Path] = field(default_factory=list)
    per_file_reduction: Dict[Path, tuple] = field(default_factory=dict)
    # path -> (original_lines, final_lines)
    total_queries: int = 0
    # Queries killed by the per-query time limit. Reported because a
    # timeout is otherwise indistinguishable from an ordinary "the bug
    # broke" rejection, and on a parser-shaped project a handful of them
    # can dominate the wall clock while looking like nothing at all.
    timed_out_queries: int = 0
    time_seconds: float = 0.0
    # Lines in files the reducer was never allowed to touch (the target
    # test and its conftest). Reported separately because they dominate
    # the total and drag the headline figure toward zero for reasons that
    # say nothing about the reducer: on requests-guess_json_utf they are
    # 1016 of the 1198 surviving lines, so a total-line figure of 89.3%
    # conceals a 98.2% reduction of the code actually under reduction.
    protected_line_count: int = 0
    # .py files left untouched because they are not valid UTF-8, so the
    # reducer cannot cut them without risking their bytes. Reported for
    # the same reason the two counts above are: a run that quietly
    # excluded part of the tree looks exactly like a run that considered
    # all of it and found nothing to remove.
    undecodable_files: List[Path] = field(default_factory=list)
    # .py files reached through a symlink that leaves the project, and so
    # never candidates: the reducer's remit is the directory it was given.
    outside_project_files: List[Path] = field(default_factory=list)
    # Oracle bookkeeping — zero when it isn't in use.
    oracle_enabled: bool = False
    oracle_skipped_attempts: int = 0   # Phase 4b removals never tried
    oracle_held_back_files: int = 0    # Phase 4a prunes it emptied out

    @property
    def reducible_reduction_rate(self) -> float:
        """Reduction over the code the reducer could actually remove.

        Protected files are constant between the original and the result,
        so they belong in neither numerator nor denominator. This is the
        number that reflects what the tool did; line_reduction_rate is
        the number that reflects what the project looks like afterwards.
        Both are worth reporting, for different questions.
        """
        original = self.original_line_count - self.protected_line_count
        final = self.final_line_count - self.protected_line_count
        if original <= 0:
            return 0.0
        return (original - final) / original

    @property
    def file_reduction_rate(self) -> float:
        if self.original_file_count == 0:
            return 0.0
        return (self.original_file_count - self.final_file_count) \
            / self.original_file_count

    @property
    def line_reduction_rate(self) -> float:
        if self.original_line_count == 0:
            return 0.0
        return (self.original_line_count - self.final_line_count) \
            / self.original_line_count


class MultiFileDebugger:
    """Multi-File Hierarchical Delta Debugging with Execution guidance."""

    # Consecutive Phase 3 rollbacks after which inlining is abandoned for
    # the run. Three is enough to distinguish "this file happens to
    # resist inlining" from "inlining cannot work on this project".
    INLINE_FAILURE_LIMIT = 3

    # Files pytest loads implicitly alongside the target test. They are
    # part of the oracle, not the code under reduction.
    ORACLE_FILENAMES = ("conftest.py",)

    # Safety valve on the Phase 5 sweep. It stops on its own as soon as a
    # round deletes nothing; this only bounds a pathological case.
    MAX_SWEEP_ROUNDS = 10

    # Queries a reduction spends, per line of reducible code. Calibrated
    # on the benchmark: requests runs 0.11, flask 0.14-0.17, tomlkit
    # 0.31-0.47, and a 41-line toy project 0.90 because the per-file
    # probes of Phases 2 and 5 are a fixed cost that a small tree cannot
    # amortize. A factor that varies 8x is not something to build a
    # countdown on, which is why it is only ever used as a rough total
    # and why progress is reported against `work_total` below instead.
    QUERIES_PER_REDUCIBLE_LINE = 0.20

    def __init__(self, verbose: bool = False, timeout: int = 60,
                 match_strategy: str = "output_match",
                 aggressive_inline: bool = False,
                 use_coverage_prune: bool = True,
                 use_learned_oracle: bool = False,
                 oracle_model_path: Optional[Path] = None,
                 python: Optional[str] = None,
                 progress: Optional[Callable[[dict], None]] = None):
        # Called with a plain dict at each phase boundary and after each
        # query. Optional and best-effort: a reduction must not depend on
        # anyone watching it. See `_emit`.
        self.progress = progress
        # Current phase, so per-query events can say where they came from
        # without every phase having to pass it along.
        self._phase = "idle"
        self._phase_message = ""
        self.verbose = verbose
        self.timeout = timeout
        self.match_strategy = match_strategy
        self.use_coverage_prune = use_coverage_prune
        self.python = python
        self.analyzer = DependencyAnalyzer(python=python)
        self.inliner = ImportInliner(aggressive=aggressive_inline)
        self.coverage_pruner = CoveragePruner()

        # Optional and best-effort: a missing model, a missing sklearn,
        # or a stale feature layout all degrade to the plain heuristic
        # rather than failing a reduction.
        self.oracle = None
        if use_learned_oracle:
            self.oracle = self._load_oracle(oracle_model_path)
            if self.oracle is None:
                self._log("  learned oracle requested but unavailable — "
                          "continuing with the heuristic")

    @staticmethod
    def _load_oracle(model_path: Optional[Path]):
        try:
            from ml.oracle import (DEFAULT_MODEL_PATH,
                                   LearnedRemovabilityOracle)
        except ImportError:
            return None
        return LearnedRemovabilityOracle.load_if_available(
            model_path or DEFAULT_MODEL_PATH)

    # ---------------------------------------------------------- public

    @staticmethod
    def _purge_bytecode(root: Path) -> None:
        """Drop every `__pycache__` under `root`.

        `oracle_env` stops the reduction's own subprocesses from *writing*
        bytecode, which is what makes the oracle sound. It cannot stop them
        reading bytecode that was already there, and anything that ran the
        project before us — a caller checking the command works, a test
        suite, an editor — leaves some. A `.pyc` is reused whenever the
        source's `(mtime, size)` matches what it recorded, so an inherited
        one plus a same-length rewrite inside the same clock second is the
        original bug again, arriving through the door we did not close.

        Clearing it here means the invariant holds because the reducer
        establishes it, not because callers happened to leave a clean tree.
        """
        for cache_dir in root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)

    def reduce_project(self, project_dir: Path,
                       test_command: List[str]) -> MultiFileReductionResult:
        project_dir = Path(project_dir).resolve()
        start = time.time()

        self._purge_bytecode(project_dir)

        # Settled before Phase 1 spends a trace on it: this depends only
        # on the command and the tree, and there is no point analyzing a
        # project we are about to refuse.
        protected = self._protected_files(project_dir, test_command)
        if self._is_test_runner(test_command) and not protected:
            # A passing test that protects nothing has a degenerate
            # solution: stub the test body to `pass`, stub the fixtures,
            # and delete the library. That is not a risk to warn about —
            # it is the outcome, every time, and it scores as a near-total
            # reduction. Refuse rather than produce it.
            raise ValueError(
                "the reproduction command runs a test framework but names "
                "no test file or directory inside the project, so there is "
                "nothing to protect as the oracle. The reducer would be "
                "free to empty the tests and then delete the code they "
                "cover. Name the test path explicitly, e.g. "
                "\"-m pytest tests/test_bug.py::test_foo\".")

        # ------------- Phase 1
        self._log("Phase 1: analyzing project...")
        self._emit(phase="analyze", queries=0, estimated_queries=0,
                   message="tracing the reproduction command under coverage")
        analysis = self.analyzer.analyze(project_dir, test_command)
        self._log(f"  {len(analysis.all_files)} files: "
                  f"{len(analysis.executed_files)} executed, "
                  f"{len(analysis.imported_only_files)} imported-only, "
                  f"{len(analysis.unreachable_files)} unreachable")

        # Seed the whole-project oracle from the tracer's run to avoid a
        # second full execution.
        trace = analysis.original_trace
        if not trace.output and trace.return_code == 0:
            # The tracer's output can be empty for silent successes; fall
            # back to a plain reproduction run to establish behavior.
            validator = MultiFileValidator(test_command, project_dir,
                                           timeout=self.timeout,
                                           match_strategy=self.match_strategy)
            validator.capture_original()
        else:
            validator = MultiFileValidator(test_command, project_dir,
                                           timeout=self.timeout,
                                           match_strategy=self.match_strategy)
            validator.set_original_from(
                trace.output, 0 if trace.success else trace.return_code)

        if analysis.outside_project_files:
            self._log(
                f"  refusing {len(analysis.outside_project_files)} file(s) "
                "reached by a symlink out of the project; they are not "
                "this reduction's to change")

        if analysis.undecodable_files:
            self._log(
                f"  leaving {len(analysis.undecodable_files)} file(s) alone: "
                "not valid UTF-8, so they cannot be cut safely — "
                + ", ".join(sorted(
                    str(p.relative_to(project_dir))
                    for p in analysis.undecodable_files))[:200])

        if protected:
            self._log(f"  protecting {len(protected)} oracle file(s): "
                      + ", ".join(sorted(
                          str(p.relative_to(project_dir)) for p in protected)))

        original_file_count = len(analysis.all_files)
        original_line_count = sum(
            self._line_count(f) for f in analysis.all_files)
        queries = 0

        # One estimate, made once, from the tree as Phase 1 found it. It
        # is deliberately not revised downward as files disappear: the
        # queries a deleted file would have cost are queries the run no
        # longer has to spend, and hiding that would turn an honest
        # over-estimate into a progress bar that never moves.
        estimated = self._estimate_queries(analysis.all_files, protected)
        estimated_files = len([f for f in analysis.all_files
                               if f.resolve() not in protected])

        def _on_query(v) -> None:
            self._emit(phase=self._phase, queries=v.queries_run,
                       estimated_queries=estimated,
                       timed_out_queries=v.timed_out_queries,
                       message=self._phase_message)

        self._phase = "analyze"
        self._phase_message = "analysing"
        validator.on_query = _on_query
        self._emit(phase="analyzed", queries=0, estimated_queries=estimated,
                   files=original_file_count, lines=original_line_count,
                   reducible_files=estimated_files,
                   protected_files=len(protected),
                   message=(f"{original_file_count} files, "
                            f"{original_line_count} lines; "
                            f"~{estimated} questions to ask"))

        # ------------- Phase 2a — delete unreachable files (batch, with
        # safety net). If the batch delete breaks the bug (dynamic imports,
        # miscounted trace, etc.), restore and fall back to per-file
        # validated deletion.
        self._set_phase("delete-unreachable",
                        "deleting files the reproduction command never ran")
        self._log("Phase 2a: deleting unreachable files...")
        unreachable_deleted: List[Path] = []
        unreachable_snapshot: Dict[Path, str] = {}
        for f in sorted(analysis.unreachable_files):
            if f.resolve() in protected:
                continue
            try:
                unreachable_snapshot[f] = f.read_text()
                f.unlink()
                unreachable_deleted.append(f)
            except OSError:
                unreachable_snapshot.pop(f, None)

        if unreachable_deleted:
            queries += 1
            if not validator.validate():
                self._log("  batch delete broke the bug — falling back to "
                          "per-file validated deletion")
                for f, content in unreachable_snapshot.items():
                    f.write_text(content)
                unreachable_deleted.clear()
                for f in sorted(unreachable_snapshot):
                    content = f.read_text()
                    f.unlink()
                    queries += 1
                    if validator.validate():
                        unreachable_deleted.append(f)
                        self._log(f"  removed {f.relative_to(project_dir)}")
                    else:
                        f.write_text(content)
            else:
                for f in unreachable_deleted:
                    self._log(f"  removed {f.relative_to(project_dir)}")

        # ------------- Phase 2b — try to delete imported-only files
        self._set_phase("probe-imported",
                        "probing files that are imported but never run")
        self._log("Phase 2b: probing imported-only files...")
        imported_deleted: List[Path] = []
        # Order: larger files first — biggest wins if they're truly dead.
        candidates = sorted(analysis.imported_only_files,
                            key=self._largest_first)
        for f in candidates:
            if not f.exists() or f.resolve() in protected:
                continue
            content = f.read_text()
            f.unlink()
            queries += 1
            if validator.validate():
                imported_deleted.append(f)
                self._log(f"  removed {f.relative_to(project_dir)}")
            else:
                f.write_text(content)

        # Recompute surviving files after Phase 2.
        surviving = [f for f in analysis.all_files if f.exists()]

        # ------------- Phase 3 — selective inlining (leaves first)
        self._set_phase("inline", "trying to inline modules into importers")
        self._log("Phase 3: inlining and collapsing files...")
        surviving_set = set(surviving)
        # Restrict dep graph to survivors.
        pruned_graph = {
            f: {d for d in analysis.dep_graph.get(f, set())
                if d in surviving_set}
            for f in surviving}
        order = topological_order(surviving, pruned_graph)
        inlined_away: List[Path] = []

        # topological_order emits dependency-free files first (c in
        # main->a->b->c), which is exactly the order Phase 3 wants: inline
        # leaves upward so each intermediate absorbs its dependencies before
        # being inlined into its own importer.
        # Inlining a module of an installed package is structurally
        # hopeless: folding src/flask/app.py into __init__.py breaks
        # every `from flask.app import X` and the layout the editable
        # install depends on. On flask all twenty attempts roll back, one
        # wasted query each. Rather than hard-code a package rule that
        # might be wrong for some project, give up after a few
        # consecutive failures — a project where inlining works keeps
        # working, and one where it cannot stops paying for the answer.
        consecutive_rollbacks = 0
        for f in order:
            if consecutive_rollbacks >= self.INLINE_FAILURE_LIMIT:
                self._log(f"  giving up on inlining after "
                          f"{consecutive_rollbacks} consecutive rollbacks")
                break
            if not f.exists() or f.resolve() in protected:
                continue
            importers = [
                imp for imp in sorted(analysis.reverse_graph.get(f, set()))
                if imp.exists() and imp != f]
            if not importers:
                continue
            inline_result = self.inliner.inline(f, importers, project_dir)
            if not inline_result.success:
                self._log(f"  skip {f.relative_to(project_dir)}: "
                          f"{inline_result.reason}")
                continue
            with validator.snapshot(list(inline_result.modified_files) + [f]) \
                    as snap:
                for importer, new_source in inline_result.modified_files.items():
                    importer.write_text(new_source)
                f.unlink()
                queries += 1
                if validator.validate():
                    snap.commit()
                    inlined_away.append(f)
                    consecutive_rollbacks = 0
                    self._log(f"  inlined away {f.relative_to(project_dir)}")
                else:
                    consecutive_rollbacks += 1
                    self._log(f"  rollback {f.relative_to(project_dir)} "
                              "(bug broke)")

        # ------------- Phase 4 — per-file HDD-E under project validation
        self._set_phase("reduce-files",
                        "removing definitions and statements, file by file")
        self._log("Phase 4: intra-file reduction...")
        per_file_reduction: Dict[Path, tuple] = {}
        oracle_skipped_attempts = 0
        oracle_held_back_files = 0
        final_survivors = [f for f in analysis.all_files if f.exists()]
        # Prioritize larger files first so big wins land early.
        final_survivors.sort(key=self._largest_first)

        # Progress is reported against lines-to-examine rather than
        # queries, because this is the quantity that is actually known in
        # advance. Phase 4b is ~97% of a run's queries and it walks this
        # list once, so "lines examined / lines to examine" is a real
        # fraction of the work, where a query estimate is a guess whose
        # per-line factor varies eightfold across projects.
        work_total = sum(self._line_count(f) for f in final_survivors
                         if f.resolve() not in protected)
        work_done = 0
        self._emit(phase=self._phase, queries=validator.queries_run,
                   estimated_queries=estimated,
                   work_done=0, work_total=work_total,
                   message=f"{work_total} lines to examine")

        # Phase 1's coverage described the original file layout. After
        # Phase 3 inlining, importers now contain code that never ran
        # in the original trace — so applying Phase 1 line numbers to
        # them would find nothing prunable. Re-trace against the
        # current tree to give the pruner accurate coverage.
        if self.use_coverage_prune:
            self._log("  refreshing coverage after inlining...")
            fresh = self.analyzer.analyze(project_dir, test_command)
            fresh_executed = fresh.executed_lines
            queries += 1  # count the re-trace as one oracle query
        else:
            fresh_executed = {}

        for f in final_survivors:
            if f.resolve() in protected:
                self._log(f"  {f.relative_to(project_dir)}: oracle file, "
                          "left intact")
                continue
            original_source = f.read_text()
            original_lines = len(original_source.splitlines())
            self._phase_message = (
                f"reducing {f.relative_to(project_dir)} "
                f"({original_lines} lines)")
            self._emit(phase=self._phase, queries=validator.queries_run,
                       estimated_queries=estimated,
                       work_done=work_done, work_total=work_total,
                       current_file=str(f.relative_to(project_dir)),
                       message=self._phase_message)
            # Coverage for this file as it currently stands. Phase 4a may
            # shift it below; Phase 4b then consumes whatever survives.
            file_executed: Optional[Set[int]] = (
                set(fresh_executed.get(f, set()))
                if self.use_coverage_prune else None)

            # Phase 4a — coverage-based bulk prune (one query per file
            # to strip everything HDD-E would otherwise discover as
            # cold, one wasted query per unit).
            if self.use_coverage_prune:
                executed = fresh_executed.get(f, set())
                prune = self.coverage_pruner.prune_source(
                    original_source, executed,
                    accept=self._oracle_accept(original_source, executed, f))

                # Distinguish "coverage found nothing" from "the oracle
                # vetoed everything coverage found" — only the second is
                # the oracle changing the outcome.
                if self.oracle is not None and not prune.any_removed:
                    unfiltered = self.coverage_pruner.prune_source(
                        original_source, executed)
                    if unfiltered.any_removed:
                        oracle_held_back_files += 1
                        self._log(
                            f"  {f.relative_to(project_dir)}: oracle held "
                            f"back all {unfiltered.n_removed} coverage "
                            f"candidates — skipping bulk prune")

                if prune.any_removed and not _is_parseable(
                        prune.pruned_source):
                    # A prune that does not parse cannot be accepted by any
                    # oracle, so asking costs a subprocess to learn nothing
                    # — and the rejection is indistinguishable from
                    # "coverage lied", which is how a dangling-decorator
                    # bug hid here for the project's whole history. Check
                    # locally instead, and say which it was.
                    self._log(f"  {f.relative_to(project_dir)}: coverage "
                              "prune produced unparseable source — skipped "
                              "(no query spent)")
                elif prune.any_removed:
                    f.write_text(prune.pruned_source)
                    queries += 1
                    if validator.validate():
                        # The prune stuck, so the file Phase 4b sees is
                        # renumbered. Shift coverage to match before it
                        # gets read against the wrong lines.
                        file_executed = remap_executed_lines(
                            original_source, prune.removed, executed)
                        self._log(
                            f"  {f.relative_to(project_dir)}: "
                            f"coverage-pruned {prune.n_removed} "
                            f"uncovered units "
                            f"({original_lines} -> "
                            f"{prune.pruned_line_count} lines)")
                    else:
                        # Coverage lied — decorator side effect, dynamic
                        # dispatch, metaclass wiring, etc. Roll back and
                        # let HDD-E figure it out the slow way.
                        f.write_text(original_source)
                        self._log(
                            f"  {f.relative_to(project_dir)}: coverage "
                            "prune broke bug — rolled back")

            # Phase 4b — standard intra-file HDD-E on whatever survived.
            current_source = f.read_text()
            per_file_validator = ProjectFileValidator(validator, f)
            reducer = HybridDeltaDebugger(validator=per_file_validator,
                                          verbose=self.verbose,
                                          oracle=self.oracle)
            try:
                result = reducer.reduce(source_code=current_source,
                                        test_command=None,
                                        cwd=project_dir,
                                        file_path=f,
                                        executed_lines=file_executed)
            except Exception as exc:
                # Counted as done either way: this file will not be
                # revisited, so leaving it out would stall the reported
                # progress on a file nothing is working on.
                work_done += original_lines
                self._log(f"  {f.relative_to(project_dir)}: reduction "
                          f"errored ({exc}); leaving as-is")
                continue

            work_done += original_lines
            queries += result.stats.queries
            oracle_skipped_attempts += result.stats.oracle_skipped
            f.write_text(result.minimized_code)
            final_lines = len(result.minimized_code.splitlines())
            per_file_reduction[f] = (original_lines, final_lines)
            self._log(f"  {f.relative_to(project_dir)}: "
                      f"{original_lines} -> {final_lines} lines")

        # ------------- Phase 5 — sweep for files that only became
        # deletable once Phase 4 removed whatever still imported them.
        # Repeat: deleting a module can orphan the one that imported it,
        # so a sweep that removes ten files often exposes more. Each
        # extra round costs one query per surviving file, and the tree is
        # small by now — the round that finds nothing is a handful of
        # queries.
        self._set_phase("final-sweep",
                        "retrying whole-file deletion now imports are gone")
        self._log("Phase 5: final file sweep...")
        for _ in range(self.MAX_SWEEP_ROUNDS):
            swept, sweep_queries = self._sweep_deletable_files(
                [f for f in analysis.all_files
                 if f.exists() and f.resolve() not in protected],
                validator, project_dir)
            queries += sweep_queries
            imported_deleted.extend(swept)
            if not swept:
                break

        # ------------- Summarize
        final_files = [f for f in analysis.all_files if f.exists()]
        final_line_count = sum(self._line_count(f) for f in final_files)

        self._set_phase("done", (
            f"{original_file_count} files / {original_line_count} lines "
            f"→ {len(final_files)} files / {final_line_count} lines "
            f"in {validator.queries_run} questions"))

        return MultiFileReductionResult(
            project_dir=project_dir,
            original_file_count=original_file_count,
            final_file_count=len(final_files),
            original_line_count=original_line_count,
            final_line_count=final_line_count,
            unreachable_deleted=unreachable_deleted,
            imported_deleted=imported_deleted,
            inlined_away=inlined_away,
            per_file_reduction=per_file_reduction,
            total_queries=queries,
            timed_out_queries=validator.timed_out_queries,
            time_seconds=time.time() - start,
            protected_line_count=sum(self._line_count(f) for f in protected
                                     if f.exists()),
            undecodable_files=list(analysis.undecodable_files),
            outside_project_files=list(analysis.outside_project_files),
            oracle_enabled=self.oracle is not None,
            oracle_skipped_attempts=oracle_skipped_attempts,
            oracle_held_back_files=oracle_held_back_files,
        )

    # ---------------------------------------------------------- utils

    # Scripts that are a project's own test runner rather than the code
    # under test. Django's suite runs only through tests/runtests.py, and
    # sympy's through bin/test; neither is pytest, and without these
    # names the command reads as "python some_script.py", where the
    # script is the subject and reducing it is the whole point.
    #
    # The distinction is not cosmetic. `_protected_files` returns nothing
    # for a command that is not a test runner, so a Django reduction
    # protected no oracle at all and the refusal guard never fired. For a
    # failing test that is survivable — the failure output pins the test
    # body — but a *passing* test run this way has the degenerate
    # solution wide open: empty the test, the runner still prints OK, and
    # the library can go.
    # Deliberately narrow. "test.py" and "tests.py" are ordinary module
    # names far more often than they are entry points, and a false
    # positive is not harmless: the named script gets protected as the
    # oracle, so if it was in fact the subject the reduction silently
    # does nothing at all.
    _RUNNER_SCRIPTS = {"runtests.py", "run_tests.py", "runtest.py",
                       "test_all.py", "test"}

    @classmethod
    def _is_test_runner(cls, test_command: List[str]) -> bool:
        """Does this command hand the verdict to a test framework?"""
        runners = {"pytest", "py.test", "unittest", "nose2"}
        for arg in test_command:
            name = Path(arg).name
            if name in runners or arg in runners:
                return True
            # Only as a path: a bare argument "test" is far more likely
            # to be a label or a directory than an invocation.
            if name in cls._RUNNER_SCRIPTS and ("/" in arg or "\\" in arg):
                return True
        return False

    def _protected_files(self, project_dir: Path,
                         test_command: List[str]) -> Set[Path]:
        """Files that define the property being preserved.

        The reproduction command names a test; that test and the fixtures
        it pulls in are the *oracle*, not the code under reduction.
        Letting the reducer touch them is a category error with a very
        comfortable failure mode: stub the test body to `pass` and stub
        every fixture to `pass`, and pytest still prints "1 passed",
        which matches the original output exactly. The reducer is then
        free to delete the entire library.

        That is not hypothetical. On flask it produced a "99.6%
        reduction" whose surviving tree was two files of empty test
        stubs, with `import flask` resolving to nothing. The run proved
        only that an empty test passes.

        Protected: every existing .py path named in the command (minus
        any `::node::id` suffix), every .py file under a directory named
        in the command, and every conftest.py from those files' directories
        up to the project root, since pytest loads those implicitly and
        they supply the fixtures.

        Directories count because `pytest tests/` is an ordinary way to
        name a test suite, and matching only `.py` arguments protected
        nothing for it — which put the whole gut-the-test failure back
        within reach through a different door. `reduce_project` refuses a
        test-runner command that ends up protecting nothing at all.
        """
        protected: Set[Path] = set()
        project_dir = project_dir.resolve()

        # Only test-runner invocations get this treatment. Under
        # `python main.py` the script *is* the subject — it is what
        # crashes, and reducing it is the whole point. That case is also
        # self-protecting: the oracle compares the traceback including
        # the offending source line, so a reduction cannot fake the
        # failure by deleting the code that causes it. A passing pytest
        # run has no such anchor, which is exactly why it needs one.
        if not self._is_test_runner(test_command):
            return protected

        for arg in test_command:
            candidate = arg.split("::", 1)[0]
            path = Path(candidate)
            if not path.is_absolute():
                path = project_dir / path
            try:
                path = path.resolve()
            except OSError:
                continue
            if not path.exists():
                continue
            # Only paths inside the project are ours to protect; an
            # argument like `-p no:cacheprovider` can resolve to something
            # that happens to exist elsewhere.
            if path != project_dir and project_dir not in path.parents:
                continue

            if path.is_dir():
                named = sorted(path.rglob("*.py"))
            elif path.suffix == ".py":
                named = [path]
            else:
                continue
            if not named:
                continue
            protected.update(named)

            # conftest.py files apply to everything at or below their
            # directory, so walk up to the project root collecting them.
            for directory in {f.parent for f in named}:
                while True:
                    for name in self.ORACLE_FILENAMES:
                        sibling = (directory / name).resolve()
                        if sibling.exists():
                            protected.add(sibling)
                    if (directory == project_dir
                            or project_dir not in directory.parents):
                        break
                    directory = directory.parent

        protected.update(self._files_named_by_label(project_dir, test_command))
        return protected

    @staticmethod
    def _files_named_by_label(project_dir: Path,
                              test_command: List[str]) -> Set[Path]:
        """Test files named as a dotted label rather than a path.

        A project with its own runner is driven by module paths:
        `tests/runtests.py apps.tests.AppsTests.test_clear_cache`. Only
        `runtests.py` matches as a path, so the file holding the test —
        the actual oracle — was left reducible while the harness around
        it was protected. This walks the dotted prefix down to whichever
        file exists.

        Labels also carry class and method names, so the search stops at
        the first prefix that is a file and ignores the rest.
        """
        found: Set[Path] = set()
        for arg in test_command:
            if arg.startswith("-") or "/" in arg or "\\" in arg:
                continue
            parts = arg.split(".")
            if len(parts) < 2:
                continue
            # Longest prefix first, down to the first component alone:
            # "test_mylib.TestX.test_add" lives in test_mylib.py, and
            # stopping at two components would never look there.
            for cut in range(len(parts), 0, -1):
                stem = Path(*parts[:cut]).with_suffix(".py")
                for base in (project_dir, project_dir / "tests"):
                    candidate = (base / stem)
                    try:
                        candidate = candidate.resolve()
                    except OSError:
                        continue
                    if (candidate.is_file()
                            and candidate.is_relative_to(project_dir)):
                        found.add(candidate)
                        break
                if found:
                    break
        return found

    def _sweep_deletable_files(self, files: List[Path], validator,
                               project_dir: Path) -> Tuple[List[Path], int]:
        """Retry deleting whole files, largest first.

        Phase 2 asks this question before Phase 4 has run, which is too
        early: a module survives Phase 2 because something still imports
        it, and Phase 4 is what removes that import. By the time the
        import is gone nothing revisits the module, so it sits in the
        output as an orphan no longer reachable from anywhere.

        Returns the files deleted and the queries spent.
        """
        deleted: List[Path] = []
        queries = 0
        for f in sorted(files, key=self._largest_first):
            if not f.exists():
                continue
            content = f.read_text()
            try:
                f.unlink()
            except OSError:
                continue
            queries += 1
            if validator.validate():
                deleted.append(f)
                self._log(f"  removed {f.relative_to(project_dir)}")
            else:
                f.write_text(content)
        return deleted, queries

    def _oracle_accept(self, source: str, executed: Set[int],
                       file_path: Path):
        """Second opinion for Phase 4a candidates, or None if no oracle.

        The pruner bets a whole file's uncovered units on a single query.
        When that bet loses, the query is wasted and HDD-E has to
        rediscover everything one unit at a time. The oracle holds back
        candidates it scores below the safe threshold, so the bet is
        placed only on units both signals agree are dead.

        Predictions are made one node at a time here rather than batched:
        the pruner discovers candidates during its walk, and the walk's
        shape depends on the answers. At ~21us per call against a
        validation query in the hundreds of milliseconds, that is a
        trade worth making.
        """
        if self.oracle is None:
            return None

        try:
            from ml.features import extract_features
            from ml.oracle import PHASE_4A_SAFE_THRESHOLD
        except ImportError:
            return None

        def accept(node) -> bool:
            try:
                feats = extract_features(node, source, executed, file_path)
                return self.oracle.predict(feats) >= PHASE_4A_SAFE_THRESHOLD
            except Exception:
                # Scoring failed for this node — fall back to trusting
                # coverage, which is the pre-oracle behavior.
                return True

        return accept

    def _line_count(self, path: Path) -> int:
        try:
            # errors="replace" only ever substitutes non-ASCII bytes, and
            # a line count depends on newlines, so the count stays exact.
            # Phase 1 already excludes files that do not decode; this is
            # here because counting lines is the first thing done to a
            # tree and should not be the thing that crashes on it.
            return len(path.read_text(errors="replace").splitlines())
        except OSError:
            return 0

    def _largest_first(self, path: Path) -> tuple:
        """Total-order sort key: biggest file first, path breaking ties.

        Size alone is not a total order — files sharing a line count are
        common, and their relative order then falls out of whatever the
        caller iterated. Where that caller is a `Set[Path]`, the order
        follows Path's string hash, which PYTHONHASHSEED randomizes per
        process: the same tree reduced twice takes different paths and
        spends a different number of queries. The path tiebreaker makes
        every size-ordered pass reproducible.
        """
        return (-self._line_count(path), path.as_posix())

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ------------------------------------------------------ progress

    def _emit(self, **fields) -> None:
        """Report progress, if anyone asked to hear it.

        Swallows observer failures for the same reason `on_query` does: a
        reduction that fails because a progress bar raised would be a
        reduction ruined by its own instrumentation.
        """
        if self.progress is None:
            return
        try:
            self.progress(dict(fields))
        except Exception:
            pass

    def _set_phase(self, phase: str, message: str) -> None:
        self._phase = phase
        self._phase_message = message
        self._emit(phase=phase, message=message)

    def _estimate_queries(self, files: List[Path],
                          protected: Set[Path]) -> int:
        """Rough total-query estimate, for an ETA and nothing else.

        Reducible lines are the denominator that matters: protected files
        are never asked about, so counting them would inflate every
        estimate by the size of the test suite. The per-line factor is
        calibrated on the benchmark and is only good to a factor of two —
        see QUERIES_PER_REDUCIBLE_LINE. Callers should present a range.
        """
        reducible = sum(self._line_count(f) for f in files
                        if f.exists() and f.resolve() not in protected)
        return max(1, int(reducible * self.QUERIES_PER_REDUCIBLE_LINE))
