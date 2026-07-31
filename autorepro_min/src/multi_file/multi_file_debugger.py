"""
AutoRepro-Min: Multi-File Reduction
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
from typing import Dict, List, Optional, Set, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from reducer import HybridDeltaDebugger

from .coverage_pruner import CoveragePruner, remap_executed_lines
from .dependency_analyzer import (
    DependencyAnalyzer,
    FileClass,
    ProjectAnalysis,
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
    time_seconds: float = 0.0
    # Lines in files the reducer was never allowed to touch (the target
    # test and its conftest). Reported separately because they dominate
    # the total and drag the headline figure toward zero for reasons that
    # say nothing about the reducer: on requests-guess_json_utf they are
    # 1016 of the 1198 surviving lines, so a total-line figure of 89.3%
    # conceals a 98.2% reduction of the code actually under reduction.
    protected_line_count: int = 0
    # Oracle bookkeeping — zero when it isn't in use.
    oracle_enabled: bool = False
    oracle_skipped_attempts: int = 0   # Phase 4b removals never tried
    oracle_held_back_files: int = 0    # Phase 4a prunes it emptied out
    # Parallel queries whose answers were thrown away because an earlier
    # candidate in the batch went the other way. Real subprocess work,
    # kept out of total_queries so that figure stays comparable with a
    # sequential run.
    speculative_discarded: int = 0

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

    def __init__(self, verbose: bool = False, timeout: int = 60,
                 match_strategy: str = "output_match",
                 aggressive_inline: bool = False,
                 use_coverage_prune: bool = True,
                 use_learned_oracle: bool = False,
                 oracle_model_path: Optional[Path] = None,
                 python: Optional[str] = None,
                 jobs: int = 1):
        self.verbose = verbose
        self.timeout = timeout
        self.match_strategy = match_strategy
        self.use_coverage_prune = use_coverage_prune
        self.python = python
        # >1 asks Phase 4b's candidates in parallel. Required to produce
        # the identical result, so it is a runtime knob and nothing else
        # -- see reducer._speculative_pass and tests/test_speculation.py.
        self.jobs = max(1, int(jobs))
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

        # ------------- Phase 1
        self._log("Phase 1: analyzing project...")
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

        protected = self._protected_files(project_dir, test_command)
        if protected:
            self._log(f"  protecting {len(protected)} oracle file(s): "
                      + ", ".join(sorted(
                          str(p.relative_to(project_dir)) for p in protected)))

        original_file_count = len(analysis.all_files)
        original_line_count = sum(
            self._line_count(f) for f in analysis.all_files)
        queries = 0

        # ------------- Phase 2a — delete unreachable files (batch, with
        # safety net). If the batch delete breaks the bug (dynamic imports,
        # miscounted trace, etc.), restore and fall back to per-file
        # validated deletion.
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
        self._log("Phase 4: intra-file reduction...")
        per_file_reduction: Dict[Path, tuple] = {}
        speculative_discarded = 0
        oracle_skipped_attempts = 0
        oracle_held_back_files = 0
        final_survivors = [f for f in analysis.all_files if f.exists()]
        # Prioritize larger files first so big wins land early.
        final_survivors.sort(key=self._largest_first)

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

        speculator = self._start_speculator(project_dir, test_command,
                                            validator)
        try:
            for f in final_survivors:
                if f.resolve() in protected:
                    self._log(f"  {f.relative_to(project_dir)}: oracle file, "
                              "left intact")
                    continue
                original_source = f.read_text()
                original_lines = len(original_source.splitlines())
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

                    if prune.any_removed:
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
                # Phase 4a may have just rewritten this file, and earlier
                # files are already reduced; bring the worker copies level
                # before they answer anything about this one.
                if speculator is not None:
                    speculator.sync()
                per_file_validator = ProjectFileValidator(validator, f)
                reducer = HybridDeltaDebugger(validator=per_file_validator,
                                              verbose=self.verbose,
                                              oracle=self.oracle,
                                              speculator=speculator)
                try:
                    result = reducer.reduce(source_code=current_source,
                                            test_command=None,
                                            cwd=project_dir,
                                            file_path=f,
                                            executed_lines=file_executed)
                except Exception as exc:
                    self._log(f"  {f.relative_to(project_dir)}: reduction "
                              f"errored ({exc}); leaving as-is")
                    continue

                queries += result.stats.queries
                oracle_skipped_attempts += result.stats.oracle_skipped
                f.write_text(result.minimized_code)
                final_lines = len(result.minimized_code.splitlines())
                per_file_reduction[f] = (original_lines, final_lines)
                speculative_discarded += result.stats.speculative_discarded
                self._log(f"  {f.relative_to(project_dir)}: "
                          f"{original_lines} -> {final_lines} lines")

        finally:
            if speculator is not None:
                speculator.close()

        # ------------- Phase 5 — sweep for files that only became
        # deletable once Phase 4 removed whatever still imported them.
        # Repeat: deleting a module can orphan the one that imported it,
        # so a sweep that removes ten files often exposes more. Each
        # extra round costs one query per surviving file, and the tree is
        # small by now — the round that finds nothing is a handful of
        # queries.
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
            time_seconds=time.time() - start,
            protected_line_count=sum(self._line_count(f) for f in protected
                                     if f.exists()),
            oracle_enabled=self.oracle is not None,
            oracle_skipped_attempts=oracle_skipped_attempts,
            oracle_held_back_files=oracle_held_back_files,
            speculative_discarded=speculative_discarded,
        )

    def _start_speculator(self, project_dir: Path, test_command: List[str],
                          validator) -> Optional[object]:
        """A started ParallelOracle, or None to stay sequential.

        Built here rather than at the top of the run for two reasons: it
        is the only place that uses it, and by now Phases 2 and 3 have
        deleted whatever they could, so the copies each worker takes are
        of the smaller tree.

        Any refusal degrades to sequential rather than raising. Being
        slow is a cost; being wrong is not something to trade for speed.
        """
        if self.jobs < 2:
            return None
        try:
            from .parallel_oracle import ParallelOracle
        except ImportError as exc:
            self._log(f"  parallel oracle unavailable ({exc}); "
                      "continuing sequentially")
            return None

        speculator = ParallelOracle(project_dir, test_command,
                                    oracle=validator._oracle,
                                    timeout=self.timeout, jobs=self.jobs)
        if not speculator.start():
            self._log(f"  parallel oracle declined ({speculator.reason}); "
                      "continuing sequentially")
            return None
        self._log(f"  parallel oracle: {self.jobs} verified worker copies")
        return speculator

    # ---------------------------------------------------------- utils

    @staticmethod
    def _is_test_runner(test_command: List[str]) -> bool:
        """Does this command hand the verdict to a test framework?"""
        runners = {"pytest", "py.test", "unittest", "nose2"}
        for arg in test_command:
            if Path(arg).name in runners or arg in runners:
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
        any `::node::id` suffix) and every conftest.py from that file's
        directory up to the project root, since pytest loads those
        implicitly and they supply the fixtures.
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
            if not candidate.endswith(".py"):
                continue
            path = Path(candidate)
            if not path.is_absolute():
                path = project_dir / path
            path = path.resolve()
            if not path.exists():
                continue
            protected.add(path)

            # conftest.py files apply to everything at or below their
            # directory, so walk up to the project root collecting them.
            directory = path.parent
            while True:
                for name in self.ORACLE_FILENAMES:
                    sibling = (directory / name).resolve()
                    if sibling.exists():
                        protected.add(sibling)
                if directory == project_dir or project_dir not in directory.parents:
                    break
                directory = directory.parent

        return protected

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
            return len(path.read_text().splitlines())
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
