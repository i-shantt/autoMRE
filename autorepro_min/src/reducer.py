"""
AutoRepro-Min: Automated Bug Reproduction Minimization
Reducer Module

This module implements the Hybrid Delta Debugging algorithm (HDD-E):
Hierarchical Delta Debugging with Execution-guided reduction.

Based on:
- Zeller & Hildebrandt (2002): ddmin algorithm
- Misherghi & Su (2006): Hierarchical Delta Debugging
- Zhou et al. (2024): Weighted Delta Debugging
- Vince et al. (2021): Hoisting extensions
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from parser import (
    byte_to_char_offsets,
    CodeUnit,
    format_code_without_units,
    PythonParser,
    remap_executed_lines,
    to_char_offset,
)
from tracer import ExecutionTracer, ExecutionTrace
from validator import Validator


def _overlaps(span: Tuple[int, int],
              spans: List[Tuple[int, int]]) -> bool:
    """True if `span` intersects anything already scheduled for removal."""
    start, end = span
    for s, e in spans:
        if start < e and s < end:
            return True
    return False


def _apply_removals(source: str, spans: List[Tuple[int, int]]) -> str:
    """Delete byte ranges, back to front so offsets stay valid."""
    out = source
    for start, end in sorted(spans, key=lambda r: -r[0]):
        out = out[:start] + out[end:]
    return out


def _apply_edits(source: str,
                 edits: List[Tuple[Tuple[int, int], str]]) -> str:
    """Replace byte ranges with text, back to front."""
    out = source
    for (start, end), replacement in sorted(edits, key=lambda e: -e[0][0]):
        out = out[:start] + replacement + out[end:]
    return out


def _largest_removable_subset(spans, ok, min_chunk: int = 1, _depth: int = 0):
    """Biggest subset of `spans` that can be removed together.

    Halving, in the spirit of ddmin: ask about the whole set first, and
    only split when the answer is no. When a file carries three hundred
    comments and exactly one `# noqa` that matters, this isolates it in
    O(log n) probes instead of asking about each comment in turn.

    `min_chunk` stops the descent: a group at or below it whose whole
    fails returns nothing instead of subdividing. That matters because
    the cost here is driven by removable *density*, not by n. When
    nothing in the set can go, the recursion visits every node of a
    binary tree over the spans — about 2n probes, where asking per span
    costs n. Callers with a per-span fallback pass behind them should
    bound the descent and let that pass do the fine-grained work;
    callers with no fallback (comments) must leave it at 1, since
    singleton probes are the only thing that can isolate the one span
    that matters.

    Every set returned has been confirmed removable by `ok`, so the
    caller never has to re-validate the result.
    """
    if not spans:
        return []
    if ok(list(spans)):
        return list(spans)
    if len(spans) <= max(1, min_chunk):
        return []

    mid = len(spans) // 2
    left = _largest_removable_subset(spans[:mid], ok, min_chunk, _depth + 1)
    right = _largest_removable_subset(spans[mid:], ok, min_chunk, _depth + 1)

    combined = left + right
    # Each half is removable alone; together they may not be (one may
    # only be safe while the other is present), so confirm before
    # claiming the union.
    if left and right and ok(combined):
        return combined
    return left if len(left) >= len(right) else right


def _is_parseable(source: str) -> bool:
    """Cheap local check that a candidate is even syntactically valid.

    Removing a unit can strand a block with no body, which no oracle
    would ever accept. Catching that here turns a ~170ms subprocess into
    a microsecond parse and rejects exactly the same candidates.
    """
    try:
        ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _block_child(node):
    for child in node.children:
        if child.type == "block":
            return child
    return None


def _body_stub_edits(tree, source: str) -> List[Tuple[Tuple[int, int], str]]:
    """Block spans that could collapse to `pass`, largest first.

    Deletion can never empty a function body: Python requires at least
    one statement, so whatever survives last — very often a multi-line
    docstring — is pinned in place with no way to remove it. Replacing
    the block with `pass` says "nothing in this body matters", which
    deletion has no vocabulary for. C-Reduce and Chisel both lean on
    this class of rewrite for the same reason.
    """
    edits: List[Tuple[Tuple[int, int], str]] = []
    offsets = byte_to_char_offsets(source)
    for node in _walk(tree.root_node):
        if node.type not in ("function_definition", "class_definition"):
            continue
        block = _block_child(node)
        if block is None:
            continue
        start = to_char_offset(offsets, block.start_byte)
        end = to_char_offset(offsets, block.end_byte)
        if source[start:end].strip() == "pass":
            continue
        indent = " " * (node.start_point[1] + 4)
        edits.append(((start, end), "\n" + indent + "pass"))
    edits.sort(key=lambda e: -(e[0][1] - e[0][0]))
    return edits


def _comment_spans(tree, source: str) -> List[Tuple[int, int]]:
    offsets = byte_to_char_offsets(source)
    return [(to_char_offset(offsets, n.start_byte),
             to_char_offset(offsets, n.end_byte))
            for n in _walk(tree.root_node) if n.type == "comment"]


def _collapse_blank_lines(source: str) -> str:
    """Drop lines left empty by byte-range removals.

    Blank lines carry no meaning in Python outside string literals, and
    a reduced file is mostly blank lines because each removal stops short
    of its trailing newline. The result is checked for parseability here
    and validated by the caller before being accepted.
    """
    kept = [line for line in source.splitlines() if line.strip()]
    if not kept:
        return source
    out = "\n".join(kept) + ("\n" if source.endswith("\n") else "")
    return out if _is_parseable(out) else source


def _make_training_logger():
    """Build a training logger if the environment asks for one.

    Import is deferred and failures are swallowed on purpose: the ml/
    package is an optional extra, and a reducer run must never fail
    because data capture is unavailable.
    """
    try:
        from ml.training_log import TrainingLogger
    except ImportError:
        try:
            from .ml.training_log import TrainingLogger  # type: ignore
        except (ImportError, ValueError):
            return None
    try:
        return TrainingLogger.from_env()
    except Exception:
        return None


@dataclass
class ReductionStats:
    """Statistics for the reduction process."""
    original_size: int  # Lines in original
    final_size: int  # Lines in minimized
    iterations: int  # Number of reduction iterations
    queries: int  # Number of validation queries
    time_seconds: float  # Total time
    successful_removals: int  # Number of successful unit removals
    failed_removals: int  # Number of failed unit removals
    oracle_skipped: int = 0  # Attempts the learned oracle declined to make
    syntax_rejected: int = 0  # Candidates rejected locally, no query spent
    # Queries run in parallel whose answer was thrown away because an
    # earlier candidate in the same batch decided differently than the
    # speculation assumed. Real subprocess work, deliberately not counted
    # in `queries`: `queries` is what the reduction logically asked, and
    # must stay comparable with a sequential run.
    speculative_discarded: int = 0

    @property
    def reduction_rate(self) -> float:
        """Calculate size reduction rate."""
        if self.original_size == 0:
            return 0.0
        return (self.original_size - self.final_size) / self.original_size


@dataclass
class ReductionResult:
    """Result of the reduction process."""
    minimized_code: str
    success: bool
    stats: ReductionStats
    original_trace: Optional[ExecutionTrace]


class HybridDeltaDebugger:
    """
    Hybrid Delta Debugging with Execution-guided reduction (HDD-E).

    Algorithm:
    1. Parse source into AST and identify removable units
    2. Collect execution traces (coverage data)
    3. Prioritize units: cold code first, then hot code by size
    4. Apply hierarchical reduction at multiple granularities
    5. Validate each reduction against original behavior
    6. Iterate until no further reduction is possible (1-minimality)
    """

    # Minimum outermost candidates before the batch/halving pass is worth
    # running. See _batch_deletion_pass for the measurements behind it.
    MIN_BATCH_CANDIDATES = 12

    # Smallest group the batch pass will subdivide. Its job is stripping
    # bulk; isolating individual units is the per-unit pass's job, and
    # doing it twice is what made the batch pass a net loss on the real
    # benchmark. See _largest_removable_subset.
    BATCH_MIN_CHUNK = 8

    def __init__(self, parser: Optional[PythonParser] = None,
                 tracer: Optional[ExecutionTracer] = None,
                 validator: Optional[Validator] = None,
                 verbose: bool = False,
                 oracle: Optional[object] = None,
                 oracle_skip_threshold: float = 0.1,
                 speculator: Optional[object] = None):
        """
        Initialize the reducer.

        Args:
            parser: PythonParser instance
            tracer: ExecutionTracer instance
            validator: Validator instance
            verbose: Print progress information
            oracle: Optional LearnedRemovabilityOracle. When present,
                units it scores below `oracle_skip_threshold` are never
                attempted. This trades output quality for queries: a
                miscalibrated model that calls a removable unit hopeless
                leaves that code in the result permanently, because
                nothing revisits a skipped unit.
            oracle_skip_threshold: p(safe) below which a unit is skipped.
            speculator: Optional started ParallelOracle. When present the
                accumulating pass asks several candidates at once. It is
                required to produce the identical accepted set, so this
                changes runtime and nothing else.
        """
        self.parser = parser or PythonParser()
        self.tracer = tracer or ExecutionTracer()
        self.validator = validator or Validator()
        self.verbose = verbose
        self.oracle = oracle
        self.oracle_skip_threshold = oracle_skip_threshold
        self._speculator = speculator
        # Set per-reduce(); see reduce() for why these exist.
        self._file_path: Optional[Path] = None
        self._executed: Optional[Set[int]] = None
        # Only non-None when AUTOREPRO_TRAINING_LOG is set, so a normal
        # reduction pays nothing and the core reducer keeps working even
        # if the optional ml/ package is absent.
        self._training_logger = _make_training_logger()

    def reduce(self, source_code: str,
               test_command: Optional[List[str]] = None,
               cwd: Optional[Path] = None,
               max_iterations: int = 100,
               file_path: Optional[Path] = None,
               executed_lines: Optional[Set[int]] = None) -> ReductionResult:
        """
        Reduce source code while preserving bug reproduction.

        Args:
            source_code: Original source code
            test_command: Command to reproduce the bug
            cwd: Working directory for execution
            max_iterations: Maximum reduction iterations
            file_path: Path the source came from. Only used as context —
                the reducer never reads or writes it — but the learned
                oracle needs it to tell a test module from a conftest.
            executed_lines: 1-indexed coverage for `source_code`. When a
                caller already knows which lines ran under the real test
                command, passing it here beats the fallback below:
                without it the reducer traces the source standalone,
                which for a library module records little more than
                import-time execution and marks every function body cold.

        Returns:
            ReductionResult with minimized code and statistics
        """
        start_time = time.time()
        original_lines = len(source_code.split('\n'))

        self._file_path = Path(file_path) if file_path is not None else None
        self._executed = set(executed_lines) if executed_lines is not None else None

        stats = ReductionStats(
            original_size=original_lines,
            final_size=original_lines,
            iterations=0,
            queries=0,
            time_seconds=0.0,
            successful_removals=0,
            failed_removals=0
        )

        # Step 1: Capture original behavior
        if self.verbose:
            print("Step 1: Capturing original behavior...")

        # This step exists to supply two things: coverage for
        # `source_code`, and an original behavior for the validator to
        # compare against. A caller that already has both — which is
        # every multi-file reduction, where the project validator holds
        # the real oracle and Phase 4 hands down real coverage — would
        # be paying a subprocess for answers that are then discarded.
        # `_current_executed` prefers `self._executed`, and
        # ProjectFileValidator consults the project's oracle rather than
        # the behavior set here, so neither result is ever read. That is
        # one wasted trace per file under reduction, and one query added
        # to the reported total for work no oracle did.
        caller_supplied_both = (
            executed_lines is not None
            and self.validator.original_behavior is not None)

        if test_command:
            orig_trace = self.tracer.trace_command(test_command, cwd=cwd)
        elif caller_supplied_both:
            orig_trace = ExecutionTrace(executed_lines={}, output="",
                                        return_code=0, success=True)
        else:
            # Try to run the code directly
            orig_trace = self._trace_source(source_code, cwd)

        if orig_trace is None:
            return ReductionResult(
                minimized_code=source_code,
                success=False,
                stats=stats,
                original_trace=None
            )

        if not caller_supplied_both:
            stats.queries += 1
            self.validator.set_original_behavior(
                orig_trace.output, 0 if orig_trace.success else 1)

        if self.verbose:
            print(f"  Original: {original_lines} lines")
            print(f"  Executed: {orig_trace.total_executed_lines} lines")

        # Step 2: Initial parse to get all units
        if self.verbose:
            print("\nStep 2: Parsing source code...")

        current_code = source_code
        current_units = self._get_units(current_code, orig_trace)

        if self.verbose:
            print(f"  Found {len(current_units)} top-level units")

        # Step 3: Hierarchical reduction
        if self.verbose:
            print("\nStep 3: Starting reduction...")

        # Strip the bulk first: ask whether all top-level units can go and
        # halve on refusal. Cheap when most of the file is dead relative to
        # the target test, which is the common case.
        for _ in range(max_iterations):
            shrunk = self._batch_deletion_pass(current_code, orig_trace,
                                               test_command, cwd, stats)
            if shrunk is None:
                break
            current_code = shrunk

        # Each pass walks every candidate once, accumulating the removals
        # that stick; passes repeat until one changes nothing (1-minimal
        # with respect to these units). The previous shape restarted from
        # the top after every success, so a unit that failed was retried
        # on every subsequent iteration — O(passes x units) queries for
        # the same result. max_iterations is now a safety valve on passes
        # rather than a cap on total removals.
        for iteration in range(1, max_iterations + 1):
            stats.iterations += 1

            if self.verbose:
                print(f"\nPass {iteration}: "
                      f"{len(current_code.split(chr(10)))} lines")

            candidates = self._candidates(current_code, orig_trace)
            if not candidates:
                break

            hopeless = self._oracle_verdicts(candidates, current_code,
                                             orig_trace)

            removed_spans = self._accumulating_pass(
                current_code, candidates, hopeless, orig_trace,
                test_command, cwd, stats)

            if not removed_spans:
                break

            # Shift coverage into the new numbering once for the whole
            # pass, before current_code moves under it.
            if self._executed is not None:
                self._executed = remap_executed_lines(
                    current_code, removed_spans, self._executed)
            current_code = _apply_removals(current_code, removed_spans)

        # Deletion alone has now converged. What it cannot express is
        # "this body is irrelevant" — a function must keep at least one
        # statement — so collapse bodies to `pass` and then let deletion
        # run again over the smaller tree.
        before_stub = current_code
        current_code = self._stub_bodies(current_code, test_command, cwd,
                                         stats)
        if current_code != before_stub:
            for _ in range(max_iterations):
                shrunk = self._deletion_pass(current_code, orig_trace,
                                             test_command, cwd, stats)
                if shrunk is None:
                    break
                current_code = shrunk

        current_code = self._strip_comments(current_code, test_command,
                                            cwd, stats)

        # Deleted statements leave their trailing newline behind, so a
        # reduced file is mostly blank lines — on real inputs ~80% of
        # what survives. That inflates the reported size and makes the
        # result useless as a reproducer. Collapsing them is one query.
        cleaned = _collapse_blank_lines(current_code)
        if cleaned != current_code:
            stats.queries += 1
            if self._validate(cleaned, test_command, cwd):
                current_code = cleaned
            elif self.verbose:
                print("  blank-line cleanup broke the bug — kept as-is")

        # Final stats
        stats.final_size = len(current_code.split('\n'))
        stats.time_seconds = time.time() - start_time

        if self.verbose:
            print(f"\nReduction complete:")
            print(f"  Original: {stats.original_size} lines")
            print(f"  Minimized: {stats.final_size} lines")
            print(f"  Reduction: {stats.reduction_rate*100:.1f}%")
            print(f"  Queries: {stats.queries}")
            print(f"  Time: {stats.time_seconds:.2f}s")

        return ReductionResult(
            minimized_code=current_code,
            success=True,
            stats=stats,
            original_trace=orig_trace
        )

    # ------------------------------------------------- accumulating pass

    def _accumulating_pass(self, current_code: str,
                           candidates: List[CodeUnit],
                           hopeless: Set[int],
                           trace: ExecutionTrace,
                           test_command, cwd, stats) -> List[Tuple[int, int]]:
        """Walk every candidate once, keeping the removals that stick.

        Dispatches to the speculative walk when a parallel oracle is
        available. The two are required to return the same list — see
        `_speculative_pass` for why that is guaranteed rather than hoped
        for, and tests/test_speculation.py for the check.
        """
        if self._speculator is not None and self._speculator.enabled:
            return self._speculative_pass(current_code, candidates, hopeless,
                                          trace, test_command, cwd, stats)
        return self._sequential_pass(current_code, candidates, hopeless,
                                     trace, test_command, cwd, stats)

    def _sequential_pass(self, current_code, candidates, hopeless, trace,
                         test_command, cwd, stats) -> List[Tuple[int, int]]:
        removed_spans: List[Tuple[int, int]] = []
        for unit in candidates:
            span = (unit.start_char, unit.end_char)

            # Skip anything already inside a removed region: its bytes are
            # gone, so the edit is meaningless and the overlapping slice
            # would corrupt the source.
            if _overlaps(span, removed_spans):
                continue

            if id(unit) in hopeless:
                stats.oracle_skipped += 1
                continue

            candidate = _apply_removals(current_code, removed_spans + [span])

            # Reject unparseable candidates locally. The oracle here is a
            # subprocess costing ~170ms; a syntax check costs microseconds
            # and rejects exactly the same candidates.
            if not _is_parseable(candidate):
                stats.syntax_rejected += 1
                continue

            is_valid = self._validate(candidate, test_command, cwd)
            stats.queries += 1

            # Log against the state the unit was measured in.
            if self._training_logger is not None:
                self._training_logger.log_attempt(
                    unit=unit,
                    source=current_code,
                    executed_lines=self._current_executed(trace),
                    file_path=self._file_path,
                    was_safely_removable=is_valid)

            if is_valid:
                removed_spans.append(span)
                stats.successful_removals += 1
                if self.verbose:
                    print(f"  Removed {unit.node_type} "
                          f"(L{unit.start_line}-{unit.end_line})")
            else:
                stats.failed_removals += 1
        return removed_spans

    def _speculative_pass(self, current_code, candidates, hopeless, trace,
                          test_command, cwd, stats) -> List[Tuple[int, int]]:
        """The same walk, asking several questions at once.

        The pass is sequential by nature: whether candidate *i+1* is even
        offered depends on whether *i* was accepted, because an accepted
        span changes both the source the next candidate is cut from and
        which later spans now overlap a hole. To ask in parallel you must
        guess the answers, and the guess decides the payoff.

        Measured on the benchmark, **74.7% of candidates in this pass are
        accepted**. So the guess is "accepted", and the batch is a *chain*:
        each entry assumes every entry before it stuck. C-Reduce
        speculates the other way, assuming candidates fail, which is right
        for its workload and wrong for this one — assuming failure here
        would cap the speedup at 1.34x however many workers you add,
        against 3.6x for this direction at eight.

        Equivalence with the sequential walk is structural, not
        statistical. Every decision in the chain is taken against exactly
        the state the sequential walk would have been in, *provided* every
        earlier guess held. So the plan is valid up to and including the
        first entry whose verdict differs from the guess; everything after
        that is discarded and rescanned from the correct state. The
        accepted set is therefore identical, and so is the query count —
        `speculative_discarded` records the extra subprocess work, which
        is the price paid, and is reported separately rather than hidden
        inside `queries`.
        """
        from multi_file.parallel_oracle import Candidate

        width = max(2, self._speculator.jobs)
        relative = self._speculator.relative_path(self._file_path)

        removed_spans: List[Tuple[int, int]] = []
        index = 0
        while index < len(candidates):
            # Build the chain, assuming each queued candidate is accepted.
            # `plan` records every decision, not just the queries, so the
            # bookkeeping below can be replayed exactly once.
            plan: List[Tuple[str, int, Optional[Tuple[int, int]],
                             Optional[str]]] = []
            trial = list(removed_spans)
            scan = index
            queued = 0
            while scan < len(candidates) and queued < width:
                unit = candidates[scan]
                span = (unit.start_char, unit.end_char)
                position = scan
                scan += 1

                if _overlaps(span, trial):
                    plan.append(("overlap", position, None, None))
                    continue
                if id(unit) in hopeless:
                    plan.append(("oracle", position, None, None))
                    continue
                candidate = _apply_removals(current_code, trial + [span])
                if not _is_parseable(candidate):
                    plan.append(("syntax", position, None, None))
                    continue

                plan.append(("query", position, span, candidate))
                trial = trial + [span]
                queued += 1

            asked = [step for step in plan if step[0] == "query"]
            if not asked:
                break

            verdicts = self._speculator.ask(
                [Candidate(path=relative, source=step[3]) for step in asked])

            # Replay the plan against the real answers, stopping at the
            # first place the guess was wrong.
            verdict_iter = iter(verdicts)
            used = 0
            broke_at: Optional[int] = None
            for kind, position, span, _source in plan:
                if kind == "overlap":
                    continue
                if kind == "oracle":
                    stats.oracle_skipped += 1
                    continue
                if kind == "syntax":
                    stats.syntax_rejected += 1
                    continue

                is_valid = next(verdict_iter)
                used += 1
                stats.queries += 1

                if self._training_logger is not None:
                    unit = candidates[position]
                    self._training_logger.log_attempt(
                        unit=unit,
                        source=current_code,
                        executed_lines=self._current_executed(trace),
                        file_path=self._file_path,
                        was_safely_removable=is_valid)

                if is_valid:
                    removed_spans.append(span)
                    stats.successful_removals += 1
                    if self.verbose:
                        unit = candidates[position]
                        print(f"  Removed {unit.node_type} "
                              f"(L{unit.start_line}-{unit.end_line})")
                else:
                    stats.failed_removals += 1
                    # The chain assumed this one stuck. Everything after
                    # it was cut from a source that does not exist.
                    broke_at = position
                    break

            stats.speculative_discarded += len(asked) - used
            index = scan if broke_at is None else broke_at + 1

        return removed_spans

    def _get_units(self, source_code: str, trace: ExecutionTrace) -> List[CodeUnit]:
        """Get removable units from source code."""
        tree = self.parser.parse_source(source_code)
        return self.parser.extract_units(tree, source_code,
                                         self._current_executed(trace))

    def _current_executed(self, trace: ExecutionTrace) -> Set[int]:
        """Coverage for the code as it stands now.

        Prefers what the caller supplied (real coverage under the test
        command, kept in step with each removal); falls back to the
        reducer's own standalone trace.
        """
        if self._executed is not None:
            return self._executed

        executed_lines: Set[int] = set()
        for lines in trace.executed_lines.values():
            executed_lines.update(lines)
        return executed_lines

    def _get_flat_units(self, source_code: str, trace: ExecutionTrace) -> List[CodeUnit]:
        """Get flattened list of removable units."""
        units = self._get_units(source_code, trace)
        return self.parser.get_flat_units(units)

    def _batch_deletion_pass(self, source: str, trace: ExecutionTrace,
                             test_command, cwd, stats) -> Optional[str]:
        """Ask whether the whole candidate set can go, then halve.

        The per-unit pass spends one query per candidate whatever the
        answer. A test exercises a narrow slice of a module, so most of
        the file is usually dead relative to it and the answer to "can
        all of this go?" is yes — one query instead of hundreds. Halving
        only pays for the boundaries it actually has to locate.

        Restricted to outermost spans. A batch holding both a function
        and a statement inside it would splice overlapping slices; the
        per-unit pass reaches inside whatever survives.
        """
        candidates = self._candidates(source, trace)
        if not candidates:
            return None

        outermost: List[Tuple[int, int]] = []
        for unit in sorted(candidates,
                           key=lambda u: (u.start_char,
                                          -(u.end_char - u.start_char))):
            span = (unit.start_char, unit.end_char)
            if not _overlaps(span, outermost):
                outermost.append(span)

        # Halving only pays when there is enough to halve. On a four-unit
        # file it went 11->15: with few candidates the probes spent
        # locating boundaries outnumber the per-unit queries they
        # replace. Below the threshold the per-unit pass is already
        # cheap, so skip straight to it.
        #
        # Measured in isolation this pass cut queries 35-40% on single
        # modules (utils.py 95->62, app.py 101->63). Do not read that as
        # the pipeline's behavior: in the pipeline Phase 4a has already
        # dropped the zero-coverage definitions, so the surviving
        # candidates are mostly live, the bulk probes fail, and the win
        # shrinks to whatever BATCH_MIN_CHUNK still allows.
        if len(outermost) < self.MIN_BATCH_CANDIDATES:
            return None

        def ok(subset: List[Tuple[int, int]]) -> bool:
            cand = _apply_removals(source, subset)
            if not _is_parseable(cand):
                stats.syntax_rejected += 1
                return False
            stats.queries += 1
            if self._validate(cand, test_command, cwd):
                return True
            stats.failed_removals += 1
            return False

        removable = _largest_removable_subset(outermost, ok,
                                              min_chunk=self.BATCH_MIN_CHUNK)
        if not removable:
            return None

        stats.successful_removals += len(removable)
        if self.verbose:
            print(f"  Batch-removed {len(removable)} of {len(outermost)} "
                  "top-level units")
        if self._executed is not None:
            self._executed = remap_executed_lines(source, removable,
                                                  self._executed)
        return _apply_removals(source, removable)

    def _deletion_pass(self, source: str, trace: ExecutionTrace,
                       test_command, cwd, stats) -> Optional[str]:
        """One accumulating deletion pass. None if nothing was removed."""
        candidates = self._candidates(source, trace)
        if not candidates:
            return None
        removed: List[Tuple[int, int]] = []
        for unit in candidates:
            span = (unit.start_char, unit.end_char)
            if _overlaps(span, removed):
                continue
            cand = _apply_removals(source, removed + [span])
            if not _is_parseable(cand):
                stats.syntax_rejected += 1
                continue
            stats.queries += 1
            if self._validate(cand, test_command, cwd):
                removed.append(span)
                stats.successful_removals += 1
            else:
                stats.failed_removals += 1
        if not removed:
            return None
        if self._executed is not None:
            self._executed = remap_executed_lines(source, removed,
                                                  self._executed)
        return _apply_removals(source, removed)

    def _stub_bodies(self, source: str, test_command, cwd,
                     stats, max_passes: int = 100) -> str:
        """Collapse function and class bodies to `pass` where possible.

        Accumulating passes, same shape as deletion: restarting after
        each accepted stub would cost O(successes x bodies) queries on a
        file with hundreds of definitions.

        Edits are applied largest-first and overlap-checked, because
        stubbing a class body subsumes every method inside it — applying
        both would splice one replacement into the middle of another.
        """
        current = source
        for _ in range(max_passes):
            tree = self.parser.parse_source(current)
            edits = _body_stub_edits(tree, current)
            if not edits:
                return current

            accepted: List[Tuple[Tuple[int, int], str]] = []
            for span, replacement in edits:
                if _overlaps(span, [s for s, _ in accepted]):
                    continue
                cand = _apply_edits(current, accepted + [(span, replacement)])
                if not _is_parseable(cand):
                    stats.syntax_rejected += 1
                    continue
                stats.queries += 1
                if self._validate(cand, test_command, cwd):
                    accepted.append((span, replacement))
                    if self.verbose:
                        print("  Stubbed a body to pass")

            if not accepted:
                return current
            current = _apply_edits(current, accepted)
        return current

    def _strip_comments(self, source: str, test_command, cwd,
                        stats) -> str:
        """Drop comments, all at once if possible.

        Comments are never semantically required, but a directive such as
        `# type: ignore` or `# noqa` can matter to a tool in the loop, so
        the batch removal is validated and falls back to one-at-a-time
        rather than assumed safe.
        """
        tree = self.parser.parse_source(source)
        spans = _comment_spans(tree, source)
        if not spans:
            return source

        def ok(spans_to_remove: List[Tuple[int, int]]) -> bool:
            cand = _apply_removals(source, spans_to_remove)
            if not _is_parseable(cand):
                stats.syntax_rejected += 1
                return False
            stats.queries += 1
            return self._validate(cand, test_command, cwd)

        removable = _largest_removable_subset(spans, ok)
        return _apply_removals(source, removable) if removable else source

    def _candidates(self, source_code: str,
                    trace: ExecutionTrace) -> List[CodeUnit]:
        """Prioritized, de-duplicated removal candidates.

        tree-sitter wraps `a = 1` in an expression_statement around an
        assignment covering the same bytes, so without deduping the same
        edit is proposed twice and costs two queries.

        Spans that would remove nothing are dropped here rather than in
        each caller, since all three deletion passes come through this
        one function. Such a span is not merely useless, it is a trap:
        the candidate equals its input, so it validates, the pass
        records a removal, the source does not shrink, and the outer
        loop comes round again on identical input until max_iterations
        stops it. A byte-vs-character offset bug produced exactly that
        and spent 100 queries on one file. The offsets are fixed; this
        keeps the next such bug from being expensive as well as wrong.
        """
        units = self._get_flat_units(source_code, trace)
        seen = set()
        unique: List[CodeUnit] = []
        for unit in units:
            if unit.end_char <= unit.start_char:
                continue
            if unit.start_char >= len(source_code):
                continue
            key = (unit.start_char, unit.end_char)
            if key in seen:
                continue
            seen.add(key)
            unique.append(unit)
        return self._prioritize_units(unique)

    def _prioritize_units(self, units: List[CodeUnit]) -> List[CodeUnit]:
        """Prioritize units for reduction (cold first, then by size)."""
        return self.parser.prioritize_units(units)

    def _oracle_verdicts(self, units: List[CodeUnit], source: str,
                         trace: ExecutionTrace) -> Set[int]:
        """ids of units the oracle says are not worth attempting.

        Identity rather than value: two textually identical statements
        are equal as dataclasses but are different removal candidates.

        Any failure here yields an empty set, so a broken or missing
        model costs nothing but the plain heuristic path.
        """
        if self.oracle is None:
            return set()

        scored = [u for u in units if getattr(u, "ts_node", None) is not None]
        if not scored:
            return set()

        try:
            from ml.features import extract_features
        except ImportError:
            return set()

        try:
            executed = self._current_executed(trace)
            feats = [extract_features(u.ts_node, source, executed,
                                      self._file_path) for u in scored]
            probs = self.oracle.predict_batch(feats)
        except Exception:
            return set()

        return {id(u) for u, p in zip(scored, probs)
                if p < self.oracle_skip_threshold}

    def _remove_units(self, source_code: str, units: List[CodeUnit]) -> str:
        """Remove specified units from source code."""
        return format_code_without_units(source_code, units)

    def _validate(self, source_code: str, test_command: Optional[List[str]],
                  cwd: Optional[Path]) -> bool:
        """Validate that source code still reproduces the bug."""
        result = self.validator.validate(source_code, test_command, cwd)
        return result.is_valid

    def _trace_source(self, source_code: str, cwd: Optional[Path]) -> Optional[ExecutionTrace]:
        """Trace execution of source code directly."""
        import tempfile
        import os

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, dir=cwd) as f:
            f.write(source_code)
            temp_file = Path(f.name)

        try:
            trace = self.tracer.trace_python_file(temp_file)
            return trace
        except Exception:
            return None
        finally:
            try:
                temp_file.unlink()
            except:
                pass


class LineLevelDeltaDebugger:
    """
    Line-level delta debugging (vanilla ddmin-style).

    Simpler but less effective than HDD for structured code.
    Used as a baseline for comparison.
    """

    def __init__(self, validator: Optional[Validator] = None,
                 verbose: bool = False):
        """Initialize line-level debugger."""
        self.validator = validator or Validator()
        self.verbose = verbose

    def reduce(self, source_code: str,
               test_command: Optional[List[str]] = None,
               cwd: Optional[Path] = None) -> ReductionResult:
        """
        Reduce source code using line-level ddmin.

        Args:
            source_code: Original source code
            test_command: Command to reproduce the bug
            cwd: Working directory

        Returns:
            ReductionResult
        """
        start_time = time.time()
        lines = source_code.split('\n')
        original_count = len(lines)

        stats = ReductionStats(
            original_size=original_count,
            final_size=original_count,
            iterations=0,
            queries=0,
            time_seconds=0.0,
            successful_removals=0,
            failed_removals=0
        )

        # Capture original behavior
        result = self.validator.validate(source_code, test_command, cwd)
        if result.matches_original:
            self.validator.set_original_behavior(result.output, result.return_code)
        else:
            # First run establishes the behavior
            self.validator.set_original_behavior(result.output, result.return_code)

        stats.queries += 1

        # ddmin algorithm
        n = 2  # Initial granularity
        current_lines = lines[:]

        while len(current_lines) >= 2:
            stats.iterations += 1
            subset_size = len(current_lines) // n

            if subset_size == 0:
                break

            improved = False

            # Try removing each subset
            for i in range(n):
                start = i * subset_size
                end = start + subset_size if i < n - 1 else len(current_lines)

                # Create candidate without this subset
                candidate_lines = current_lines[:start] + current_lines[end:]
                candidate = '\n'.join(candidate_lines)

                if not candidate.strip():
                    continue

                # Validate
                is_valid = self._validate(candidate, test_command, cwd)
                stats.queries += 1

                if is_valid:
                    # Successful removal
                    current_lines = candidate_lines
                    stats.successful_removals += 1
                    improved = True

                    if self.verbose:
                        print(f"  Removed lines {start}-{end}")

                    break
                else:
                    stats.failed_removals += 1

            if not improved:
                # Increase granularity
                if n >= len(current_lines):
                    break
                n = min(n * 2, len(current_lines))

        final_code = '\n'.join(current_lines)
        stats.final_size = len(current_lines)
        stats.time_seconds = time.time() - start_time

        return ReductionResult(
            minimized_code=final_code,
            success=True,
            stats=stats,
            original_trace=None
        )

    def _validate(self, source_code: str, test_command: Optional[List[str]],
                  cwd: Optional[Path]) -> bool:
        """Validate source code."""
        result = self.validator.validate(source_code, test_command, cwd)
        return result.is_valid


# Factory function
def create_reducer(algorithm: str = "hdd-e", **kwargs) -> HybridDeltaDebugger:
    """
    Factory function to create a reducer.

    Args:
        algorithm: "hdd-e" (hybrid), "line" (line-level), or "ddmin"
        **kwargs: Additional arguments for the reducer

    Returns:
        Reducer instance
    """
    if algorithm == "hdd-e":
        return HybridDeltaDebugger(**kwargs)
    elif algorithm in ("line", "ddmin"):
        return LineLevelDeltaDebugger(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
