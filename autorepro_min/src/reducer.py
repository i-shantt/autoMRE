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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from parser import (
    CodeUnit,
    format_code_without_units,
    PythonParser,
    remap_executed_lines,
)
from tracer import ExecutionTracer, ExecutionTrace
from validator import Validator


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

    def __init__(self, parser: Optional[PythonParser] = None,
                 tracer: Optional[ExecutionTracer] = None,
                 validator: Optional[Validator] = None,
                 verbose: bool = False):
        """
        Initialize the reducer.

        Args:
            parser: PythonParser instance
            tracer: ExecutionTracer instance
            validator: Validator instance
            verbose: Print progress information
        """
        self.parser = parser or PythonParser()
        self.tracer = tracer or ExecutionTracer()
        self.validator = validator or Validator()
        self.verbose = verbose
        # Set per-reduce(); see reduce() for why these exist.
        self._file_path: Optional[Path] = None
        self._executed: Optional[Set[int]] = None

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

        if test_command:
            orig_trace = self.tracer.trace_command(test_command, cwd=cwd)
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

        stats.queries += 1
        self.validator.set_original_behavior(orig_trace.output,
                                             0 if orig_trace.success else 1)

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

        iteration = 0
        improved = True

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            stats.iterations += 1

            if self.verbose:
                print(f"\nIteration {iteration}: {len(current_code.split(chr(10)))} lines")

            # Get flat list of removable units
            flat_units = self._get_flat_units(current_code, orig_trace)

            if not flat_units:
                break

            # Prioritize units (execution-guided, then size)
            prioritized = self._prioritize_units(flat_units)

            # Try removing each unit
            for unit in prioritized:
                # Attempt removal
                candidate = self._remove_units(current_code, [unit])

                # Validate
                is_valid = self._validate(candidate, test_command, cwd)
                stats.queries += 1

                if is_valid:
                    # Successful removal. Shift the coverage set into the
                    # new numbering before current_code changes under it,
                    # otherwise every later iteration reads coverage
                    # against lines that have moved.
                    if self._executed is not None:
                        self._executed = remap_executed_lines(
                            current_code,
                            [(unit.start_byte, unit.end_byte)],
                            self._executed)

                    current_code = candidate
                    stats.successful_removals += 1
                    improved = True

                    if self.verbose:
                        print(f"  Removed {unit.node_type} (L{unit.start_line}-{unit.end_line})")

                    # Re-parse with updated code
                    break  # Start new iteration
                else:
                    stats.failed_removals += 1

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

    def _prioritize_units(self, units: List[CodeUnit]) -> List[CodeUnit]:
        """Prioritize units for reduction (cold first, then by size)."""
        return self.parser.prioritize_units(units)

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
