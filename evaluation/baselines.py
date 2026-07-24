"""
AutoRepro-Min: Automated Bug Reproduction Minimization
Baseline Algorithms Module

Implements baseline reduction algorithms for comparison:
1. Vanilla ddmin (line-level, no AST awareness)
2. Syntax-aware only (AST-guided, no execution trace)
3. Random reduction (sanity check)

Based on:
- Zeller & Hildebrandt (2002): ddmin algorithm
- Sun et al. (2018): Perses syntax-guided reduction
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'autorepro_min' / 'src'))

from parser import PythonParser, format_code_without_units
from tracer import ExecutionTracer
from validator import Validator


@dataclass
class BaselineResult:
    """Result from a baseline reduction."""
    minimized_code: str
    original_size: int
    minimized_size: int
    queries: int
    time_seconds: float
    success: bool

    @property
    def reduction_rate(self) -> float:
        """Calculate reduction rate."""
        if self.original_size == 0:
            return 0.0
        return (self.original_size - self.minimized_size) / self.original_size


class VanillaDDMin:
    """
    Vanilla ddmin algorithm (line-level).

    No AST awareness, no execution guidance.
    Pure binary search reduction on lines.
    """

    def __init__(self, validator: Optional[Validator] = None,
                 timeout: int = 30,
                 verbose: bool = False):
        """Initialize ddmin."""
        self.validator = validator or Validator()
        self.timeout = timeout
        self.verbose = verbose

    def reduce(self, source_code: str,
               test_command: Optional[List[str]] = None,
               cwd: Optional[Path] = None,
               max_iterations: int = 100) -> BaselineResult:
        """
        Reduce source code using vanilla ddmin.

        Args:
            source_code: Original source code
            test_command: Command to reproduce bug
            cwd: Working directory
            max_iterations: Maximum iterations

        Returns:
            BaselineResult
        """
        import subprocess
        import tempfile

        start_time = time.time()
        lines = source_code.split('\n')
        original_count = len(lines)

        # Capture original behavior using subprocess
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, dir=cwd) as f:
            f.write(source_code)
            temp_file = Path(f.name)

        try:
            if test_command:
                cmd = test_command
            else:
                cmd = ['python', str(temp_file)]

            result = subprocess.run(cmd, cwd=cwd, capture_output=True,
                                   text=True, timeout=self.timeout)
            self.validator.set_original_behavior(
                result.stdout + result.stderr,
                result.returncode
            )
        finally:
            temp_file.unlink()

        queries = 1
        n = 2  # Initial granularity
        current_lines = lines[:]
        iteration = 0

        while len(current_lines) >= 2 and iteration < max_iterations:
            iteration += 1
            subset_size = max(1, len(current_lines) // n)

            if subset_size == 0:
                break

            improved = False

            # Try removing each subset
            for i in range(n):
                start = i * subset_size
                end = min(start + subset_size, len(current_lines))

                # Create candidate without this subset
                candidate_lines = current_lines[:start] + current_lines[end:]
                candidate = '\n'.join(candidate_lines)

                if not candidate.strip():
                    continue

                # Validate
                val_result = self.validator.validate(candidate, test_command, cwd)
                queries += 1

                if val_result.is_valid:
                    # Successful removal
                    current_lines = candidate_lines
                    improved = True

                    if self.verbose:
                        print(f"  Removed lines {start}-{end}")

                    break

            if not improved:
                # Increase granularity
                if n >= len(current_lines):
                    break
                n = min(n * 2, len(current_lines))

        final_code = '\n'.join(current_lines)
        time_seconds = time.time() - start_time

        return BaselineResult(
            minimized_code=final_code,
            original_size=original_count,
            minimized_size=len(current_lines),
            queries=queries,
            time_seconds=time_seconds,
            success=True
        )


class SyntaxAwareReducer:
    """
    Syntax-aware reduction (AST-guided, no execution trace).

    Uses AST structure but not execution data.
    """

    def __init__(self, validator: Optional[Validator] = None,
                 timeout: int = 30,
                 verbose: bool = False):
        """Initialize reducer."""
        self.parser = PythonParser()
        self.validator = validator or Validator()
        self.timeout = timeout
        self.verbose = verbose

    def reduce(self, source_code: str,
               test_command: Optional[List[str]] = None,
               cwd: Optional[Path] = None,
               max_iterations: int = 100) -> BaselineResult:
        """
        Reduce using syntax-aware approach.

        Args:
            source_code: Original source code
            test_command: Command to reproduce bug
            cwd: Working directory
            max_iterations: Maximum iterations

        Returns:
            BaselineResult
        """
        start_time = time.time()
        original_lines = len(source_code.split('\n'))

        # Capture original behavior using subprocess
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, dir=cwd) as f:
            f.write(source_code)
            temp_file = Path(f.name)

        try:
            if test_command:
                cmd = test_command
            else:
                cmd = ['python', str(temp_file)]

            result = subprocess.run(cmd, cwd=cwd, capture_output=True,
                                   text=True, timeout=self.timeout)
            self.validator.set_original_behavior(
                result.stdout + result.stderr,
                result.returncode
            )
        finally:
            temp_file.unlink()

        queries = 1
        current_code = source_code
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Parse current code
            tree = self.parser.parse_source(current_code)
            # No execution data, so all units treated equally
            units = self.parser.extract_units(tree, current_code, set())
            flat_units = self.parser.get_flat_units(units)

            if not flat_units:
                break

            improved = False

            # Try removing each unit (in order, no prioritization)
            for unit in flat_units:
                candidate = format_code_without_units(current_code, [unit])

                # Validate
                val_result = self.validator.validate(candidate, test_command, cwd)
                queries += 1

                if val_result.is_valid:
                    current_code = candidate
                    improved = True

                    if self.verbose:
                        print(f"  Removed {unit.node_type}")

                    break

            if not improved:
                break

        minimized_lines = len(current_code.split('\n'))
        time_seconds = time.time() - start_time

        return BaselineResult(
            minimized_code=current_code,
            original_size=original_lines,
            minimized_size=minimized_lines,
            queries=queries,
            time_seconds=time_seconds,
            success=True
        )


class RandomReducer:
    """
    Random reduction (sanity check baseline).

    Randomly removes code units and validates.
    """

    def __init__(self, validator: Optional[Validator] = None,
                 timeout: int = 30,
                 verbose: bool = False,
                 seed: int = 42):
        """Initialize reducer."""
        self.parser = PythonParser()
        self.validator = validator or Validator()
        self.timeout = timeout
        self.verbose = verbose
        self.random = random.Random(seed)

    def reduce(self, source_code: str,
               test_command: Optional[List[str]] = None,
               cwd: Optional[Path] = None,
               max_queries: int = 100) -> BaselineResult:
        """
        Reduce using random approach.

        Args:
            source_code: Original source code
            test_command: Command to reproduce bug
            cwd: Working directory
            max_queries: Maximum validation queries

        Returns:
            BaselineResult
        """
        start_time = time.time()
        original_lines = len(source_code.split('\n'))

        # Capture original behavior using subprocess
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, dir=cwd) as f:
            f.write(source_code)
            temp_file = Path(f.name)

        try:
            if test_command:
                cmd = test_command
            else:
                cmd = ['python', str(temp_file)]

            result = subprocess.run(cmd, cwd=cwd, capture_output=True,
                                   text=True, timeout=self.timeout)
            self.validator.set_original_behavior(
                result.stdout + result.stderr,
                result.returncode
            )
        finally:
            temp_file.unlink()

        queries = 1
        current_code = source_code

        # Parse to get removable units
        tree = self.parser.parse_source(current_code)
        units = self.parser.extract_units(tree, current_code, set())
        flat_units = self.parser.get_flat_units(units)

        # Shuffle units randomly
        self.random.shuffle(flat_units)

        # Try removing units in random order
        for unit in flat_units:
            if queries >= max_queries:
                break

            candidate = format_code_without_units(current_code, [unit])

            # Validate
            val_result = self.validator.validate(candidate, test_command, cwd)
            queries += 1

            if val_result.is_valid:
                current_code = candidate

                if self.verbose:
                    print(f"  Removed {unit.node_type}")

                # Re-parse to get updated units
                tree = self.parser.parse_source(current_code)
                units = self.parser.extract_units(tree, current_code, set())
                flat_units = self.parser.get_flat_units(units)
                self.random.shuffle(flat_units)

        minimized_lines = len(current_code.split('\n'))
        time_seconds = time.time() - start_time

        return BaselineResult(
            minimized_code=current_code,
            original_size=original_lines,
            minimized_size=minimized_lines,
            queries=queries,
            time_seconds=time_seconds,
            success=True
        )


# Factory function
def create_baseline(algorithm: str, **kwargs):
    """
    Factory function to create baseline reducer.

    Args:
        algorithm: 'ddmin', 'syntax', or 'random'
        **kwargs: Additional arguments

    Returns:
        Baseline reducer instance
    """
    if algorithm == 'ddmin':
        return VanillaDDMin(**kwargs)
    elif algorithm == 'syntax':
        return SyntaxAwareReducer(**kwargs)
    elif algorithm == 'random':
        return RandomReducer(**kwargs)
    else:
        raise ValueError(f"Unknown baseline: {algorithm}")
