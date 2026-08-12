"""
autoMRE: Automated Bug Reproduction Minimization
Execution Tracer Module

This module provides execution tracing using coverage.py to identify
which lines of code are executed during test reproduction.

Based on:
- coverage.py: Standard Python coverage measurement
- Eghbali & Pradel (2022): Dynamic analysis frameworks
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import coverage

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from validator import oracle_env  # noqa: E402


@dataclass
class ExecutionTrace:
    """Represents the execution trace of a program run."""
    executed_lines: Dict[str, Set[int]]  # file_path -> set of line numbers
    output: str  # stdout + stderr
    return_code: int
    success: bool  # Did the command complete without error?

    def get_lines_for_file(self, file_path: str) -> Set[int]:
        """Get executed lines for a specific file."""
        # Normalize path for lookup
        abs_path = str(Path(file_path).resolve())
        for path, lines in self.executed_lines.items():
            if Path(path).resolve() == Path(abs_path).resolve():
                return lines
        return set()

    @property
    def total_executed_lines(self) -> int:
        """Total number of unique lines executed across all files."""
        return sum(len(lines) for lines in self.executed_lines.values())


class ExecutionTracer:
    """
    Traces execution of Python code using coverage.py.

    Provides detailed information about which lines were executed,
    enabling execution-guided reduction (cold code removal first).
    """

    def __init__(self, timeout: int = 60, python: Optional[str] = None):
        """
        Initialize the tracer.

        Args:
            timeout: Maximum time in seconds to wait for execution
            python: Interpreter to run coverage under. Must be the same
                one the reproduction command uses, or the trace measures
                a different environment than the reduction is validated
                in. Defaults to the interpreter running this process.

                This is not hypothetical: tracing flask under the ambient
                Python (pytest 9.1.1) errored during collection and
                collected 2 files, while the same command under the
                benchmark venv (pytest 9.0.0) passed and collected 23.
                Phase 1 then classified 51 of 82 files as unreachable,
                Phase 2's batch delete broke the test, and the whole
                reduction rolled back to zero.
        """
        self.timeout = timeout
        self.python = python or sys.executable

    def trace_command(self, command: List[str], cwd: Optional[Path] = None,
                      env: Optional[Dict[str, str]] = None) -> ExecutionTrace:
        """
        Execute a command with coverage tracing.

        Args:
            command: Command to execute (e.g., ['python', 'test.py'])
            cwd: Working directory for execution
            env: Environment variables

        Returns:
            ExecutionTrace with coverage data
        """
        # Create temporary coverage configuration
        with tempfile.TemporaryDirectory() as tmpdir:
            coverage_file = Path(tmpdir) / ".coverage"

            # Set up coverage environment. Routed through oracle_env so the
            # trace sees the same code the validator will — see its docstring
            # for why stale bytecode has to be ruled out.
            trace_env = oracle_env(env)
            trace_env['COVERAGE_FILE'] = str(coverage_file)

            # Run command with coverage
            try:
                result = subprocess.run(
                    [self.python, '-m', 'coverage', 'run', '--branch',
                     '--data-file', str(coverage_file),
                     '--source', '.'] + command,
                    cwd=cwd,
                    env=trace_env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                # Parse coverage data
                executed_lines = self._parse_coverage(coverage_file, cwd)

                return ExecutionTrace(
                    executed_lines=executed_lines,
                    output=result.stdout + result.stderr,
                    return_code=result.returncode,
                    success=result.returncode == 0
                )

            except subprocess.TimeoutExpired:
                return ExecutionTrace(
                    executed_lines={},
                    output="Timeout: Execution exceeded time limit",
                    return_code=-1,
                    success=False
                )
            except Exception as e:
                return ExecutionTrace(
                    executed_lines={},
                    output=f"Error: {str(e)}",
                    return_code=-1,
                    success=False
                )

    def _parse_coverage(self, coverage_file: Path,
                        cwd: Optional[Path]) -> Dict[str, Set[int]]:
        """
        Parse coverage data file to extract executed lines.

        Args:
            coverage_file: Path to coverage data file
            cwd: Working directory (for path resolution)

        Returns:
            Dictionary mapping file paths to sets of line numbers
        """
        executed_lines = {}

        if not coverage_file.exists():
            return executed_lines

        try:
            cov = coverage.Coverage(data_file=str(coverage_file))
            cov.load()

            data = cov.get_data()
            files = data.measured_files()

            for file_path in files:
                try:
                    # data.lines() returns lines actually executed.
                    # (Previously used analysis2()[1], which returns all
                    # executable statements — not what we want.)
                    executed = data.lines(file_path)
                    executed = set(executed) if executed else set()
                    if executed:
                        executed_lines[file_path] = executed
                except Exception:
                    continue

        except Exception:
            # Coverage parsing failed, return empty
            pass

        return executed_lines

    def trace_python_file(self, file_path: Path, args: List[str] = None) -> ExecutionTrace:
        """
        Convenience method to trace execution of a single Python file.

        Args:
            file_path: Path to Python file
            args: Additional arguments to pass to the script

        Returns:
            ExecutionTrace with coverage data
        """
        args = args or []
        command = [str(file_path)] + args
        return self.trace_command(command, cwd=file_path.parent)


def quick_trace(source_file: Path, command: Optional[List[str]] = None) -> ExecutionTrace:
    """
    Quick function to trace a file or command.

    Args:
        source_file: Path to source file or directory
        command: Optional command to run (defaults to running source_file)

    Returns:
        ExecutionTrace
    """
    tracer = ExecutionTracer()

    if command:
        return tracer.trace_command(command, cwd=source_file.parent)
    else:
        return tracer.trace_python_file(source_file)


# For backward compatibility
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tracer.py <python_file> [args...]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    args = sys.argv[2:]

    tracer = ExecutionTracer()
    trace = tracer.trace_python_file(file_path, args)

    print("Execution Trace:")
    print(f"  Success: {trace.success}")
    print(f"  Return code: {trace.return_code}")
    print(f"  Total executed lines: {trace.total_executed_lines}")
    print("\nExecuted lines by file:")
    for file_path, lines in trace.executed_lines.items():
        print(f"  {file_path}: {len(lines)} lines")
