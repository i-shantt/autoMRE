"""
AutoRepro-Min: Automated Bug Reproduction Minimization
Validator Module

This module validates whether a minimized test case still reproduces
the original bug. The validator acts as the oracle for delta debugging.

Based on:
- Zeller & Hildebrandt (2002): Test oracle concept in ddmin
- Jiang et al. (2026): Metamorphic testing for oracle-independent reduction
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class ValidationResult:
    """Result of validating a minimized test case."""
    is_valid: bool  # Does it still reproduce the bug?
    output: str  # stdout + stderr from execution
    return_code: int
    matches_original: bool  # Does output match the original?
    error_type_match: bool  # Does error type match?
    error_message_similarity: float  # 0.0 to 1.0 similarity


@dataclass
class OriginalBehavior:
    """Stores the original behavior for comparison."""
    output: str
    return_code: int
    error_type: Optional[str]  # Exception type or error category
    error_message: Optional[str]
    stack_trace: Optional[str]

    @classmethod
    def from_execution(cls, output: str, return_code: int) -> "OriginalBehavior":
        """
        Parse execution output to extract OriginalBehavior.

        Args:
            output: stdout + stderr from execution
            return_code: Process return code

        Returns:
            OriginalBehavior instance
        """
        error_type = None
        error_message = None
        stack_trace = None

        if return_code != 0:
            # Try to extract error type and message
            error_type, error_message, stack_trace = cls._parse_error(output)

        return cls(
            output=output,
            return_code=return_code,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace
        )

    @staticmethod
    def _parse_error(output: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse Python error output to extract type, message, and stack trace.

        Args:
            output: stderr output containing error

        Returns:
            Tuple of (error_type, error_message, stack_trace)
        """
        lines = output.strip().split('\n')

        error_type = None
        error_message = None
        stack_trace = None

        # Look for the last exception line (format: "ErrorType: message")
        for i, line in enumerate(reversed(lines)):
            match = re.match(r'^(\w+Error|\w+Exception):\s*(.*)$', line)
            if match:
                error_type = match.group(1)
                error_message = match.group(2)

                # Extract stack trace (lines before the error)
                trace_lines = lines[:-i-1] if i > 0 else lines[:-1]
                stack_trace = '\n'.join(trace_lines)
                break

        return error_type, error_message, stack_trace


class Validator:
    """
    Validates whether minimized code still reproduces the original bug.

    Acts as the oracle for delta debugging - determines if a reduction
    preserves the target property (bug reproduction).
    """

    def __init__(self, original_behavior: Optional[OriginalBehavior] = None,
                 timeout: int = 30,
                 match_strategy: str = "error_type"):
        """
        Initialize validator.

        Args:
            original_behavior: Original behavior to match against
            timeout: Maximum execution time in seconds
            match_strategy: How to determine match ("exact", "error_type", "error_message")
        """
        self.original_behavior = original_behavior
        self.timeout = timeout
        self.match_strategy = match_strategy

    def set_original_behavior(self, output: str, return_code: int):
        """
        Set the original behavior from execution results.

        Args:
            output: stdout + stderr from original execution
            return_code: Return code from original execution
        """
        self.original_behavior = OriginalBehavior.from_execution(output, return_code)

    def validate(self, source_code: str, command: Optional[List[str]] = None,
                 cwd: Optional[Path] = None) -> ValidationResult:
        """
        Validate that minimized code reproduces the original behavior.

        Args:
            source_code: Minimized source code to validate
            command: Command to execute (if None, executes as Python script)
            cwd: Working directory for execution

        Returns:
            ValidationResult indicating whether the reduction is valid
        """
        if self.original_behavior is None:
            raise ValueError("Original behavior must be set before validation")

        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False, dir=cwd) as f:
            f.write(source_code)
            temp_file = Path(f.name)

        try:
            # Execute the minimized code
            if command:
                # Use provided command
                cmd = command
            else:
                # Default: run as Python script
                cmd = [sys.executable, str(temp_file)]

            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            output = result.stdout + result.stderr
            return_code = result.returncode

            # Determine if this matches the original behavior
            matches = self._matches_original(output, return_code)

            return ValidationResult(
                is_valid=matches,
                output=output,
                return_code=return_code,
                matches_original=matches,
                error_type_match=self._error_type_matches(output),
                error_message_similarity=self._error_message_similarity(output)
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                is_valid=False,
                output="Validation timeout",
                return_code=-1,
                matches_original=False,
                error_type_match=False,
                error_message_similarity=0.0
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                output=f"Validation error: {str(e)}",
                return_code=-1,
                matches_original=False,
                error_type_match=False,
                error_message_similarity=0.0
            )
        finally:
            # Cleanup temp file
            try:
                temp_file.unlink()
            except:
                pass

    def _matches_original(self, output: str, return_code: int) -> bool:
        """
        Check if output matches original behavior.

        Args:
            output: Execution output
            return_code: Execution return code

        Returns:
            True if behavior matches original
        """
        if self.original_behavior is None:
            return False

        if self.match_strategy == "exact":
            # Exact match on output and return code
            return (output == self.original_behavior.output and
                    return_code == self.original_behavior.return_code)

        elif self.match_strategy == "error_type":
            # Match on error type (most permissive)
            orig_error = self.original_behavior.error_type
            current_error, _, _ = OriginalBehavior._parse_error(output)

            if orig_error is None:
                # Original succeeded - current must also succeed
                return return_code == 0

            # Original failed - current must fail with same error type
            return current_error == orig_error

        elif self.match_strategy == "error_message":
            # Match on error type and similar message
            orig_error = self.original_behavior.error_type
            orig_message = self.original_behavior.error_message or ""
            current_error, current_message, _ = OriginalBehavior._parse_error(output)

            if orig_error is None:
                return return_code == 0

            if current_error != orig_error:
                return False

            # Check message similarity
            if orig_message and current_message:
                similarity = self._text_similarity(orig_message, current_message)
                return similarity > 0.5  # At least 50% similar

            return True

        return False

    def _error_type_matches(self, output: str) -> bool:
        """Check if error type matches original."""
        if self.original_behavior is None:
            return False

        orig_error = self.original_behavior.error_type
        current_error, _, _ = OriginalBehavior._parse_error(output)

        return orig_error == current_error

    def _error_message_similarity(self, output: str) -> float:
        """Calculate similarity between error messages."""
        if self.original_behavior is None:
            return 0.0

        orig_message = self.original_behavior.error_message or ""
        _, current_message, _ = OriginalBehavior._parse_error(output)
        current_message = current_message or ""

        return self._text_similarity(orig_message, current_message)

    @staticmethod
    def _text_similarity(text1: str, text2: str) -> float:
        """
        Calculate simple text similarity (Jaccard on words).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0.0 to 1.0
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


class PropertyTester:
    """
    Generic property tester for delta debugging.

    Wraps a user-provided test function to determine if a reduction
    preserves the target property.
    """

    def __init__(self, test_func: Callable[[str], bool]):
        """
        Initialize with a test function.

        Args:
            test_func: Function that takes source code and returns True
                      if the property is preserved
        """
        self.test_func = test_func

    def test(self, source_code: str) -> bool:
        """
        Test if source code preserves the property.

        Args:
            source_code: Source code to test

        Returns:
            True if property is preserved
        """
        try:
            return self.test_func(source_code)
        except Exception:
            return False


# For backward compatibility
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validator.py <python_file>")
        sys.exit(1)

    # Example: validate that a file still raises an error
    file_path = Path(sys.argv[1])

    # First, capture original behavior
    result = subprocess.run([sys.executable, str(file_path)],
                           capture_output=True, text=True)

    validator = Validator()
    validator.set_original_behavior(result.stdout + result.stderr,
                                    result.returncode)

    print(f"Original behavior captured:")
    print(f"  Error type: {validator.original_behavior.error_type}")
    print(f"  Error message: {validator.original_behavior.error_message}")
    print(f"  Return code: {validator.original_behavior.return_code}")
