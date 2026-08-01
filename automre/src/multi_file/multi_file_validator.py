"""
autoMRE: Multi-File Reduction
Whole-Project Validator

Runs a reproduction command against a full project directory and compares
the result with a captured original behavior (error type / return code /
output). Provides in-memory file snapshot/restore helpers so callers can
speculatively mutate files and roll back on failure.

Also exposes ProjectFileValidator — a Validator subclass that plugs into
the existing HDD-E reducer, but writes candidate source to a *specific*
project file and runs the whole-project reproduction command instead of
executing a temp file in isolation.
"""

from __future__ import annotations

import subprocess
import sys as _sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from validator import (
    OriginalBehavior,
    ValidationResult,
    Validator,
    oracle_env,
)


@dataclass
class ProjectRunResult:
    """Result of executing the reproduction command on the project."""
    output: str
    return_code: int
    timed_out: bool = False


class MultiFileValidator:
    """Runs the reproduction command against a project directory.

    Typical usage:

        v = MultiFileValidator(command, project_dir)
        v.capture_original()          # establishes the oracle
        with v.snapshot([file_a]):    # remember file contents
            file_a.unlink()
            if not v.validate():      # bug disappeared
                # snapshot context restores file_a on exit
                raise SomeRollback
    """

    def __init__(self, command: List[str], project_dir: Path,
                 timeout: int = 60,
                 match_strategy: str = "output_match"):
        self.command = list(command)
        self.project_dir = Path(project_dir).resolve()
        self.timeout = timeout
        # Delegate output matching to the existing single-file Validator so
        # we get identical error-type / message semantics.
        self._oracle = Validator(match_strategy=match_strategy,
                                 timeout=timeout)

    # ------------------------------------------------------ capture

    def capture_original(self) -> ProjectRunResult:
        """Run the reproduction command once and record the oracle."""
        run = self._run()
        self._oracle.set_original_behavior(run.output, run.return_code)
        return run

    def set_original_from(self, output: str, return_code: int) -> None:
        """Seed the oracle when the caller already has an original trace."""
        self._oracle.set_original_behavior(output, return_code)

    @property
    def original_behavior(self) -> Optional[OriginalBehavior]:
        return self._oracle.original_behavior

    # ------------------------------------------------------ validation

    def validate(self) -> bool:
        """Return True iff the bug still reproduces on the current tree."""
        run = self._run()
        return self._oracle_matches(run)

    def validate_detailed(self) -> ValidationResult:
        run = self._run()
        matches = self._oracle_matches(run)
        return ValidationResult(
            is_valid=matches,
            output=run.output,
            return_code=run.return_code,
            matches_original=matches,
            error_type_match=self._oracle._error_type_matches(run.output),
            error_message_similarity=self._oracle._error_message_similarity(
                run.output),
        )

    # ------------------------------------------------------ snapshot

    @contextmanager
    def snapshot(self, files: List[Path]) -> Iterator[Dict[Path, Optional[str]]]:
        """Snapshot file contents; restore on exit if `restore_on_exit` set.

        The context yields a dict mapping path -> original content (or None
        if the file didn't exist). The caller can decide whether to keep the
        mutation (call `commit()` — implemented as clearing the dict) or
        allow the automatic rollback.
        """
        saved: Dict[Path, Optional[str]] = {}
        for f in files:
            f = Path(f)
            saved[f] = f.read_text() if f.exists() else None

        rollback = {"active": True}

        class _Snap(dict):
            def commit(self_inner):
                rollback["active"] = False

        snap = _Snap(saved)
        try:
            yield snap
        finally:
            if rollback["active"]:
                for path, content in saved.items():
                    if content is None:
                        if path.exists():
                            try:
                                path.unlink()
                            except OSError:
                                pass
                    else:
                        path.write_text(content)

    # ------------------------------------------------------ internals

    def _run(self) -> ProjectRunResult:
        try:
            proc = subprocess.run(
                self.command,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=oracle_env(),
            )
            return ProjectRunResult(
                output=(proc.stdout or "") + (proc.stderr or ""),
                return_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ProjectRunResult(
                output="Timeout: reproduction command exceeded time limit",
                return_code=-1,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return ProjectRunResult(output=f"Command not found: {exc}",
                                    return_code=-1)

    def _oracle_matches(self, run: ProjectRunResult) -> bool:
        if run.timed_out:
            return False
        return self._oracle._matches_original(run.output, run.return_code)


class ProjectFileValidator(Validator):
    """Adapter so single-file HDD-E can validate against the whole project.

    HDD-E hands its Validator a candidate source string and expects a
    boolean answer. This adapter writes the string to a *specific* target
    file inside the real project tree, then delegates to a
    MultiFileValidator to run the actual reproduction command. If the run
    doesn't match the original behavior, the caller's snapshot machinery
    is expected to roll back — but for HDD-E's inner loop we restore the
    file ourselves after every query so a failing candidate never leaks
    into subsequent iterations.
    """

    def __init__(self, project_validator: MultiFileValidator,
                 target_file: Path):
        super().__init__(project_validator.original_behavior,
                         timeout=project_validator.timeout,
                         match_strategy=project_validator._oracle.match_strategy)
        self._project = project_validator
        self._target = Path(target_file)

    def validate(self, source_code: str,
                 command: Optional[List[str]] = None,
                 cwd: Optional[Path] = None) -> ValidationResult:
        # `command` and `cwd` are ignored — the whole-project validator
        # already knows how to run the reproduction command.
        original = self._target.read_text() if self._target.exists() else None
        self._target.write_text(source_code)
        try:
            return self._project.validate_detailed()
        finally:
            # Always restore between candidates. HDD-E carries the
            # accepted source in memory; the debugger writes the final
            # version to disk once the reducer returns.
            if original is None:
                if self._target.exists():
                    try:
                        self._target.unlink()
                    except OSError:
                        pass
            else:
                self._target.write_text(original)
