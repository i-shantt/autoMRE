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
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, Iterator, List, Optional

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
    duration: float = 0.0


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

    # ---- query timeout policy
    #
    # A reduction asks the same question thousands of times and the answers
    # cluster tightly: on tomlkit-write_backslash the median query is
    # 0.137 s and p99 is 0.404 s. A query that runs for two minutes is not
    # a slow query, it is a different event — most often a stubbed body
    # that made a `while` loop non-terminating, which a parser repo
    # produces easily. Waiting the user's full timeout for it buys nothing:
    # ten such queries out of 2,699 were 76% of that task's entire query
    # time.
    #
    # So `timeout` becomes a ceiling and the working limit is derived from
    # what this project's queries actually cost, floored so a fast project
    # still gets real headroom. The verdict on a timeout does not change —
    # see `_oracle_matches`.
    TIMEOUT_FLOOR = 10.0
    TIMEOUT_MULTIPLE = 20.0
    # Calibrate on the slowest of a recent window rather than the slowest
    # ever seen. A running maximum only ratchets up, so one unusually slow
    # query — a cold import, a machine hiccup — would restore the old
    # ceiling for the rest of the run and quietly undo this.
    TIMEOUT_WINDOW = 64

    def __init__(self, command: List[str], project_dir: Path,
                 timeout: int = 60,
                 match_strategy: str = "output_match"):
        self.command = list(command)
        self.project_dir = Path(project_dir).resolve()
        self.timeout = timeout
        # Durations of the most recent completed runs. Calibrates the
        # limit below; empty until the first query returns, when the
        # ceiling applies.
        self._recent: Deque[float] = deque(maxlen=self.TIMEOUT_WINDOW)
        # Queries killed by the limit. Counted because a timeout is
        # otherwise indistinguishable from an ordinary rejection, which is
        # how the cost above went unnoticed for so long.
        self.timed_out_queries = 0
        # Every oracle question in a reduction comes through _run, so this
        # is the one place a live query count can be exact. The debugger's
        # own tally is assembled from what each phase reports and is only
        # correct once that phase returns.
        self.queries_run = 0
        # Optional observer, called with this validator after every query.
        # Used to drive progress reporting without threading a callback
        # through six phases.
        self.on_query: Optional[Callable[["MultiFileValidator"], None]] = None
        # Delegate output matching to the existing single-file Validator so
        # we get identical error-type / message semantics.
        self._oracle = Validator(match_strategy=match_strategy,
                                 timeout=timeout)

    @property
    def query_timeout(self) -> float:
        """Seconds a single query is allowed, given what they have cost.

        Note this is machine-dependent where the old fixed constant was
        not. It does not make runs less reproducible in practice: with a
        10 s floor against a sub-second query, the only thing that crosses
        the limit is a non-terminating candidate, and those do not finish
        on a faster machine either.
        """
        if not self._recent:
            return float(self.timeout)
        return min(float(self.timeout),
                   max(self.TIMEOUT_FLOOR,
                       self.TIMEOUT_MULTIPLE * max(self._recent)))

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
        result = self._run_once()
        self.queries_run += 1
        if self.on_query is not None:
            try:
                self.on_query(self)
            except Exception:
                # An observer is for reporting. It must never be able to
                # change the outcome of a reduction, including by failing.
                pass
        return result

    def _run_once(self) -> ProjectRunResult:
        limit = self.query_timeout
        started = time.monotonic()
        try:
            proc = subprocess.run(
                self.command,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=limit,
                env=oracle_env(),
            )
            elapsed = time.monotonic() - started
            self._recent.append(elapsed)
            return ProjectRunResult(
                output=(proc.stdout or "") + (proc.stderr or ""),
                return_code=proc.returncode,
                duration=elapsed,
            )
        except subprocess.TimeoutExpired:
            self.timed_out_queries += 1
            return ProjectRunResult(
                output=(f"Timeout: reproduction command exceeded "
                        f"{limit:.1f}s"),
                return_code=-1,
                timed_out=True,
                duration=limit,
            )
        except FileNotFoundError as exc:
            return ProjectRunResult(output=f"Command not found: {exc}",
                                    return_code=-1)

    def _oracle_matches(self, run: ProjectRunResult) -> bool:
        if run.timed_out:
            # A timed-out candidate is one whose tree does not reproduce
            # the bug in bounded time, so refusing it keeps the code — the
            # conservative verdict, and the correct one. A tighter limit
            # can cost reduction; it cannot make the oracle unsound.
            #
            # This is deliberately *not* treated as "unknown, retry
            # later". The usual cause is a candidate that made a loop
            # non-terminating, and it will hang again.
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
