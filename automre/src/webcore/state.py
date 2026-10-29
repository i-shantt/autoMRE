"""What a job looks like from outside, and how progress updates it.

The progress model is the part worth explaining. A reduction's unit of
work is the *query* — one run of the reproduction command — and the number
of queries a project needs is not knowable in advance, because it depends
on how much of the code turns out to be removable. So the estimate is a
calibrated guess (see MultiFileDebugger) and this module turns it into a
*range* rather than a countdown. An honest wide range is more useful than
a precise-looking number that is wrong.

`JobState` holds only what can be serialized, so the same fields describe
a job whether they live in a worker's memory or in a `status.json` inside
a sandbox. Anything tied to one machine's disk — a work directory, a
result zip, a cancellation Event — belongs to the thing that owns the
disk, which is why `web/worker/jobs.py` subclasses this rather than the
other way round.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

# How far the ETA range is spread either side of the point estimate. The
# per-line query factor is calibrated across repos that differ by a factor
# of four (requests 0.11 q/line, tomlkit 0.48), so a tight range would be
# a lie.
ETA_LOW = 0.55
ETA_HIGH = 2.0


@dataclass
class JobState:
    # job_id and command come first and carry no defaults, deliberately.
    # Subclasses add defaulted fields, and a dataclass forbids a
    # non-default field after a defaulted one — so putting anything with a
    # default ahead of these would make `Job(job_id=..., command=[...])`
    # impossible to construct positionally or by keyword.
    job_id: str
    command: List[str]
    status: str = "queued"      # queued|preparing|running|done|failed|cancelled
    phase: str = "queued"
    message: str = "waiting to start"
    queries: int = 0
    estimated_queries: int = 0
    timed_out_queries: int = 0
    # Lines to examine, and lines examined. This is the honest progress
    # measure: the reducer walks the surviving files once and this is a
    # real fraction of that walk. Queries are shown too, but only as a
    # count of work done — the number a project *will* need varies
    # eightfold per line and cannot carry a progress bar.
    work_total: int = 0
    work_done: int = 0
    work_started_at: float = 0.0
    current_file: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Optional[str] = None

    original_files: int = 0
    original_lines: int = 0
    final_files: int = 0
    final_lines: int = 0
    # TODO(phase 8): this should be Optional and default to None. A
    # cancelled job never produces a summary, so "0 protected lines" gets
    # rendered as though it were measured — a lie that looks like data.
    # Left as-is here so this extraction changes no observable behavior.
    protected_lines: int = 0

    # What one run of the reproduction command costs, measured by the gate.
    # With `estimated_queries` this is what prices a reduction before it
    # starts: see `projected_seconds`.
    baseline_seconds: float = 0.0

    # -------------------------------------------------- derived views

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def fraction(self) -> Optional[float]:
        """How far through the file walk we are, or None before it starts."""
        if self.work_total <= 0:
            return None
        return min(1.0, self.work_done / self.work_total)

    @property
    def projected_seconds(self) -> Optional[float]:
        """A worst-case ceiling on the reduction, not a prediction.

        queries x one run of the command. Both halves are known early: the
        gate measures the run time before the first query, and the reducer
        estimates the query count after its first phase.

        **It is not accurate and must not be shown as a clock.** Measured
        against two real runs it missed in both directions:

            26-line project:  projected   0.3s, actual  1.4s  (x0.24)
            requests, 11209:  projected 2458s,  actual  367s  (x6.71)

        A 28x swing between two points. Both errors have causes, and
        neither is going away:

        * `estimated_queries` counts the per-file walk, which is ~97% of
          queries on a large project and a minority of them on a tiny one,
          so it under-counts small jobs and over-counts large ones (2038
          estimated against 1038 actual on requests).
        * `baseline_seconds` times a run against the *whole* tree, while
          almost every query runs against a tree that is already mostly
          deleted. On requests the real mean was 0.353s against a 1.206s
          baseline — 3.4x out, and that gap widens the more there is to
          remove.

        So it is used for one thing: refusing work that is obviously too
        big, where being 7x pessimistic is survivable and the alternative
        is a user waiting out a budget for nothing. It is not what tells
        anyone how long to wait — `eta_range` does that, extrapolated from
        work actually completed, and it is the only estimate here measured
        against progress rather than guessed before it.
        """
        if not self.baseline_seconds or not self.estimated_queries:
            return None
        return self.estimated_queries * self.baseline_seconds

    def eta_range(self):
        """(low, high) seconds remaining, or None when it cannot be said.

        Extrapolates from lines-examined-per-second, measured over the file
        walk only — the phases before it are a fixed prelude whose cost
        says nothing about the rest.

        Stays None until a real fraction of the work is behind us. The
        first file is the largest (the reducer sorts that way), so an
        estimate drawn from it alone would be systematically pessimistic;
        and a wide range that appears late beats a confident one that
        appears immediately and is wrong.
        """
        if self.status != "running":
            return None
        frac = self.fraction
        if frac is None or frac <= 0.03:
            return None
        spent = time.time() - self.work_started_at if self.work_started_at else 0
        if spent <= 0:
            return None
        point = spent * (1 - frac) / frac
        return (point * ETA_LOW, point * ETA_HIGH)

    def snapshot(self) -> Dict[str, Any]:
        eta = self.eta_range()
        projected = self.projected_seconds
        return {
            "job_id": self.job_id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "queries": self.queries,
            "estimated_queries": self.estimated_queries,
            "timed_out_queries": self.timed_out_queries,
            "work_total": self.work_total,
            "work_done": self.work_done,
            "fraction": self.fraction,
            "current_file": self.current_file,
            "elapsed_seconds": round(self.elapsed, 1),
            "eta_low_seconds": round(eta[0]) if eta else None,
            "eta_high_seconds": round(eta[1]) if eta else None,
            "error": self.error,
            "original_files": self.original_files,
            "original_lines": self.original_lines,
            "final_files": self.final_files,
            "final_lines": self.final_lines,
            "protected_lines": self.protected_lines,
            "baseline_seconds": round(self.baseline_seconds, 3) or None,
            # Named a ceiling, not a projection, so nothing downstream
            # renders it as "time remaining". See projected_seconds: it
            # was 6.7x high on requests and 4x low on a tiny project.
            # eta_range is the number fit to show a waiting user.
            "projected_ceiling_seconds": (round(projected) if projected
                                          else None),
        }

    # ------------------------------------------------ serialization

    def to_dict(self) -> Dict[str, Any]:
        """Every stored field, for writing to a file.

        Not `snapshot()`: that is the view an HTTP client gets, with
        derived values folded in. This is the state itself, so that
        `from_dict(to_dict(x))` round-trips.
        """
        out = asdict(self)
        out["command"] = list(self.command)
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in out.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobState":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class ProgressProjector:
    """Folds a reducer progress event into a JobState.

    This exists as one named thing because it is the most drift-prone
    logic in the system and there are now two callers. The subtleties are
    not obvious from the field names: `work_started_at` is latched once
    and only once, `estimated_queries` is tested for truthiness rather
    than presence because the reducer emits 0 before it knows, and the
    same is true of `work_total` and `current_file`. Two copies of this
    would agree on the day they were written and not a week later.

    It deliberately does *not* decide anything about cancellation. The
    worker tests a threading.Event and the sandbox runner tests a sentinel
    file, and raising out of a progress callback is a choice the caller has
    to make, not this.
    """

    @staticmethod
    def apply(state: JobState, event: Dict[str, Any]) -> JobState:
        state.phase = event.get("phase", state.phase)
        state.message = event.get("message", state.message)
        if "queries" in event:
            state.queries = event["queries"]
        if event.get("estimated_queries"):
            state.estimated_queries = event["estimated_queries"]
        if "timed_out_queries" in event:
            state.timed_out_queries = event["timed_out_queries"]
        if event.get("work_total"):
            if not state.work_started_at:
                # The clock for the ETA starts when the file walk does, not
                # when the job did. Everything before it — unpacking, pip
                # install, the coverage trace — is a fixed prelude that
                # would skew the rate.
                state.work_started_at = time.time()
            state.work_total = event["work_total"]
        if "work_done" in event:
            state.work_done = event["work_done"]
        if event.get("current_file"):
            state.current_file = event["current_file"]
        return state
