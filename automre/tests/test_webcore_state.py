"""JobState: the progress rules, and the honesty of the cost ceiling.

These are unit tests on purpose. The projection and the ETA are the two
numbers a waiting user reads, one of them is unreliable by a factor of
seven, and the difference between them is the kind of thing that gets
quietly lost in a refactor.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from webcore import JobState, ProgressProjector  # noqa: E402


def _state(**kw) -> JobState:
    return JobState(job_id="j", command=["pytest", "t.py::k"], **kw)


# ----------------------------------------------------------- projection

def test_the_cost_ceiling_needs_both_halves():
    """Neither number alone says anything, so the answer is None."""
    assert _state().projected_seconds is None
    assert _state(baseline_seconds=1.2).projected_seconds is None
    assert _state(estimated_queries=2038).projected_seconds is None


def test_the_cost_ceiling_is_queries_times_one_run():
    state = _state(baseline_seconds=1.206, estimated_queries=2038)
    assert state.projected_seconds == 2038 * 1.206


def test_the_ceiling_is_not_called_a_projection_in_the_snapshot():
    """It was 6.7x high on requests; a UI must not print it as an ETA.

    Measured: requests at 11,209 lines projected 2,458s against an actual
    367s, while a 26-line project projected 0.3s against an actual 1.4s.
    The name is the guard rail — anything reading `projected_seconds` off
    a snapshot and rendering "time remaining" is reading a field that
    does not exist.
    """
    snap = _state(baseline_seconds=1.206, estimated_queries=2038).snapshot()
    assert "projected_seconds" not in snap
    assert snap["projected_ceiling_seconds"] == round(2038 * 1.206)


def test_the_eta_is_the_number_fit_to_show_and_waits_for_real_progress():
    """The ETA extrapolates from work done, which is why it is trusted.

    It stays silent until a real fraction of the walk is behind it: the
    reducer takes the largest file first, so an estimate drawn from the
    start is systematically pessimistic.
    """
    state = _state(status="running", work_total=1000, work_done=5)
    state.work_started_at = time.time() - 10
    assert state.eta_range() is None, "spoke too early"

    state.work_done = 500
    low, high = state.eta_range()
    assert 0 < low < high
    # The spread is deliberately wide: the per-line query factor varies
    # fourfold across repos, so a tight range would be a lie.
    assert high > low * 3


def test_the_eta_is_silent_unless_running():
    state = _state(status="done", work_total=1000, work_done=500)
    state.work_started_at = time.time() - 10
    assert state.eta_range() is None


# ------------------------------------------------------------- progress

def test_the_work_clock_is_latched_exactly_once():
    """The ETA's clock starts at the file walk, not at the job.

    Unpacking, pip install and the coverage trace are a fixed prelude.
    Re-latching on a later event would restart the clock mid-walk and
    make the rate — and so the ETA — wrong for the rest of the run.
    """
    state = _state()
    ProgressProjector.apply(state, {"work_total": 100})
    first = state.work_started_at
    assert first > 0

    time.sleep(0.01)
    ProgressProjector.apply(state, {"work_total": 120, "work_done": 10})
    assert state.work_started_at == first
    assert state.work_total == 120


def test_zero_estimates_do_not_overwrite_a_real_one():
    """The reducer emits 0 before it knows, hence truthiness not presence.

    Testing `"estimated_queries" in event` instead would let a later
    event reset a known estimate back to zero, and the progress bar would
    jump back to indeterminate partway through a run.
    """
    state = _state()
    ProgressProjector.apply(state, {"estimated_queries": 2038})
    ProgressProjector.apply(state, {"estimated_queries": 0})
    assert state.estimated_queries == 2038

    ProgressProjector.apply(state, {"work_total": 500})
    ProgressProjector.apply(state, {"work_total": 0})
    assert state.work_total == 500

    ProgressProjector.apply(state, {"current_file": "mylib.py"})
    ProgressProjector.apply(state, {"current_file": None})
    assert state.current_file == "mylib.py"


def test_counts_that_can_legitimately_be_zero_are_tested_for_presence():
    """The mirror image of the rule above, and it has to be the other way.

    `queries` and `work_done` start at zero and a zero is real data, so
    these use presence. Using truthiness here would drop the first
    event of every run.
    """
    state = _state(queries=5, work_done=5)
    ProgressProjector.apply(state, {"queries": 0, "work_done": 0})
    assert state.queries == 0
    assert state.work_done == 0


def test_progress_does_not_decide_anything_about_cancellation():
    """Cancellation is the caller's call, and the two callers differ.

    The worker tests a threading.Event, the sandbox runner tests for a
    sentinel file. If this ever raises or sets a flag of its own, one of
    them is wrong.
    """
    state = _state()
    ProgressProjector.apply(state, {"phase": "reduce-files", "queries": 9})
    assert state.status == "queued"


# -------------------------------------------------------- round-tripping

def test_state_survives_a_trip_through_json():
    """status.json is how a sandbox reports, so this has to hold."""
    state = _state(status="running", queries=42, baseline_seconds=1.206,
                   estimated_queries=2038)
    state.current_file = "mylib.py"
    back = JobState.from_dict(state.to_dict())
    assert back == state


def test_from_dict_ignores_fields_it_does_not_know():
    """A status file written by a newer runner must not crash an older one."""
    data = _state(queries=7).to_dict()
    data["something_invented_later"] = True
    assert JobState.from_dict(data).queries == 7
