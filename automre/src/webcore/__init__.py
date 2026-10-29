"""Job mechanics shared by every front door onto the reducer.

There are two: the FastAPI worker in `web/worker/`, which runs a job on a
thread and keeps it in memory, and the runner that executes inside a
sandbox and reports through a file. They differ in where state lives and
how cancellation is signalled, and in nothing else — so everything else
lives here.

The split exists because the two were about to be written twice. The
progress projection alone (see `state.ProgressProjector`) has four
non-obvious rules, and two copies of it would agree on the day they were
written and not a week later.

Nothing here may import `fastapi`, or anything from `web/`.
"""

from __future__ import annotations

from .archive import (
    MAX_ARCHIVE_BYTES,
    MAX_FILES,
    MAX_UNCOMPRESSED_BYTES,
    count_python,
    project_root,
    safe_extract,
)
from .errors import JobError
from .pipeline import (
    DEFAULT_QUERY_TIMEOUT,
    PipelineOutcome,
    check_ready,
    package,
    prepare_env,
    resolve_command,
    run_pipeline,
)
from .state import ETA_HIGH, ETA_LOW, JobState, ProgressProjector

__all__ = [
    "DEFAULT_QUERY_TIMEOUT",
    "ETA_HIGH",
    "ETA_LOW",
    "MAX_ARCHIVE_BYTES",
    "MAX_FILES",
    "MAX_UNCOMPRESSED_BYTES",
    "JobError",
    "JobState",
    "PipelineOutcome",
    "ProgressProjector",
    "check_ready",
    "count_python",
    "package",
    "prepare_env",
    "project_root",
    "resolve_command",
    "run_pipeline",
    "safe_extract",
]
