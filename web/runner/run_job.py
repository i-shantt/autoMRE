"""Run one reduction to completion, reporting through files.

This is the half of the web service that executes inside a sandbox. It is
a plain command-line program on purpose: the thing that starts it cannot
stay on the phone for twenty minutes, so there is no request to hold open
and no server to run. It writes `status.json` as it goes, and on its way
out it writes the result, a manifest and a final status — in that order,
so that "a final status exists" always implies "the result exists".

Every exit path flushes. A job that is cancelled, runs out of budget or
raises still packages whatever has been reduced so far, because the
reducer only ever commits validated candidates: the tree at a query
boundary is a project that genuinely still reproduces the failure, just
less reduced than it was going to be. Throwing that away because the
clock ran out would discard a correct answer.

The reduction itself is `webcore.run_pipeline`, the same engine the
FastAPI worker drives. Nothing about *how a reduction works* lives here.

Usage:
    run_job.py --job-id ID --archive PATH --command "pytest tests/x.py::y"
               --sink DIR [--budget 1380] [--cancel-file PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "automre" / "src"))

from webcore import (  # noqa: E402
    JobError,
    JobState,
    count_python,
    package,
    project_root,
    run_pipeline,
    safe_extract,
)

# Time held back from the budget so the flush below always has room. A
# reduction stopped at its deadline still has to zip a tree, count it and
# write three files; if the sandbox is killed during that, the job looks
# lost rather than partial.
DEFAULT_FLUSH_RESERVE = 120

# The smallest gap between status writes. A progress event fires once per
# query — thousands of times — and the reader polls every second or two,
# so writing every event would be pure disk churn. Phase and status
# changes bypass this: they are what the reader is actually waiting for.
STATUS_MIN_INTERVAL = 1.0


class Reporter:
    """Writes status.json, atomically and not too often."""

    def __init__(self, sink: Path):
        self.sink = sink
        self.path = sink / "status.json"
        self._last_write = 0.0
        self._last_key: Optional[tuple] = None

    def write(self, state: JobState, *, force: bool = False) -> None:
        snap = state.snapshot()
        snap["updated_at"] = time.time()
        # A reader that finds status.json stale can tell the difference
        # between "working" and "died" only if this moves, so it is
        # stamped on every write rather than derived from mtime, which
        # does not survive being copied out of the sandbox.
        key = (state.status, state.phase, state.message)
        now = time.time()
        if not force and key == self._last_key:
            if now - self._last_write < STATUS_MIN_INTERVAL:
                return
        self._last_key = key
        self._last_write = now
        _write_json_atomic(self.path, snap)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write to a temp file in the same directory, then rename over.

    A reader polls this file while it is being written. Without the
    rename it would eventually read half a JSON document and conclude the
    job had failed.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _rel(path: Path, root: Path) -> str:
    """Path relative to the project root, resolving both sides first.

    Both halves have to be resolved or this silently yields absolute
    paths: the reducer resolves the paths it reports, and on macOS a
    temp directory is handed out as /var/folders/... while resolving to
    /private/var/folders/..., so `relative_to` raises and every entry in
    the manifest comes out absolute — which leaks a sandbox path to the
    user and makes the viewer unable to match anything.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_manifest(job_id: str, state: JobState, root: Path,
                   command: List[str], summary: Any) -> Dict[str, Any]:
    """Per-file account of what the reduction did.

    The reducer already knows all of this and the web service has been
    throwing it away, keeping two numbers out of a dozen. Without it a
    user downloads a zip and has to diff it against their own project to
    learn anything.

    `root_prefix` is load-bearing. `project_root` strips a single
    wrapping directory from the upload, so without recording which one,
    every path here is off by one directory against the archive the user
    sent and a viewer cannot line the two sides up.
    """
    files: List[Dict[str, Any]] = []

    def add(path: Path, status: str, before: Optional[int],
            after: Optional[int]) -> None:
        files.append({"path": _rel(path, root), "status": status,
                      "before_lines": before, "after_lines": after})

    if summary is not None:
        # Protected files first, and from `protected_files` rather than
        # from `per_file_reduction`. They are excluded from the reducer's
        # file walk by definition, so they never appear in the walk's
        # results — listing them only from there dropped the one file a
        # reader most wants explained, the test itself.
        protected = set()
        for p in summary.protected_files:
            rel = _rel(p, root)
            protected.add(rel)
            try:
                n = len(p.read_text().splitlines())
            except (OSError, UnicodeDecodeError):
                n = None
            # Identical on both sides on purpose: this file is the
            # question the reduction was answering.
            files.append({"path": rel, "status": "protected",
                          "before_lines": n, "after_lines": n})

        gone = {"unreachable": summary.unreachable_deleted,
                "imported-only": summary.imported_deleted}
        for why, paths in gone.items():
            for p in paths:
                files.append({"path": _rel(p, root), "status": "deleted",
                              "reason": why, "before_lines": None,
                              "after_lines": 0})
        for p in summary.inlined_away:
            add(p, "inlined", None, 0)
        for p, (before, after) in summary.per_file_reduction.items():
            rel = _rel(p, root)
            if rel in protected:
                continue
            if after == 0:
                add(p, "deleted", before, 0)
            elif after < before:
                add(p, "reduced", before, after)
            else:
                add(p, "unchanged", before, after)
        for p in summary.undecodable_files:
            add(p, "untouchable", None, None)
        for p in summary.outside_project_files:
            add(p, "outside-project", None, None)
        for rel, err in getattr(summary, "reduction_errors", {}).items():
            files.append({"path": str(rel), "status": "errored",
                          "error": str(err),
                          "before_lines": None, "after_lines": None})

    return {
        "job_id": job_id,
        "command": list(command),
        "status": state.status,
        "root_prefix": root.name,
        "stats": state.snapshot(),
        "files": sorted(files, key=lambda f: f["path"]),
        "truncated": False,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--archive", required=True,
                    help="path to the uploaded .zip")
    ap.add_argument("--command", required=True,
                    help='reproduction command, e.g. "pytest tests/t.py::k"')
    ap.add_argument("--sink", required=True,
                    help="directory to write status.json, reduced.zip, "
                         "manifest.json and status-final.json into")
    ap.add_argument("--budget", type=int, default=1380,
                    help="seconds for the whole job, flush reserve included")
    ap.add_argument("--flush-reserve", type=int, default=DEFAULT_FLUSH_RESERVE)
    ap.add_argument("--cancel-file", default=None,
                    help="job stops when this path appears")
    ap.add_argument("--query-timeout", type=int, default=120)
    args = ap.parse_args(argv)

    sink = Path(args.sink).resolve()
    sink.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(sink)

    command = shlex.split(args.command)
    state = JobState(job_id=args.job_id, command=command)
    state.started_at = time.time()
    reporter.write(state, force=True)

    cancel_file = Path(args.cancel_file) if args.cancel_file else None

    def should_cancel() -> bool:
        return cancel_file is not None and cancel_file.exists()

    # The reducer's deadline stops short of the sandbox's, so the flush
    # below is inside the budget rather than racing it.
    reduce_deadline = state.started_at + max(
        1, args.budget - args.flush_reserve)

    tmp = Path(tempfile.mkdtemp(prefix=f"automre_{args.job_id}_"))
    root: Optional[Path] = None
    summary = None
    try:
        state.status = "preparing"
        state.message = "unpacking the project"
        reporter.write(state, force=True)

        extracted = tmp / "src"
        extracted.mkdir()
        # Validated before anything is provisioned, so a hostile or
        # oversized archive costs a few seconds rather than a venv.
        safe_extract(Path(args.archive), extracted)
        root = project_root(extracted)

        outcome = run_pipeline(
            root, command,
            venv_dir=tmp / "venv",
            state=state,
            on_update=reporter.write,
            should_cancel=should_cancel,
            deadline=reduce_deadline,
            query_timeout=args.query_timeout,
        )
        summary = outcome.summary

        if outcome.cancelled:
            if should_cancel():
                state.status = "cancelled"
                state.message = ("cancelled; the partial result is still "
                                 "available")
            else:
                # Out of budget, not the user's choice. Reported apart
                # because the two need different words: one is a decision,
                # the other is us running out of room and owing an
                # explanation.
                state.status = "timed_out"
                state.message = (
                    "stopped at the time limit; everything reduced so far "
                    "is verified and included")
        else:
            state.status = "done"

    except JobError as exc:
        state.status = "failed"
        state.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        state.status = "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        # The flush. Deliberately outside every except above, so a partial
        # tree survives a cancel, a timeout and a crash alike.
        try:
            if root is not None and root.exists():
                state.final_files, state.final_lines = count_python(root)
                package(sink / "reduced.zip", root)
                _write_json_atomic(
                    sink / "manifest.json",
                    build_manifest(args.job_id, state, root, command, summary))
            state.finished_at = time.time()
            reporter.write(state, force=True)
            # Written last: a reader that sees this file can rely on the
            # zip and the manifest already being there.
            _write_json_atomic(sink / "status-final.json", state.snapshot())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return 0 if state.status in {"done", "cancelled", "timed_out"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
