"""A per-file account of what a reduction did.

The reducer has always known this and the web service has always thrown
it away. `MultiFileReductionResult` reports `per_file_reduction`,
`unreachable_deleted`, `imported_deleted`, `inlined_away`,
`protected_files`, `undecodable_files` and `reduction_errors`; the job
layer kept two numbers out of a dozen, so a user downloaded a zip and had
to diff it against their own project to find out anything.

This is not new analysis. It is the existing result, written down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .state import JobState

# Statuses a file can end up with, as the viewer understands them.
#   deleted         the file is gone
#   inlined         its contents were folded into an importer, then it went
#   reduced         it survived, smaller
#   unchanged       it survived at the same size
#   protected       never a candidate: the test being preserved, or a conftest
#   untouchable     not valid UTF-8, so the reducer would not risk its bytes
#   outside-project reached through a symlink leaving the tree
#   errored         its own reduction raised; the run continued without it
FILE_STATUSES = ("deleted", "inlined", "reduced", "unchanged", "protected",
                 "untouchable", "outside-project", "errored")


def relative_to_root(path: Path, root: Path) -> str:
    """Path relative to the project root, resolving both sides first.

    Both halves have to be resolved or this silently yields absolute
    paths: the reducer resolves the paths it reports, and on macOS a temp
    directory is handed out as /var/folders/... while resolving to
    /private/var/folders/..., so `relative_to` raises and every entry
    comes out absolute — which leaks a server path to the user and leaves
    the viewer unable to match anything against the upload.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_manifest(job_id: str, state: JobState, root: Path,
                   command: List[str], summary: Any,
                   max_files: Optional[int] = None) -> Dict[str, Any]:
    """Everything the reduction did to every file.

    `root_prefix` is load-bearing. `project_root` strips a single
    wrapping directory from the upload, so without recording which one,
    every path here is off by one directory against the archive the user
    sent and a viewer cannot line the two sides up.

    `summary` may be None — a job that was cancelled during provisioning
    never produced one. The manifest is still written, with no file rows,
    because its absence is what a reader would interpret as a lost job.
    """
    files: List[Dict[str, Any]] = []

    def add(path: Path, status: str, before: Optional[int],
            after: Optional[int], **extra: Any) -> None:
        row = {"path": relative_to_root(path, root), "status": status,
               "before_lines": before, "after_lines": after}
        row.update(extra)
        files.append(row)

    if summary is not None:
        # Protected files first, and from `protected_files` rather than
        # from `per_file_reduction`. They are excluded from the reducer's
        # file walk by definition, so they never appear in the walk's
        # results — listing them only from there dropped the one file a
        # reader most wants explained, the test itself.
        protected = set()
        for path in summary.protected_files:
            rel = relative_to_root(path, root)
            protected.add(rel)
            try:
                count = len(path.read_text().splitlines())
            except (OSError, UnicodeDecodeError):
                count = None
            # Identical on both sides on purpose: this file is the
            # question the reduction was answering, not part of the answer.
            files.append({"path": rel, "status": "protected",
                          "before_lines": count, "after_lines": count})

        for why, paths in (("unreachable", summary.unreachable_deleted),
                           ("imported-only", summary.imported_deleted)):
            for path in paths:
                add(path, "deleted", None, 0, reason=why)

        for path in summary.inlined_away:
            add(path, "inlined", None, 0)

        for path, (before, after) in summary.per_file_reduction.items():
            if relative_to_root(path, root) in protected:
                continue
            if after == 0:
                add(path, "deleted", before, 0)
            elif after < before:
                add(path, "reduced", before, after)
            else:
                add(path, "unchanged", before, after)

        for path in summary.undecodable_files:
            add(path, "untouchable", None, None)
        for path in summary.outside_project_files:
            add(path, "outside-project", None, None)
        for rel, err in getattr(summary, "reduction_errors", {}).items():
            files.append({"path": str(rel), "status": "errored",
                          "error": str(err), "before_lines": None,
                          "after_lines": None})

    files.sort(key=lambda f: f["path"])
    truncated = False
    if max_files is not None and len(files) > max_files:
        # A reduced project is small by construction, so this only fires
        # on the *original* tree's deletions. Keeping the survivors is
        # what matters: a reader wants to see what is left, and "4,900
        # other files were deleted" is one line, not 4,900.
        keep = [f for f in files if f["status"] != "deleted"]
        deleted = [f for f in files if f["status"] == "deleted"]
        files = (keep + deleted)[:max_files]
        files.sort(key=lambda f: f["path"])
        truncated = True

    return {
        "job_id": job_id,
        "command": list(command),
        "status": state.status,
        "root_prefix": root.name,
        "stats": state.snapshot(),
        "files": files,
        "truncated": truncated,
    }
