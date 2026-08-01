"""
autoMRE: Learned Removability Oracle
Training Data Capture

Records one JSONL row per Phase 4b removal attempt so the oracle can be
trained on what the delta debugger actually discovered, rather than on a
guess about what should be removable.

Why Phase 4b only
-----------------
Phase 4b attempts units one at a time and validates after each, so the
label is unambiguous: this exact unit was removed, and the bug either
survived or it didn't. Phase 4a removes a whole file's worth of
uncovered units in a single query. When that bulk attempt fails we learn
only that *something* in the batch was load-bearing, with no way to
attribute the failure to a unit — a noisy label that would teach the
model the wrong thing. So 4a stays uninstrumented.

Activation
----------
Off unless AUTOMRE_TRAINING_LOG names a file. That keeps a normal
reduction free of logging cost and stops the reducer from depending on
anything in this package at runtime.

Companion env vars, both optional and used only to tag rows so
leave-one-repo-out CV can group them:

    AUTOMRE_TASK_ID     benchmark task the row came from
    AUTOMRE_REPO_SLUG   repo the task belongs to

Rows append, so a generator script can point many reducer runs at one
file and concatenate naturally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

TRAINING_LOG_ENV = "AUTOMRE_TRAINING_LOG"
TASK_ID_ENV = "AUTOMRE_TASK_ID"
REPO_SLUG_ENV = "AUTOMRE_REPO_SLUG"


class TrainingLogger:
    """Appends feature/label rows for oracle training."""

    def __init__(self, path: Path | str,
                 task_id: str = "",
                 repo_slug: str = ""):
        self.path = Path(path)
        self.task_id = task_id
        self.repo_slug = repo_slug
        self.rows_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Optional["TrainingLogger"]:
        """Build a logger if AUTOMRE_TRAINING_LOG is set, else None."""
        target = os.environ.get(TRAINING_LOG_ENV)
        if not target:
            return None
        return cls(
            path=target,
            task_id=os.environ.get(TASK_ID_ENV, ""),
            repo_slug=os.environ.get(REPO_SLUG_ENV, ""),
        )

    def log_attempt(self,
                    unit: Any,
                    source: str,
                    executed_lines: Set[int],
                    file_path: Optional[Path],
                    was_safely_removable: bool,
                    phase: str = "4b") -> bool:
        """Record one removal attempt.

        Returns True if a row was written. Silently declines when the
        unit has no tree-sitter node to describe (features need the
        structural context), and swallows write errors — a data-capture
        run must never take down a reduction.
        """
        node = getattr(unit, "ts_node", None)
        if node is None:
            return False

        try:
            from .features import extract_features
        except ImportError:  # pragma: no cover - features is dependency-free
            return False

        try:
            feats = extract_features(node, source, executed_lines, file_path)
            row: Dict[str, Any] = feats.as_dict()
            row.update({
                "was_safely_removable": int(bool(was_safely_removable)),
                "task_id": self.task_id,
                "repo_slug": self.repo_slug,
                "file_path": str(file_path) if file_path else "",
                "node_type": node.type,
                "start_line": node.start_point[0] + 1,
                "phase": phase,
            })
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:
            return False

        self.rows_written += 1
        return True
