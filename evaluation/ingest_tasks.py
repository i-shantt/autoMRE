"""Turn a list of (repo, commit, failing test) into a task manifest.

The benchmark's ten tasks are hand-written, and so are the eighteen the
learned oracle trains on. That is the whole reason autoMRE has been
measured on three repositories: adding a fourth means a person working
out how to install it and which test to name. This script does that part.

Its input is deliberately dull — a JSON or JSONL file where each record
names a repository, a commit, and one or more tests. SWE-bench Verified
exports in exactly that shape (`repo`, `base_commit`, `FAIL_TO_PASS`),
and so would SWE-Hub's task tuple if its data is ever released, but
nothing here is tied to either: a hand-written list of three dicts works
just as well.

One thing worth stating, because it is what makes SWE-bench-shaped data
usable here at all. Those instances are pinned at the commit *before* the
fix, so their FAIL_TO_PASS test fails. That is not an obstacle — it is
the ordinary case for a minimal reproducer. autoMRE's question is "what
is the smallest project that still does this", and "this" is usually a
crash. The benchmark's passing tests are the special case, not these.

Every instance is cloned, provisioned and put through the readiness gate.
Accepted ones become a manifest that `gistify_runner.py --tasks` will
reduce. Rejected ones are written out with the reason, because a silently
dropped instance is how a benchmark starts measuring nothing.

Usage:
    python3 evaluation/ingest_tasks.py --instances swebench_verified.json
    python3 evaluation/ingest_tasks.py --instances tasks.jsonl --limit 15
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from provision import ProvisionError, check, provision  # noqa: E402

from gistify_runner import (  # noqa: E402
    GistifyTask,
    _count_files_and_lines,
    _ensure_repo,
)

# Copied out of the checkout before anything runs, mirroring what
# gistify_runner reduces, so the gate sees the tree the reducer will.
_IGNORE = shutil.ignore_patterns(".git", "*.egg-info", "__pycache__",
                                 ".pytest_cache", "build", "dist")


@dataclass
class Ingested:
    """One instance's verdict, whichever way it went."""
    task_id: str
    repo: str
    commit: str
    accepted: bool
    reason: Optional[str] = None
    files: int = 0
    lines: int = 0
    gate_runs: int = 0
    seconds: float = 0.0


# ------------------------------------------------------------- reading

def load_instances(path: Path) -> List[Dict[str, Any]]:
    """Read JSON or JSONL, and don't be fussy about which."""
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{") and "\n" not in stripped.rstrip():
        return [json.loads(text)]
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def to_task(record: Dict[str, Any]) -> GistifyTask:
    """Map one record onto the manifest shape the runner already reads.

    Only the first named test is used. A SWE-bench instance often lists
    several FAIL_TO_PASS tests for one bug, and reducing against one of
    them is a well-defined minimal-reproducer task that costs a fraction
    of the queries — per-query cost is what the whole runtime is made of.
    """
    repo = record.get("repo") or record.get("repository")
    if not repo:
        raise ValueError(f"record has no repo: {sorted(record)}")
    # "psf/requests" is the shorthand every SWE-bench-shaped export uses.
    # A path or a URL is left alone: cloning from a local mirror is how
    # you ingest a hundred instances of twelve repositories without
    # fetching each one a hundred times.
    if not (repo.startswith(("http://", "https://", "git@", "ssh://",
                             "file://", "/", "."))
            or Path(repo).exists()):
        repo = f"https://github.com/{repo}"

    commit = (record.get("base_commit") or record.get("commit")
              or record.get("environment_setup_commit"))
    if not commit:
        raise ValueError(f"record has no commit: {sorted(record)}")

    tests = _test_ids(record)
    if not tests:
        raise ValueError(f"record names no test: {sorted(record)}")

    slug = repo.rstrip("/").removesuffix(".git").split("/")[-1]
    task_id = record.get("instance_id") or record.get("task_id") \
        or f"{slug}-{commit[:8]}"

    return GistifyTask(
        task_id=str(task_id),
        repo=repo,
        commit=str(commit),
        test_command=["python3", "-m", "pytest", tests[0], "-x", "-q"],
        notes=f"ingested; {len(tests)} test(s) named, first one used",
    )


def _test_ids(record: Dict[str, Any]) -> List[str]:
    """Node ids, from whichever field the source happened to use.

    SWE-bench stores FAIL_TO_PASS as a JSON-encoded string in some
    exports and a real list in others.
    """
    for key in ("FAIL_TO_PASS", "fail_to_pass", "tests", "test_ids",
                "TP2F"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [value]
        if isinstance(value, list) and value:
            return [str(v) for v in value]
    return []


# ------------------------------------------------------------ ingesting

def ingest_one(task: GistifyTask, workspace: Path,
               timeout: int = 120, verbose: bool = False,
               cache_dir: Optional[Path] = None) -> Ingested:
    """Clone, provision, and gate one instance."""
    start = time.time()
    result = Ingested(task_id=task.task_id, repo=task.repo,
                      commit=task.commit, accepted=False)

    try:
        source = _ensure_repo(task, verbose=verbose, cache_dir=cache_dir)
    except subprocess.CalledProcessError as exc:
        result.reason = f"could not check out {task.commit}: {exc}"
        result.seconds = time.time() - start
        return result

    work = workspace / task.task_id.replace("/", "_")
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(source, work, ignore=_IGNORE)
    result.files, result.lines = _count_files_and_lines(work)

    try:
        # The venv is a sibling of the work copy, never inside it, and
        # the work copy is what gets `pip install -e .` — so the tree the
        # reducer will cut is the tree the tests import. The benchmark
        # needs two installs to arrange that; here it falls out.
        spec = provision(work, workspace / f"{work.name}.venv")
    except ProvisionError as exc:
        result.reason = str(exc)
        result.seconds = time.time() - start
        return result

    command = [spec.python] + list(task.test_command[1:])
    verdict = check(work, command, timeout=timeout)
    result.gate_runs = verdict.command_runs
    result.accepted = verdict.ok
    result.reason = verdict.reason
    result.seconds = time.time() - start
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instances", required=True, type=Path,
                    help="JSON or JSONL file of records to ingest")
    ap.add_argument("--output", type=Path,
                    default=_ROOT / "evaluation" / "tasks_ingested.json",
                    help="manifest of accepted tasks, in gistify_tasks shape")
    ap.add_argument("--rejects", type=Path,
                    default=_ROOT / "evaluation" / "tasks_rejected.json",
                    help="rejected instances, with the reason for each")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many instances (use for a pilot)")
    ap.add_argument("--workspace", type=Path, default=None,
                    help="where checkouts and venvs go; a temp dir by default")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--cache", type=Path, default=None,
                    help="where clones are kept; the shared repo cache "
                         "by default")
    ap.add_argument("--keep-workspace", action="store_true",
                    help="do not delete the workspace; it holds the venvs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    records = load_instances(args.instances)
    if args.limit:
        records = records[:args.limit]
    print(f"[ingest] {len(records)} instance(s) from {args.instances.name}")

    holder = None
    if args.workspace:
        workspace = args.workspace
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        holder = tempfile.TemporaryDirectory(prefix="automre_ingest_")
        workspace = Path(holder.name)

    results: List[Ingested] = []
    accepted_tasks: List[GistifyTask] = []
    try:
        for i, record in enumerate(records, 1):
            try:
                task = to_task(record)
            except ValueError as exc:
                print(f"[{i}/{len(records)}] unreadable record: {exc}")
                results.append(Ingested(task_id=f"record-{i}", repo="?",
                                        commit="?", accepted=False,
                                        reason=str(exc)))
                continue

            print(f"[{i}/{len(records)}] {task.task_id} ...",
                  end=" ", flush=True)
            outcome = ingest_one(task, workspace, timeout=args.timeout,
                                 verbose=args.verbose,
                                 cache_dir=args.cache)
            results.append(outcome)
            if outcome.accepted:
                print(f"ok  ({outcome.lines} lines, {outcome.seconds:.0f}s)")
                accepted_tasks.append(task)
            else:
                print(f"REJECT — {outcome.reason}")
    finally:
        if holder is not None and not args.keep_workspace:
            holder.cleanup()

    accepted = [r for r in results if r.accepted]
    _write_manifest(args.output, accepted_tasks, args.instances)
    args.rejects.write_text(json.dumps(
        {"rejected": [asdict(r) for r in results if not r.accepted]},
        indent=2) + "\n")

    print()
    print(f"[ingest] accepted {len(accepted)}/{len(results)}")
    print(f"[ingest] manifest: {args.output}")
    print(f"[ingest] rejects:  {args.rejects}")
    if accepted:
        total_lines = sum(r.lines for r in accepted)
        print(f"[ingest] {total_lines} lines across accepted tasks; at the "
              f"measured 0.11-0.48 queries per line that is roughly "
              f"{int(total_lines * 0.11):,}-{int(total_lines * 0.48):,} "
              f"queries to reduce them all")
    return 0 if accepted else 1


def _write_manifest(path: Path, tasks: List[GistifyTask],
                    source: Path) -> None:
    path.write_text(json.dumps({
        "_description": "Tasks ingested from an instance list and passed "
                        "through the readiness gate. Reduce them with "
                        "`gistify_runner.py --tasks <this file>`.",
        "_source": str(source),
        "_gate": "Each task's command was run at least three times before "
                 "it was accepted: the test exists, the command is "
                 "deterministic, and removing the package changes the "
                 "result.",
        "tasks": [asdict(t) for t in tasks],
    }, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
