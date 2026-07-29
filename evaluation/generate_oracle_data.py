"""
AutoRepro-Min: Learned Removability Oracle
Training Data Generation

Runs the full reduction pipeline over every task in
`oracle_training_tasks.json` with AUTOREPRO_TRAINING_LOG set, so each
Phase 4b removal attempt appends a labelled row to one JSONL file.

The labels are observations, not annotations: the reducer removes a unit,
runs the real test command, and records whether the bug survived. Nobody
decides in advance what "should" be removable.

This run is oracle-free by construction — the labels have to come from
watching the un-assisted reducer work — so it doubles as the heuristic
baseline and also writes the scored tasks' results. Running it separately
from the benchmark would pay for the same pipeline twice.

Usage:
    python evaluation/generate_oracle_data.py
    python evaluation/generate_oracle_data.py --train-only-per-repo 1
    python evaluation/generate_oracle_data.py --only requests --append

Budget several minutes per task: once removals genuinely affect the test,
a single requests task runs 1300-1500 validation queries at roughly
170ms each, and flask is about twice the code. Rows land in
evaluation/oracle_training_data.jsonl; benchmark results in
evaluation/results_gistify_baseline_real.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from dataclasses import asdict
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from gistify_runner import (  # noqa: E402
    GistifyTask,
    _ensure_bench_python,
    run_task,
    summarize,
)
from ml.training_log import (  # noqa: E402
    REPO_SLUG_ENV,
    TASK_ID_ENV,
    TRAINING_LOG_ENV,
)

_DEFAULT_MANIFEST = _ROOT / "evaluation" / "oracle_training_tasks.json"
_DEFAULT_OUTPUT = _ROOT / "evaluation" / "oracle_training_data.jsonl"


def load_tasks(path: Path) -> List[GistifyTask]:
    data = json.loads(path.read_text())
    return [GistifyTask(**t) for t in data["tasks"]]


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def select_tasks(tasks: List[GistifyTask],
                 train_only_per_repo: Optional[int]) -> List[GistifyTask]:
    """All benchmark tasks, plus a capped number of train-only ones.

    Every scored task is always included — this run doubles as the
    heuristic baseline, so it has to cover the full benchmark. Train-only
    tasks are capped per repo because they hit diminishing returns fast:
    a single task now yields well over a thousand labelled attempts, so
    the tenth requests task mostly restates the first.
    """
    selected = [t for t in tasks if t.split == "benchmark"]
    if train_only_per_repo is None:
        selected.extend(t for t in tasks if t.split != "benchmark")
        return selected

    seen: Counter = Counter()
    for t in tasks:
        if t.split == "benchmark":
            continue
        if seen[t.repo_slug] >= train_only_per_repo:
            continue
        seen[t.repo_slug] += 1
        selected.append(t)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Phase 4b training data for the oracle.")
    parser.add_argument("--tasks", default=str(_DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--only", default=None,
                        help="Substring filter on task_id.")
    parser.add_argument("--train-only-per-repo", type=int, default=None,
                        help="Cap train-only tasks per repo. Benchmark "
                             "tasks are always included in full.")
    parser.add_argument("--benchmark-output", default=str(
                            _ROOT / "evaluation" /
                            "results_gistify_baseline_real.json"),
                        help="Where to write baseline results for the "
                             "scored tasks. This run is oracle-free, so "
                             "it doubles as the heuristic baseline.")
    parser.add_argument("--append", action="store_true",
                        help="Keep existing rows instead of starting fresh.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.append:
        out_path.unlink()

    tasks = load_tasks(Path(args.tasks))
    tasks = select_tasks(tasks, args.train_only_per_repo)
    if args.only:
        tasks = [t for t in tasks if args.only in t.task_id]
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 1

    bench_python = _ensure_bench_python(verbose=args.verbose)
    print(f"[oracle-data] interpreter: {bench_python}")
    print(f"[oracle-data] {len(tasks)} tasks -> {out_path}")
    print()

    # The reducer builds its logger from the environment, and everything
    # runs in this process, so setting these here is what routes rows to
    # the right file with the right grouping tags.
    os.environ[TRAINING_LOG_ENV] = str(out_path)

    started = time.time()
    per_task = []
    benchmark_results = []
    for i, task in enumerate(tasks, 1):
        os.environ[TASK_ID_ENV] = task.task_id
        os.environ[REPO_SLUG_ENV] = task.repo_slug

        before = _count_rows(out_path)
        print(f"[{i}/{len(tasks)}] {task.task_id} ({task.split})", flush=True)
        t0 = time.time()
        result = run_task(task, timeout=args.timeout, verbose=args.verbose,
                          python=bench_python)
        added = _count_rows(out_path) - before

        status = "ok" if result.execution_fidelity else "FIDELITY-FAIL"
        if result.error:
            status = f"ERROR: {result.error}"
        print(f"      {status}  +{added} rows  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if task.split == "benchmark":
            benchmark_results.append(result)
        per_task.append({
            "task_id": task.task_id,
            "split": task.split,
            "repo_slug": task.repo_slug,
            "rows": added,
            "execution_fidelity": result.execution_fidelity,
            "error": result.error,
        })

    # A task that lost fidelity still produced honest per-unit labels —
    # each row was validated individually — so its rows are kept. It is
    # reported here because a run where many tasks fail usually means the
    # environment is wrong, not that the code is unremovable.
    print()
    print("=" * 62)
    total = _count_rows(out_path)
    print(f"rows written : {total}")
    print(f"elapsed      : {time.time() - started:.0f}s")

    if total:
        with out_path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        labels = Counter(r["was_safely_removable"] for r in rows)
        repos = Counter(r["repo_slug"] for r in rows)
        print(f"label balance: safe={labels.get(1, 0)} "
              f"unsafe={labels.get(0, 0)}")
        print("rows per repo:")
        for repo, n in repos.most_common():
            print(f"  {repo or '<unset>'}: {n}")

    failed = [t for t in per_task if not t["execution_fidelity"]]
    if failed:
        print(f"\n{len(failed)} task(s) did not preserve the bug:")
        for t in failed:
            print(f"  {t['task_id']}: {t['error'] or 'fidelity 0'}")

    # Same pipeline, no oracle — so these numbers are the heuristic
    # baseline the oracle run will be compared against. Writing them here
    # avoids paying for an identical second pass.
    if benchmark_results:
        bench_path = Path(args.benchmark_output)
        bench_path.write_text(json.dumps({
            "config": {
                "prioritizer": "heuristic",
                "coverage_prune": True,
                "learned_oracle": False,
                "test_interpreter": bench_python,
                "note": ("measured during oracle training-data generation; "
                         "identical pipeline, logging only appends rows"),
            },
            "summary": summarize(benchmark_results),
            "runs": [asdict(r) for r in benchmark_results],
        }, indent=2))
        print(f"baseline -> {bench_path}")

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "manifest": str(args.tasks),
        "interpreter": bench_python,
        "total_rows": total,
        "per_task": per_task,
    }, indent=2))
    print(f"\nsummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
