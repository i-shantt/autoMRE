#!/usr/bin/env python3
"""Run the Gistify benchmark on Colab, Kaggle, or any fresh Linux box.

What this adds over calling `gistify_runner.py` directly:

  * installs what the reducer needs and provisions the pinned benchmark
    venv, so a bare notebook works from a clean checkout;
  * runs **one task per process** and writes each result separately.
    That matters more in a notebook than locally: a disconnected session
    or a hit runtime limit otherwise throws away an hour of work, and a
    re-run picks up where it stopped instead of starting over;
  * picks a default `--jobs` from the machine it is actually on, and
    says what it picked.

A caution worth reading before you spend a session on this. The benchmark
is CPU-bound in `pytest` subprocesses, and free cloud runtimes are small:
Colab's free tier gives 2 vCPU and Kaggle 4, against 10 on the M4 laptop
these numbers were measured on. Running here will most likely be
*slower*, not faster. It is worth doing to leave your machine free, to
reproduce a result somewhere neutral, or to run several configurations
side by side in separate sessions — not to make one run finish sooner.

Usage
-----
    python evaluation/cloud_bench.py                 # all six tasks
    python evaluation/cloud_bench.py --only requests # just the fast ones
    python evaluation/cloud_bench.py --jobs 4        # parallel oracle
    python evaluation/cloud_bench.py --summary       # re-print results
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "evaluation" / "cloud_results"
_TASKS = _ROOT / "evaluation" / "gistify_tasks.json"

# What the reducer itself imports. The target projects' test dependencies
# are installed into a separate pinned venv by the runner.
_REQUIREMENTS = ["tree-sitter>=0.21", "tree-sitter-python>=0.21",
                 "coverage>=7.0"]


def _sh(cmd, **kw):
    return subprocess.run(cmd, **kw)


def install() -> None:
    print("installing reducer dependencies...", flush=True)
    _sh([sys.executable, "-m", "pip", "install", "--quiet", *_REQUIREMENTS],
        check=True)
    try:
        import tree_sitter_python  # noqa: F401
        print("  tree-sitter ready", flush=True)
    except ImportError as exc:  # pragma: no cover - environment specific
        raise SystemExit(f"tree-sitter still unavailable after install: {exc}")


def default_jobs() -> int:
    """A worker count worth using, which is smaller than you would guess.

    Measured on `requests-guess_json_utf`, 10-core M4:

        jobs=1   222.2s  1.00x
        jobs=3   174.0s  1.28x
        jobs=6   166.8s  1.33x   <- peak
        jobs=10  185.9s  1.20x   <- worse than 6

    It turns down because two costs grow with width. Speculation discards
    the tail of a batch whenever a candidate goes the way it was not
    predicted to, and at jobs=6 that is already 42% of all subprocesses.
    Meanwhile raw throughput saturates near 2.5x whatever you do, because
    the cores are not equal and `pytest` does not love the small ones.

    Four is the conservative pick: within 5% of the peak for less than
    half the wasted work. A machine with more *equal* cores may support
    more -- the discard rate is printed so you can tell.
    """
    cores = os.cpu_count() or 1
    return max(1, min(4, cores))


def describe_machine() -> None:
    cores = os.cpu_count() or 1
    where = "unknown"
    if "COLAB_GPU" in os.environ or Path("/content").exists():
        where = "Google Colab"
    elif Path("/kaggle").exists():
        where = "Kaggle"
    print(f"host: {where}, {cores} logical CPUs, python {sys.version.split()[0]}")
    if cores < 4:
        print("  note: fewer cores than the machine these numbers were "
              "measured on (10). Expect this to run slower, and expect "
              "--jobs above ~2 to buy little here.", flush=True)


def task_ids(only: str | None) -> list:
    tasks = json.loads(_TASKS.read_text())["tasks"]
    ids = [t["task_id"] for t in tasks]
    if only:
        ids = [i for i in ids if only in i]
    # Fast repo first, so a session that dies early still leaves numbers.
    return sorted(ids, key=lambda i: (not i.startswith("requests"), i))


def run(ids: list, jobs: int, timeout: int) -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    manifests = _RESULTS / "manifests"
    manifests.mkdir(exist_ok=True)
    all_tasks = {t["task_id"]: t
                 for t in json.loads(_TASKS.read_text())["tasks"]}

    for task_id in ids:
        out = _RESULTS / f"{task_id}.json"
        if out.exists():
            print(f"[skip] {task_id} (already have a result)", flush=True)
            continue
        manifest = manifests / f"{task_id}.json"
        manifest.write_text(json.dumps({"tasks": [all_tasks[task_id]]},
                                       indent=2))
        print(f"[run ] {task_id} (jobs={jobs})", flush=True)
        proc = _sh([sys.executable, str(_ROOT / "evaluation" /
                                        "gistify_runner.py"),
                    "--tasks", str(manifest), "--output", str(out),
                    "--jobs", str(jobs), "--timeout", str(timeout)])
        if proc.returncode != 0:
            print(f"[FAIL] {task_id} exited {proc.returncode}", flush=True)


def summarize() -> None:
    files = sorted(_RESULTS.glob("*.json"))
    if not files:
        print("no results yet")
        return
    rows = []
    for path in files:
        payload = json.loads(path.read_text())
        rows.extend(payload.get("runs", []))
    if not rows:
        print("no runs recorded")
        return

    print(f"\n{'task':32s} {'lines':>16s} {'queries':>8s} {'time':>8s}  fid")
    for r in sorted(rows, key=lambda r: r["task_id"]):
        print(f"{r['task_id']:32s} "
              f"{r['original_lines']:6d} -> {r['final_lines']:5d} "
              f"{r['total_queries']:8d} {r['time_seconds']:7.0f}s  "
              f"{r['execution_fidelity']}")

    orig = sum(r["original_lines"] for r in rows)
    final = sum(r["final_lines"] for r in rows)
    prot = sum(r.get("protected_lines", 0) for r in rows)
    fid = sum(r["execution_fidelity"] for r in rows)
    discarded = sum(r.get("speculative_discarded", 0) for r in rows)
    print(f"\n  tasks              {len(rows)}")
    print(f"  execution fidelity {fid}/{len(rows)}")
    print(f"  aggregate          {100 * (orig - final) / orig:.2f}%")
    if orig - prot:
        print(f"  reducible-only     "
              f"{100 * ((orig - prot) - (final - prot)) / (orig - prot):.2f}%")
    print(f"  queries            {sum(r['total_queries'] for r in rows)}")
    if discarded:
        print(f"  speculative work discarded {discarded} "
              f"(extra subprocesses, not counted as queries)")
    print("\nreference (M4 MacBook Air, sequential): 88.52% aggregate, "
          "97.45% reducible-only, 6/6 fidelity, 10,620 queries.")
    print("Compare queries and lines, not wall clock -- wall clock is not "
          "comparable across machines.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="Substring filter on task id, e.g. 'requests'.")
    ap.add_argument("--jobs", type=int, default=None,
                    help="Parallel oracle workers. Default: min(8, cores).")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="Print results already on disk and exit.")
    args = ap.parse_args()

    if args.summary:
        summarize()
        return 0

    describe_machine()
    if not args.skip_install:
        install()

    jobs = args.jobs if args.jobs is not None else default_jobs()
    ids = task_ids(args.only)
    if not ids:
        print(f"no tasks match {args.only!r}")
        return 1
    print(f"tasks: {', '.join(ids)}")
    run(ids, jobs=jobs, timeout=args.timeout)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
