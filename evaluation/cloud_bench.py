#!/usr/bin/env python3
"""Run the Gistify benchmark on Colab, Kaggle, or any fresh Linux box.

What this adds over calling `gistify_runner.py` directly:

  * installs what the reducer needs and provisions the pinned benchmark
    venv, so a bare notebook works from a clean checkout;
  * runs **one task per process** and writes each result separately.
    That matters more in a notebook than locally: a disconnected session
    or a hit runtime limit otherwise throws away hours of work, and a
    re-run picks up where it stopped instead of starting over;
  * `--ablation coverage-prune` runs both arms of an ablation
    *interleaved per task*, which is the only way the comparison is
    worth anything (see below).

A caution worth reading before you spend a session on this. The benchmark
is CPU-bound in `pytest` subprocesses, and free cloud runtimes are small:
Colab's free tier gives 2 vCPU and Kaggle 4, against 10 on the M4 laptop
these numbers were measured on. Running here will most likely be
*slower*, not faster. It is worth doing to leave your machine free, to
reproduce a result somewhere neutral, or to run several configurations
side by side in separate sessions — not to make one run finish sooner.

Why the ablation interleaves
----------------------------
Reduction is deterministic in `lines` and `queries` but not in wall
clock, and those deterministic axes are what an ablation is asking
about. They are only machine-independent if both arms saw the same
interpreter, the same pytest, and the same target checkouts — so the
honest comparison runs both arms **on the same host, back to back, per
task**, rather than diffing a fresh run against numbers measured on
another machine on another day. This is why `--ablation` re-runs the
baseline arm here even though `results_gistify_threerepo.json` already
holds one: that file was measured on the laptop under CPython 3.13, and
a Kaggle result is not comparable to it row by row.

Usage
-----
    python evaluation/cloud_bench.py                      # all ten tasks
    python evaluation/cloud_bench.py --only requests      # just the fast ones
    python evaluation/cloud_bench.py --ablation coverage-prune
    python evaluation/cloud_bench.py --ablation coverage-prune --only requests
    python evaluation/cloud_bench.py --summary            # re-print results
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

# Arm name -> extra flags for gistify_runner.py.
_ABLATIONS = {
    "coverage-prune": {
        "on": [],
        "off": ["--no-coverage-prune"],
        "question": ("Does coverage-based bulk pruning (Phase 4a) earn its "
                     "place, or does Phase 4b find the same code anyway?"),
    },
}


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


def describe_machine() -> None:
    cores = os.cpu_count() or 1
    where = "unknown"
    if "COLAB_GPU" in os.environ or Path("/content").exists():
        where = "Google Colab"
    elif Path("/kaggle").exists():
        where = "Kaggle"
    print(f"host: {where}, {cores} logical CPUs, "
          f"python {sys.version.split()[0]}")
    if cores < 4:
        print("  note: fewer cores than the machine these numbers were "
              "measured on (10). Expect this to run slower.", flush=True)
    print("  reduction is sequential here; only one task runs at a time, "
          "because each task pip-installs into a shared venv.", flush=True)


def task_ids(only: str | None) -> list:
    tasks = json.loads(_TASKS.read_text())["tasks"]
    ids = [t["task_id"] for t in tasks]
    if only:
        ids = [i for i in ids if only in i]
    # Fast repo first, so a session that dies early still leaves numbers.
    return sorted(ids, key=lambda i: (not i.startswith("requests"), i))


def _run_one(task_id: str, out: Path, manifest: Path, extra: list,
             timeout: int, label: str) -> bool:
    if out.exists():
        print(f"[skip] {task_id} [{label}] (already have a result)",
              flush=True)
        return True
    print(f"[run ] {task_id} [{label}]", flush=True)
    proc = _sh([sys.executable,
                str(_ROOT / "evaluation" / "gistify_runner.py"),
                "--tasks", str(manifest), "--output", str(out),
                "--timeout", str(timeout), *extra])
    if proc.returncode != 0:
        print(f"[FAIL] {task_id} [{label}] exited {proc.returncode}",
              flush=True)
        return False
    return True


def _manifest_for(task_id: str, manifests: Path) -> Path:
    all_tasks = {t["task_id"]: t
                 for t in json.loads(_TASKS.read_text())["tasks"]}
    manifest = manifests / f"{task_id}.json"
    manifest.write_text(json.dumps({"tasks": [all_tasks[task_id]]}, indent=2))
    return manifest


def run(ids: list, timeout: int, ablation: str | None) -> None:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    manifests = _RESULTS / "manifests"
    manifests.mkdir(exist_ok=True)

    for task_id in ids:
        manifest = _manifest_for(task_id, manifests)
        if ablation is None:
            _run_one(task_id, _RESULTS / f"{task_id}.json", manifest, [],
                     timeout, "default")
            continue
        # Interleaved: both arms of this task, back to back, before
        # moving on. A session that dies mid-way leaves complete pairs
        # rather than a lopsided set that cannot be compared.
        spec = _ABLATIONS[ablation]
        _run_one(task_id, _RESULTS / f"{task_id}.json", manifest,
                 spec["on"], timeout, f"{ablation}=on")
        _run_one(task_id, _RESULTS / f"{task_id}.{ablation}-off.json",
                 manifest, spec["off"], timeout, f"{ablation}=off")


def _rows(pattern: str, exclude_suffix: str | None = None) -> dict:
    """task_id -> run row, over result files matching `pattern`.

    `exclude_suffix` matters because the default arm's files are plain
    `<task>.json`, which `*.json` matches for the ablated arm too. Left
    unfiltered, every off-arm row would overwrite its own on-arm row and
    the ablation would compare each run against itself — reporting a
    perfect null result no matter what the truth was.
    """
    out = {}
    for path in sorted(_RESULTS.glob(pattern)):
        if exclude_suffix and path.name.endswith(exclude_suffix):
            continue
        for r in json.loads(path.read_text()).get("runs", []):
            out[r["task_id"]] = r
    return out


def summarize(ablation: str | None = None) -> None:
    if ablation:
        return summarize_ablation(ablation)

    # Any ablated arm on disk is a different configuration; it must not be
    # pooled into a headline number that claims to describe the default.
    rows = list(_rows("*.json", exclude_suffix="-off.json").values())
    if not rows:
        print("no results yet")
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
    print(f"\n  tasks              {len(rows)}")
    print(f"  execution fidelity {fid}/{len(rows)}")
    print(f"  aggregate          {100 * (orig - final) / orig:.2f}%")
    if orig - prot:
        print(f"  reducible-only     "
              f"{100 * ((orig - prot) - (final - prot)) / (orig - prot):.2f}%")
    print(f"  queries            {sum(r['total_queries'] for r in rows)}")
    print("\nreference (M4 MacBook Air, sequential, 10 tasks / 3 repos): "
          "86.53% aggregate, 95.23% reducible-only, 10/10 fidelity, "
          "23,326 queries.")
    print("Compare queries and lines, not wall clock -- wall clock is not "
          "comparable across machines.")


def summarize_ablation(ablation: str) -> None:
    off_suffix = f".{ablation}-off.json"
    on = _rows("*.json", exclude_suffix=off_suffix)
    off = _rows(f"*{off_suffix}")

    paired = [t for t in on if t in off]
    if not paired:
        print(f"no complete {ablation} pairs yet "
              f"({len(on)} on-arm, {len(off)} off-arm results)")
        return

    spec = _ABLATIONS[ablation]
    print(f"\nAblation: {ablation} -- {spec['question']}")
    print(f"{len(paired)} paired task(s), both arms run on this host.\n")
    print(f"{'task':30s} {'queries on':>10s} {'off':>8s} {'Δq':>8s} "
          f"{'lines on':>9s} {'off':>7s} {'Δlines':>7s}  fid")
    tq_on = tq_off = tl_on = tl_off = 0
    for t in sorted(paired):
        a, b = on[t], off[t]
        tq_on += a["total_queries"]; tq_off += b["total_queries"]
        tl_on += a["final_lines"];   tl_off += b["final_lines"]
        print(f"{t:30s} {a['total_queries']:10d} {b['total_queries']:8d} "
              f"{b['total_queries'] - a['total_queries']:+8d} "
              f"{a['final_lines']:9d} {b['final_lines']:7d} "
              f"{b['final_lines'] - a['final_lines']:+7d}  "
              f"{a['execution_fidelity']}/{b['execution_fidelity']}")
    print(f"\n{'TOTAL':30s} {tq_on:10d} {tq_off:8d} {tq_off - tq_on:+8d} "
          f"{tl_on:9d} {tl_off:7d} {tl_off - tl_on:+7d}")
    if tq_on:
        print(f"\nturning {ablation} OFF costs "
              f"{100 * (tq_off - tq_on) / tq_on:+.1f}% queries and "
              f"{tl_off - tl_on:+d} lines of output "
              f"({100 * (tl_off - tl_on) / tl_on:+.2f}%).")
    print("Δlines > 0 means the ablated arm reduced WORSE. Wall clock is "
          "deliberately not compared.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="Substring filter on task id, e.g. 'requests'.")
    ap.add_argument("--ablation", choices=sorted(_ABLATIONS), default=None,
                    help="Run both arms of an ablation, interleaved per task.")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--summary", action="store_true",
                    help="Print results already on disk and exit.")
    args = ap.parse_args()

    if args.summary:
        summarize(args.ablation)
        return 0

    describe_machine()
    if not args.skip_install:
        install()

    ids = task_ids(args.only)
    if not ids:
        print(f"no tasks match {args.only!r}")
        return 1
    print(f"tasks: {', '.join(ids)}")
    if args.ablation:
        print(f"ablation: {args.ablation} (each task run twice, "
              f"interleaved)")
    run(ids, timeout=args.timeout, ablation=args.ablation)
    summarize(args.ablation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
