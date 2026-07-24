"""
AutoRepro-Min: Benchmark Runner

Runs `reduce-project` on each project in the benchmark dataset with a
specified prioritizer configuration and records metrics.

Named `bugsinpy_runner` for continuity with the plan — the harness is
designed to accept BugsInPy checkouts too (just pass a list of
BenchmarkBug entries pointing at your BugsInPy work directory), but the
default dataset is the built-in synthetic bugs so results are
reproducible without extra setup.

Usage:
    # Heuristic baseline
    python evaluation/bugsinpy_runner.py --prioritizer heuristic

    # Small LLM (requires `pip install .[llm]`)
    python evaluation/bugsinpy_runner.py --prioritizer llm --model small

    # Custom output path
    python evaluation/bugsinpy_runner.py --prioritizer heuristic \
        --output evaluation/results_heuristic.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from ml import build_prioritizer  # noqa: E402
from multi_file import MultiFileDebugger  # noqa: E402

from benchmark_dataset import BenchmarkBug, load_dataset  # noqa: E402


@dataclass
class RunResult:
    bug_id: str
    prioritizer: str
    model: Optional[str]
    success: bool                    # bug still reproduces after reduction
    original_files: int
    final_files: int
    original_lines: int
    final_lines: int
    total_queries: int
    time_seconds: float
    error: Optional[str] = None      # populated if the run failed outright

    @property
    def file_reduction_rate(self) -> float:
        return ((self.original_files - self.final_files) /
                self.original_files) if self.original_files else 0.0

    @property
    def line_reduction_rate(self) -> float:
        return ((self.original_lines - self.final_lines) /
                self.original_lines) if self.original_lines else 0.0


def _still_reproduces(work_dir: Path, cmd: List[str],
                      expected_error_type: str, timeout: int = 30) -> bool:
    """Run the reproduction command and confirm the expected error fires."""
    try:
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return False  # bug should NOT have been fixed
    return expected_error_type in output


def run_one(bug: BenchmarkBug, prioritizer_kind: str, model: Optional[str],
            verbose: bool = False) -> RunResult:
    """Reduce a single benchmark bug and return metrics."""
    prioritizer = build_prioritizer(kind=prioritizer_kind, model=model,
                                    verbose=verbose)

    with tempfile.TemporaryDirectory(prefix="ampl_bench_") as tmp:
        work_dir = Path(tmp) / bug.project_dir.name
        shutil.copytree(bug.project_dir, work_dir)

        try:
            debugger = MultiFileDebugger(verbose=verbose,
                                         prioritizer=prioritizer)
            start = time.time()
            summary = debugger.reduce_project(
                work_dir, bug.reproduction_command)
            elapsed = time.time() - start
        except Exception as exc:
            return RunResult(
                bug_id=bug.bug_id, prioritizer=prioritizer_kind,
                model=model, success=False,
                original_files=0, final_files=0,
                original_lines=0, final_lines=0,
                total_queries=0, time_seconds=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        success = _still_reproduces(
            work_dir, bug.reproduction_command,
            bug.expected_error_type)

        return RunResult(
            bug_id=bug.bug_id,
            prioritizer=prioritizer_kind,
            model=model,
            success=success,
            original_files=summary.original_file_count,
            final_files=summary.final_file_count,
            original_lines=summary.original_line_count,
            final_lines=summary.final_line_count,
            total_queries=summary.total_queries,
            time_seconds=elapsed,
        )


def run_all(bugs: List[BenchmarkBug], prioritizer_kind: str,
            model: Optional[str], verbose: bool = False) -> List[RunResult]:
    results: List[RunResult] = []
    for bug in bugs:
        print(f"[{prioritizer_kind}"
              f"{'/'+model if model else ''}] {bug.bug_id}...", flush=True)
        r = run_one(bug, prioritizer_kind, model, verbose=verbose)
        icon = "OK" if r.success else "FAIL"
        if r.error:
            print(f"  {icon}  errored: {r.error}")
        else:
            print(f"  {icon}  files {r.original_files}->{r.final_files} "
                  f"lines {r.original_lines}->{r.final_lines} "
                  f"queries={r.total_queries} time={r.time_seconds:.2f}s")
        results.append(r)
    return results


def summarize(results: List[RunResult]) -> Dict:
    """Aggregate a run into a summary dict suitable for JSON output."""
    successful = [r for r in results if r.success]
    total = len(results)
    n_ok = len(successful)

    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    return {
        "n_bugs": total,
        "n_success": n_ok,
        "success_rate": (n_ok / total) if total else 0.0,
        "median_file_reduction": _median(
            [r.file_reduction_rate for r in successful]),
        "median_line_reduction": _median(
            [r.line_reduction_rate for r in successful]),
        "median_queries": _median(
            [r.total_queries for r in successful]),
        "median_time_seconds": _median(
            [r.time_seconds for r in successful]),
        "total_time_seconds": sum(r.time_seconds for r in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AutoRepro-Min benchmark on the built-in "
                    "multi-file bug dataset.")
    parser.add_argument("--prioritizer", default="heuristic",
                        choices=["heuristic", "llm"],
                        help="Prioritization strategy to evaluate")
    parser.add_argument("--model", default=None,
                        choices=["tiny", "small", "medium", "large", "alt"],
                        help="Model tier when --prioritizer=llm")
    parser.add_argument("--output", default=None,
                        help="Where to write the JSON result "
                             "(default: evaluation/results_bench_<key>.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Chatty per-run output")
    args = parser.parse_args()

    bugs = load_dataset()
    if not bugs:
        print("No benchmark projects found; expected them under "
              "autorepro_min/examples/multi_file/", file=sys.stderr)
        return 1

    results = run_all(bugs, args.prioritizer, args.model,
                      verbose=args.verbose)
    summary = summarize(results)

    output = args.output
    if not output:
        key = args.prioritizer + (f"_{args.model}" if args.model else "")
        output = str(_ROOT / "evaluation" / f"results_bench_{key}.json")

    payload = {
        "config": {
            "prioritizer": args.prioritizer,
            "model": args.model,
        },
        "summary": summary,
        "runs": [asdict(r) for r in results],
    }
    Path(output).write_text(json.dumps(payload, indent=2))
    print()
    print("=" * 60)
    print(f"Results written to: {output}")
    print(f"  Success:              {summary['n_success']}/{summary['n_bugs']}")
    print(f"  Median file reduction: "
          f"{summary['median_file_reduction']*100:.1f}%")
    print(f"  Median line reduction: "
          f"{summary['median_line_reduction']*100:.1f}%")
    print(f"  Median queries:        {summary['median_queries']:.1f}")
    print(f"  Median time / bug:     "
          f"{summary['median_time_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
