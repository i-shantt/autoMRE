"""
AutoRepro-Min: Gistify-Style Benchmark Runner

Runs `reduce-project` on real repositories from the Gistify benchmark
(Lee et al., ICLR 2026, arXiv 2510.26790).

The Gistify metric — Execution Fidelity — is binary per test:

    1[ runs(c, G) AND out(c, G) = out(c, C) ]

...where c is the reproduction command, C is the original codebase,
and G is the reduced ("gistified") output. Success rate across N tasks
is what the paper reports. Best reported number in the paper: 58.7%
(Copilot + Claude-4).

Note: this harness uses a CUSTOM test selection (see
`evaluation/gistify_tasks.json`) because the paper's exact 25-test
list isn't publicly released. Same repos, same evaluation shape,
different specific tests — so numbers are directionally comparable
but not a formal head-to-head on identical inputs.

Usage:
    python evaluation/gistify_runner.py
    python evaluation/gistify_runner.py --tasks path/to/tasks.json --output x.json
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
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from multi_file import MultiFileDebugger  # noqa: E402
from validator import Validator  # noqa: E402


# On-disk cache for cloned repos so we don't re-clone across runs.
_REPO_CACHE = _ROOT / ".gistify_repo_cache"

# Benchmark interpreter, provisioned on first use.
#
# The benchmark is only meaningful if the target test passes before we
# reduce anything. flask's tests/conftest.py imports the private sentinel
# `_pytest.monkeypatch.notset`, which pytest dropped in 9.1 — under a
# newer pytest every flask task dies during collection. That failure is
# quiet and dangerous: the harness would faithfully capture "conftest
# raises AttributeError" as the behavior to preserve, and since only
# conftest.py is needed to keep raising it, the reducer could delete
# nearly the whole repo and still score execution fidelity 1. A pinned
# interpreter turns an ambient-environment dependency into a declared one.
#
# Every flask tag from 3.0.3 through 3.1.3 uses the private API, so the
# pin has to be on pytest, not on flask.
_BENCH_VENV = _ROOT / ".gistify_venv"
_PINNED_PYTEST = "pytest==9.0.0"


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_bench_python(verbose: bool = False) -> str:
    """Path to the benchmark interpreter, creating the venv if needed."""
    py = _venv_python(_BENCH_VENV)
    if py.exists():
        return str(py)

    print(f"  provisioning benchmark venv at {_BENCH_VENV.name} "
          f"({_PINNED_PYTEST})...", flush=True)
    subprocess.run([sys.executable, "-m", "venv", str(_BENCH_VENV)],
                   check=True)
    subprocess.run([str(py), "-m", "pip", "install", "--quiet",
                    "--upgrade", "pip"], check=False)
    proc = subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", _PINNED_PYTEST],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not install {_PINNED_PYTEST}: {proc.stderr[-800:]}")
    return str(py)


def _resolve_test_command(cmd: List[str], python: str) -> List[str]:
    """Point a manifest's `python3 -m pytest ...` at `python`.

    Manifests stay portable by naming a bare interpreter; the harness
    decides which one actually runs.
    """
    if not cmd:
        return cmd
    head = Path(cmd[0]).name
    if head in ("python", "python3", "python3.13", sys.executable):
        return [python] + list(cmd[1:])
    return list(cmd)


@dataclass
class GistifyTask:
    task_id: str
    repo: str
    commit: str
    test_command: List[str]
    notes: str = ""


@dataclass
class GistifyResult:
    task_id: str
    execution_fidelity: int          # 0 or 1 (Gistify metric)
    original_files: int
    final_files: int
    original_lines: int
    final_lines: int
    single_file_output: bool         # Self-containment proxy
    total_queries: int
    time_seconds: float
    error: Optional[str] = None


# ------------------------------------------------------------ repo mgmt

def _ensure_repo(task: GistifyTask, verbose: bool = False) -> Path:
    """Clone the task's repo into the cache and check out the commit."""
    _REPO_CACHE.mkdir(exist_ok=True)
    slug = task.repo.rstrip("/").split("/")[-1]
    if slug.endswith(".git"):
        slug = slug[:-4]
    repo_dir = _REPO_CACHE / slug
    if not repo_dir.exists():
        if verbose:
            print(f"  cloning {task.repo}...")
        subprocess.run(
            ["git", "clone", "--quiet", task.repo, str(repo_dir)],
            check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--quiet", "--tags"],
        check=False)
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--quiet", task.commit],
        check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "clean", "-fdxq"],
        check=False)
    return repo_dir


def _install_repo(repo_dir: Path, python: str,
                  verbose: bool = False) -> bool:
    """`pip install -e .` the checked-out repo into the benchmark env."""
    stamp = repo_dir / ".autorepro_installed"
    if stamp.exists():
        return True
    if verbose:
        print(f"  installing {repo_dir.name} (one-time)...")
    proc = subprocess.run(
        [python, "-m", "pip", "install", "--quiet", "-e", str(repo_dir)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  install failed: {proc.stderr[-800:]}", file=sys.stderr)
        return False
    stamp.touch()
    return True


# ------------------------------------------------------------ eval

def _capture_baseline(work_dir: Path, cmd: List[str],
                      timeout: int) -> tuple:
    try:
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True,
                              text=True, timeout=timeout)
        return (proc.stdout or "") + (proc.stderr or ""), proc.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1


def _check_execution_fidelity(work_dir: Path, cmd: List[str],
                              baseline_output: str, baseline_rc: int,
                              timeout: int) -> bool:
    """Does the reduced tree produce the same normalized output?"""
    try:
        proc = subprocess.run(cmd, cwd=work_dir, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != baseline_rc:
        return False
    got = (proc.stdout or "") + (proc.stderr or "")
    return (Validator._normalize_output(got) ==
            Validator._normalize_output(baseline_output))


def _count_files_and_lines(root: Path) -> tuple:
    files = list(root.rglob("*.py"))
    total_lines = 0
    for f in files:
        try:
            total_lines += len(f.read_text().splitlines())
        except (OSError, UnicodeDecodeError):
            continue
    return len(files), total_lines


# ------------------------------------------------------------ per-task

def run_task(task: GistifyTask, timeout: int = 120,
             verbose: bool = False,
             use_coverage_prune: bool = True,
             python: Optional[str] = None) -> GistifyResult:
    python = python or sys.executable
    test_command = _resolve_test_command(task.test_command, python)
    print(f"  → cloning/preparing {task.task_id}...", flush=True)
    try:
        source_dir = _ensure_repo(task, verbose=verbose)
        if not _install_repo(source_dir, python, verbose=verbose):
            return GistifyResult(
                task_id=task.task_id, execution_fidelity=0,
                original_files=0, final_files=0,
                original_lines=0, final_lines=0,
                single_file_output=False,
                total_queries=0, time_seconds=0.0,
                error="repo install failed")
    except subprocess.CalledProcessError as exc:
        return GistifyResult(
            task_id=task.task_id, execution_fidelity=0,
            original_files=0, final_files=0,
            original_lines=0, final_lines=0,
            single_file_output=False,
            total_queries=0, time_seconds=0.0,
            error=f"repo prep failed: {exc}")

    with tempfile.TemporaryDirectory(prefix="gistify_") as tmp:
        work_dir = Path(tmp) / source_dir.name
        shutil.copytree(source_dir, work_dir,
                        ignore=shutil.ignore_patterns(
                            ".git", "*.egg-info", "__pycache__",
                            ".pytest_cache", "build", "dist"))

        print(f"  → capturing baseline output...", flush=True)
        baseline_out, baseline_rc = _capture_baseline(
            work_dir, test_command, timeout=timeout)
        if baseline_rc < 0 and "TIMEOUT" in baseline_out:
            return GistifyResult(
                task_id=task.task_id, execution_fidelity=0,
                original_files=0, final_files=0,
                original_lines=0, final_lines=0,
                single_file_output=False,
                total_queries=0, time_seconds=0.0,
                error="baseline timed out")

        orig_files, orig_lines = _count_files_and_lines(work_dir)
        print(f"  → baseline ok (rc={baseline_rc}); "
              f"repo has {orig_files} .py files, "
              f"{orig_lines} lines", flush=True)

        debugger = MultiFileDebugger(
            verbose=verbose, timeout=timeout,
            match_strategy="output_match",
            aggressive_inline=True,
            use_coverage_prune=use_coverage_prune,
        )

        print(f"  → reducing...", flush=True)
        start = time.time()
        try:
            summary = debugger.reduce_project(work_dir, test_command)
        except Exception as exc:
            return GistifyResult(
                task_id=task.task_id, execution_fidelity=0,
                original_files=orig_files, final_files=orig_files,
                original_lines=orig_lines, final_lines=orig_lines,
                single_file_output=False,
                total_queries=0, time_seconds=time.time() - start,
                error=f"reduction crashed: {type(exc).__name__}: {exc}")
        elapsed = time.time() - start

        final_files, final_lines = _count_files_and_lines(work_dir)
        print(f"  → checking execution fidelity...", flush=True)
        fidelity = _check_execution_fidelity(
            work_dir, test_command,
            baseline_out, baseline_rc, timeout=timeout)

        return GistifyResult(
            task_id=task.task_id,
            execution_fidelity=1 if fidelity else 0,
            original_files=orig_files,
            final_files=final_files,
            original_lines=orig_lines,
            final_lines=final_lines,
            single_file_output=(final_files == 1),
            total_queries=summary.total_queries,
            time_seconds=elapsed,
        )


# ------------------------------------------------------------ driver

def load_tasks(path: Path) -> List[GistifyTask]:
    data = json.loads(path.read_text())
    return [GistifyTask(**t) for t in data["tasks"]]


def summarize(results: List[GistifyResult]) -> Dict:
    total = len(results)
    fidelity_sum = sum(r.execution_fidelity for r in results)
    single_file_sum = sum(int(r.single_file_output) for r in results)
    return {
        "n_tasks": total,
        "execution_fidelity_rate": fidelity_sum / total if total else 0.0,
        "single_file_rate": single_file_sum / total if total else 0.0,
        "avg_time_seconds": (sum(r.time_seconds for r in results) /
                             total) if total else 0.0,
        "avg_queries": (sum(r.total_queries for r in results) /
                        total) if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gistify-style benchmark harness for AutoRepro-Min.")
    parser.add_argument("--tasks", default=str(
        _ROOT / "evaluation" / "gistify_tasks.json"),
        help="Path to a tasks JSON manifest")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-command timeout in seconds")
    parser.add_argument("--output", default=None,
        help="Where to write the JSON result. Default depends on flags: "
             "results_gistify_heuristic.json or "
             "results_gistify_heuristic_no_coverage.json.")
    parser.add_argument("--no-coverage-prune", action="store_true",
        help="Disable coverage-based bulk pruning (ablation).")
    parser.add_argument("--python", default=None,
        help="Interpreter to run the target tests with. Defaults to a "
             f"pinned benchmark venv ({_PINNED_PYTEST}).")
    parser.add_argument("--no-venv", action="store_true",
        help="Run tests with the ambient interpreter instead of the "
             "pinned benchmark venv. Results may not be comparable.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        suffix = "_no_coverage" if args.no_coverage_prune else ""
        args.output = str(_ROOT / "evaluation" /
                          f"results_gistify_heuristic{suffix}.json")

    tasks = load_tasks(Path(args.tasks))
    if not tasks:
        print("No tasks in manifest.", file=sys.stderr)
        return 1

    if args.python:
        bench_python = args.python
    elif args.no_venv:
        bench_python = sys.executable
    else:
        bench_python = _ensure_bench_python(verbose=args.verbose)
    print(f"[gistify] test interpreter: {bench_python}", flush=True)

    results: List[GistifyResult] = []
    for task in tasks:
        print(f"[gistify] {task.task_id}", flush=True)
        r = run_task(task, timeout=args.timeout, verbose=args.verbose,
                     use_coverage_prune=not args.no_coverage_prune,
                     python=bench_python)
        icon = "PASS" if r.execution_fidelity else "FAIL"
        if r.error:
            print(f"  {icon} error: {r.error}")
        else:
            print(f"  {icon} files {r.original_files}->{r.final_files} "
                  f"lines {r.original_lines}->{r.final_lines} "
                  f"single_file={r.single_file_output} "
                  f"queries={r.total_queries} "
                  f"time={r.time_seconds:.1f}s")
        results.append(r)

    summary = summarize(results)
    payload = {
        "config": {
            "prioritizer": "heuristic",
            "coverage_prune": not args.no_coverage_prune,
            "test_interpreter": bench_python,
            "pinned_pytest": (None if (args.python or args.no_venv)
                              else _PINNED_PYTEST),
        },
        "summary": summary,
        "runs": [asdict(r) for r in results],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print()
    print("=" * 60)
    print(f"Results written to: {args.output}")
    print(f"Execution fidelity: "
          f"{summary['execution_fidelity_rate']*100:.1f}% "
          f"({int(summary['execution_fidelity_rate']*summary['n_tasks'])}"
          f"/{summary['n_tasks']})")
    print(f"Single-file output: "
          f"{summary['single_file_rate']*100:.1f}%")
    print(f"Avg time / task:    {summary['avg_time_seconds']:.1f}s")
    print(f"Gistify best (paper): 58.7% execution fidelity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
