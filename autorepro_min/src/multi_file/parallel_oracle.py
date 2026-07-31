"""AutoRepro-Min: Parallel Oracle

A pool of project copies, so several candidate removals can be validated
at once.

Why this exists
---------------
Profiling put 98-99% of a reduction inside the oracle subprocess, and a
breakdown of one query put ~95% of *that* in fixed setup — interpreter
start, `import flask`, pytest collection — with the test body itself
around 5%. There is no hot spot left in the reducer to remove and no
meaningful way to make one query cheaper without changing the
reproduction command, which is the user's and defines the property being
preserved. What is left is asking several questions at the same time.

Why a copy per worker
---------------------
Two concurrent queries need two different contents at the same path, so
they cannot share a directory. Each worker therefore gets its own copy of
the tree, kept in step with the real one by `sync()`.

The dangerous part is not the copying, it is *import resolution*. The
benchmark makes the tree under reduction authoritative with
`pip install -e`, which points one venv at one directory. A worker whose
`import flask` resolved back to that directory would answer every
question about code nobody is editing — which is precisely the bug that
made an entire earlier generation of this benchmark meaningless (see the
first entry in the README's trustworthiness section). So each worker
prepends its own copy to `PYTHONPATH`, and `start()` refuses to enable
parallelism unless it has *verified*, per worker, that the package
resolves inside that worker's tree and that the unmodified tree still
reproduces. Refusing is the correct failure mode here: a wrong answer is
far more expensive than a slow one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys as _sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

from validator import oracle_env  # noqa: E402


@dataclass
class Candidate:
    """One question: with `source` written at `path`, does the bug hold?

    `path` is relative to the project root, so it can be replayed inside
    any worker's copy.
    """
    path: Path
    source: str


class ParallelOracle:
    """`jobs` copies of a project, each able to answer one query.

    Threads rather than processes: every worker spends its life blocked
    in `subprocess.run`, which releases the GIL, so threads cost nothing
    here and keep the failure modes simple.
    """

    def __init__(self, project_dir: Path, command: Sequence[str],
                 oracle, timeout: int = 60, jobs: int = 1):
        self.project_dir = Path(project_dir).resolve()
        self.command = list(command)
        self.timeout = timeout
        self.jobs = max(1, int(jobs))
        self._oracle = oracle          # a Validator, for _matches_original
        self._roots: List[Path] = []
        self._pool: Optional[ThreadPoolExecutor] = None
        self._stamps: List[Dict[Path, Tuple[int, int]]] = []
        self.enabled = False
        self.reason: Optional[str] = None

    # ------------------------------------------------------------ setup

    def start(self) -> bool:
        """Build and verify the worker copies. False means stay sequential."""
        if self.jobs < 2:
            self.reason = "jobs < 2"
            return False

        parent = self.project_dir.parent / f".{self.project_dir.name}_workers"
        shutil.rmtree(parent, ignore_errors=True)
        parent.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(self.jobs):
                root = parent / f"w{i}"
                shutil.copytree(
                    self.project_dir, root,
                    ignore=shutil.ignore_patterns("__pycache__", ".git",
                                                  "*.egg-info", ".pytest_cache"))
                self._roots.append(root)
                self._stamps.append(self._stat_tree(root))
        except OSError as exc:
            self.reason = f"could not build worker copies: {exc}"
            self.close()
            return False

        bad = self._verify()
        if bad is not None:
            self.reason = bad
            self.close()
            return False

        self._pool = ThreadPoolExecutor(max_workers=self.jobs,
                                        thread_name_prefix="oracle")
        self.enabled = True
        return True

    def _verify(self) -> Optional[str]:
        """None if every worker is trustworthy, else why it is not."""
        packages = self._top_level_packages()
        for root in self._roots:
            for pkg in packages:
                proc = subprocess.run(
                    [self._interpreter(), "-c",
                     f"import {pkg}, os; print(os.path.realpath({pkg}.__file__))"],
                    cwd=root, env=self._env(root),
                    capture_output=True, text=True, timeout=self.timeout)
                if proc.returncode != 0:
                    return (f"worker {root.name} cannot import {pkg}: "
                            f"{proc.stderr.strip()[-200:]}")
                resolved = Path(proc.stdout.strip()).resolve()
                if root.resolve() not in resolved.parents:
                    return (f"worker {root.name} resolves {pkg} to "
                            f"{resolved}, outside its own copy — every "
                            f"answer would describe code nobody is editing")

            # The tree is unmodified, so it must still reproduce. This
            # catches anything the resolution check cannot see: missing
            # dist metadata, a plugin that reads an absolute path, a
            # fixture keyed to the original directory.
            run = self._run_in(root)
            if not self._oracle._matches_original(run[0], run[1]):
                return (f"worker {root.name} does not reproduce the bug on "
                        f"an unmodified copy")
        return None

    def _top_level_packages(self) -> List[str]:
        """Importable package names the copy provides."""
        names: List[str] = []
        for base in (self.project_dir / "src", self.project_dir):
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if (child / "__init__.py").exists() and not \
                        child.name.startswith((".", "test")):
                    names.append(child.name)
            if names:
                break
        return names

    # ------------------------------------------------------------ running

    def _interpreter(self) -> str:
        for arg in self.command:
            if "python" in Path(arg).name:
                return arg
        return _sys.executable

    def _env(self, root: Path) -> Dict[str, str]:
        env = oracle_env()
        # Prepend this worker's copy so it wins over the editable install
        # that points at the real tree. sys.path takes PYTHONPATH ahead of
        # site-packages, which is what makes this work at all.
        src = root / "src" if (root / "src").is_dir() else root
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (f"{src}{os.pathsep}{existing}" if existing
                             else str(src))
        return env

    def _run_in(self, root: Path) -> Tuple[str, int]:
        try:
            proc = subprocess.run(
                self.command, cwd=root, capture_output=True, text=True,
                timeout=self.timeout, env=self._env(root))
            return (proc.stdout or "") + (proc.stderr or ""), proc.returncode
        except subprocess.TimeoutExpired:
            return "Timeout: reproduction command exceeded time limit", -1
        except FileNotFoundError as exc:
            return f"Command not found: {exc}", -1

    # ------------------------------------------------------------ syncing

    @staticmethod
    def _stat_tree(root: Path) -> Dict[Path, Tuple[int, int]]:
        out: Dict[Path, Tuple[int, int]] = {}
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            out[path.relative_to(root)] = (st.st_size, st.st_mtime_ns)
        return out

    def sync(self) -> None:
        """Bring every worker copy in line with the real tree.

        Only `.py` files move, because they are the only thing reduction
        writes or deletes. Comparing (size, mtime) rather than content is
        safe in the direction that matters: the reducer rewrites through
        `write_text`, which updates both.
        """
        if not self.enabled:
            return
        current = self._stat_tree(self.project_dir)
        for index, root in enumerate(self._roots):
            previous = self._stamps[index]
            for rel, stamp in current.items():
                if previous.get(rel) == stamp:
                    continue
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.project_dir / rel, target)
            for rel in previous.keys() - current.keys():
                try:
                    (root / rel).unlink()
                except OSError:
                    pass
            self._stamps[index] = dict(current)

    def relative_path(self, path) -> Path:
        """A project path expressed the way a worker copy can replay it."""
        path = Path(path)
        try:
            return path.resolve().relative_to(self.project_dir)
        except ValueError:
            return Path(path.name)

    # ------------------------------------------------------------ asking

    def ask(self, candidates: Sequence[Candidate]) -> List[bool]:
        """Verdicts for `candidates`, in the order given.

        Each candidate is written into one worker, judged, and the worker
        is put back the way it was. Workers are never shared between two
        candidates at the same time, so no locking is needed beyond the
        pool's own scheduling.
        """
        if not self.enabled or self._pool is None:
            raise RuntimeError("ParallelOracle.ask() before a successful start()")
        if not candidates:
            return []

        def one(index_and_candidate):
            index, candidate = index_and_candidate
            root = self._roots[index % len(self._roots)]
            target = root / candidate.path
            saved = target.read_text() if target.exists() else None
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(candidate.source)
                output, code = self._run_in(root)
                return self._oracle._matches_original(output, code)
            finally:
                if saved is None:
                    try:
                        target.unlink()
                    except OSError:
                        pass
                else:
                    target.write_text(saved)
                # The worker's stamp for this file is now meaningless;
                # force the next sync to refresh it rather than trust a
                # size/mtime that a restore may have reproduced exactly.
                self._stamps[index % len(self._roots)].pop(
                    candidate.path, None)

        # Chunked so a batch larger than the pool still maps one candidate
        # to one worker at a time.
        results: List[bool] = []
        for start in range(0, len(candidates), len(self._roots)):
            chunk = list(candidates[start:start + len(self._roots)])
            results.extend(self._pool.map(one, enumerate(chunk)))
        return results

    # ------------------------------------------------------------ teardown

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.enabled = False
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)
        if self._roots:
            parent = self._roots[0].parent
            shutil.rmtree(parent, ignore_errors=True)
        self._roots = []
        self._stamps = []

    def __enter__(self) -> "ParallelOracle":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
