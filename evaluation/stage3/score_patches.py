"""Stage 3: score model patches by running the repository's own tests.

A patch is judged by execution, never by resemblance to the ground
truth: apply it to the original repository at the instance's commit, run
the test the instance says must go from failing to passing, and if it
does, run the tests that were already passing to check they still are.

Two controls run before any model output is scored, per instance:

  * the *unpatched* tree must FAIL the target test. If it passes, the
    checkout is not the pre-fix commit and every score for that instance
    is meaningless.
  * the *ground-truth* patch must PASS it. If it does not, the harness
    is broken — wrong command, wrong environment, wrong tree — and a row
    of zeros would look exactly like a model that could not fix
    anything.

The second control is the one this project has learned to insist on.
Its whole benchmark has been voided seven times by measuring something
other than what the number claimed, and "every arm scored 0" is the most
comfortable-looking form that failure takes.

Usage:
    python evaluation/stage3/score_patches.py \
        --generations evaluation/stage3/generations.jsonl \
        --out evaluation/stage3/results_stage3.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from gistify_runner import GistifyTask, _ensure_repo  # noqa: E402
from ingest_tasks import _LABEL  # noqa: E402
from provision import provision, ProvisionError  # noqa: E402
# Private, and taken deliberately: `locate` is the public door and walks
# the whole tree per name, which is 105 walks for one sympy instance.
from provision.discovery import _walk  # noqa: E402

_HERE = Path(__file__).resolve().parent


# ------------------------------------------------------- parsing edits

_BLOCK = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n?={5,}\s*\n(.*?)\n?>{5,}\s*REPLACE",
    re.DOTALL)
# A path on its own line, with or without the ### the prompt asks for,
# with or without a fence opening after it.
_PATH = re.compile(r"^\s*(?:#{1,6}\s*)?[`\"']?"
                   r"([\w./\-]+\.py)[`\"']?\s*:?\s*$", re.MULTILINE)


@dataclass
class Edit:
    path: str
    search: str
    replace: str


def parse_edits(text: str) -> Tuple[List[Edit], Optional[str]]:
    """Pull SEARCH/REPLACE edits out of a model's reply.

    Deliberately lenient about the file path — models write `### a.py`,
    ```python:a.py, or just a.py above the block — and deliberately
    strict about the block itself. A misparsed path is recoverable by
    looking harder; a misparsed SEARCH body would silently corrupt the
    file it is applied to.
    """
    edits: List[Edit] = []
    last_end = 0
    for match in _BLOCK.finditer(text):
        before = text[last_end:match.start()]
        paths = _PATH.findall(before)
        if not paths:
            # No path since the previous block: models often emit several
            # edits to one file under a single heading.
            if edits:
                paths = [edits[-1].path]
            else:
                last_end = match.end()
                continue
        edits.append(Edit(paths[-1], match.group(1), match.group(2)))
        last_end = match.end()
    if not edits:
        return [], "no SEARCH/REPLACE block found"
    return edits, None


# ------------------------------------------------------ applying edits

def apply_edits(tree: Path, edits: List[Edit]) -> Tuple[Dict[str, str], str]:
    """Apply edits in place; return the originals and a status.

    Exact text match only. A fuzzy match would let an edit land somewhere
    the model did not mean, and "the patch applied" would stop meaning
    what it says. Whitespace-insensitive retries are counted separately
    by the caller so the reduced arm's particular failure mode — an
    anchor spanning lines the reducer deleted — stays visible instead of
    being papered over.
    """
    originals: Dict[str, str] = {}
    for edit in edits:
        path = tree / edit.path
        if not path.is_file():
            _restore(tree, originals)
            return {}, f"no such file: {edit.path}"
        text = path.read_text(encoding="utf-8", errors="replace")
        if edit.path not in originals:
            originals[edit.path] = text
        if edit.search not in text:
            _restore(tree, originals)
            return {}, "SEARCH block not found in file"
        if text.count(edit.search) > 1:
            _restore(tree, originals)
            return {}, "SEARCH block is ambiguous"
        path.write_text(text.replace(edit.search, edit.replace, 1),
                        encoding="utf-8")
    return originals, "applied"


def _restore(tree: Path, originals: Dict[str, str]) -> None:
    for rel, text in originals.items():
        (tree / rel).write_text(text, encoding="utf-8")


def apply_diff(tree: Path, patch: str, reverse: bool = False) -> bool:
    """Apply (or reverse) a unified diff — the ground-truth control only.

    Reversing is how the control cleans up after itself. `git checkout --
    .` would be the obvious move and is wrong here: the instance's
    test_patch is an *uncommitted* change in this checkout, so restoring
    from HEAD would delete the test the instance is about to be scored
    on, and every later sample would be scored against a test that no
    longer exists.
    """
    cmd = ["git", "-C", str(tree), "apply"]
    if reverse:
        cmd.append("-R")
    proc = subprocess.run(cmd + ["-"], input=patch, text=True,
                          capture_output=True)
    return proc.returncode == 0


# -------------------------------------------------------- running tests

def _purge(tree: Path) -> None:
    for cache in tree.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def run_tests(tree: Path, command: List[str], python: str,
              timeout: int = 900) -> Tuple[bool, str]:
    """True when the command reports success. Stale bytecode purged first.

    The purge is not defensive tidiness. This harness has already been
    fooled once by an interpreter answering from a .pyc for source that
    had changed on disk, which made a run judge code nobody had written.
    """
    _purge(tree)
    cmd = [python if c in ("python3", "python") else c for c in command]
    try:
        proc = subprocess.run(cmd, cwd=tree, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out[-4000:]


def collectable(tree: Path, python: str, timeout: int) -> set:
    """Every node id pytest can actually collect in this tree.

    Needed because SWE-bench's PASS_TO_PASS lists are not all runnable
    as written. pylint's contains

        tests/config/test_config.py::test_csv_regex_comma_in_quantifier[foo,

    — a parametrized id whose own parameter contains a comma, truncated
    where the export split on commas. Handing that to pytest fails the
    whole batch, which made the *ground-truth* patch look like it broke
    previously-passing tests.
    """
    proc = subprocess.run(
        [python, "-m", "pytest", "--collect-only", "-q", "--no-header"],
        cwd=tree, capture_output=True, text=True, timeout=timeout)
    return {line.strip() for line in (proc.stdout or "").splitlines()
            if "::" in line and not line.startswith(" ")}


def p2p_command(tree: Path, test_ids: List[str], python: str,
                timeout: int) -> Tuple[Optional[List[str]], List[str]]:
    """One command that runs every previously-passing test, and what it drops.

    Batched rather than run one at a time: 248 separate pytest startups
    cost minutes per sample, and the question here is only whether any of
    them broke.
    """
    if not test_ids:
        return None, []

    # One walk of the tree, not one per name: sympy names 105 bare test
    # functions and `locate` AST-parses every test file on each call.
    index: Dict[str, List[str]] = defaultdict(list)
    if any("::" not in t and not _LABEL.match(t) for t in test_ids):
        for node_id in _walk(tree):
            index[node_id.rsplit("::", 1)[-1]].append(node_id)

    labels, node_ids = [], []
    for test_id in test_ids:
        if "::" in test_id:
            node_ids.append(test_id)
        elif _LABEL.match(test_id):
            labels.append(_LABEL.match(test_id).group(2))
        else:
            found = index.get(test_id, [])
            if len(found) == 1:
                node_ids.append(found[0])

    if labels and not node_ids:
        return ["python3", "tests/runtests.py", *labels, "-v", "0"], []
    if not node_ids:
        return None, list(test_ids)

    real = collectable(tree, python, timeout)
    keep = [n for n in node_ids if n in real]
    dropped = [n for n in node_ids if n not in real]
    if not keep:
        return None, dropped
    return ["python3", "-m", "pytest", *keep, "-q"], dropped


# --------------------------------------------------------------- rig

@dataclass
class Instance:
    record: dict
    task: GistifyTask
    tree: Path
    python: str
    f2p_command: List[str]
    p2p_ids: List[str]
    controls: dict = field(default_factory=dict)
    # Set by prepare(), and only once the ground-truth patch has been
    # shown to pass it. None means this instance's P2P list could not be
    # made runnable, and no sample is judged on it.
    p2p: Optional[List[str]] = None
    p2p_dropped: List[str] = field(default_factory=list)


def prepare(record: dict, task: GistifyTask, workspace: Path,
            cache_dir: Optional[Path], timeout: int) -> Instance:
    """Check out, install, and prove the instance is scoreable."""
    task_id = record["instance_id"]
    print(f"[{task_id}] preparing...", flush=True)
    source = _ensure_repo(task, cache_dir=cache_dir)
    tree = workspace / task_id
    shutil.copytree(source, tree,
                    ignore=shutil.ignore_patterns(
                        "*.egg-info", "__pycache__", ".pytest_cache",
                        "build", "dist"))
    spec = provision(tree, workspace / f"venv-{task_id}")

    f2p = task.test_command
    p2p_ids = (json.loads(record["PASS_TO_PASS"])
               if isinstance(record["PASS_TO_PASS"], str)
               else record["PASS_TO_PASS"])

    inst = Instance(record, task, tree, spec.python, f2p, p2p_ids)
    command, dropped = p2p_command(tree, p2p_ids, spec.python, timeout)
    inst.p2p, inst.p2p_dropped = command, dropped

    passes_before, _ = run_tests(tree, f2p, spec.python, timeout)
    gold_ok = p2p_ok = False
    p2p_out = "not attempted"
    if apply_diff(tree, record["patch"]):
        gold_ok, gold_out = run_tests(tree, f2p, spec.python, timeout)
        # The ground-truth patch must also leave the previously-passing
        # tests passing. If it does not, the fault is in this rig's P2P
        # command, not in the patch — and scoring a model against it
        # would call every arm a regression.
        if command:
            p2p_ok, p2p_out = run_tests(tree, command, spec.python, timeout)
        if not apply_diff(tree, record["patch"], reverse=True):
            gold_out += "\n[control] ground-truth patch would not reverse"
            gold_ok = False
    else:
        gold_out = "ground-truth patch did not apply"
    if not p2p_ok:
        inst.p2p = None
    inst.controls = {
        "target_test_fails_before_fix": not passes_before,
        "ground_truth_patch_resolves": gold_ok,
        "gold_output_tail": "" if gold_ok else gold_out[-1500:],
        "p2p_checked": bool(inst.p2p),
        "p2p_tests_run": len(command) - 3 if command else 0,
        "p2p_ids_not_collectable": dropped,
        "p2p_output_tail": "" if p2p_ok else p2p_out[-1500:],
    }
    print(f"  fails before fix: {not passes_before}   "
          f"gold patch resolves: {gold_ok}   "
          f"p2p usable: {bool(inst.p2p)}"
          + (f" ({len(dropped)} ids not collectable)" if dropped else ""),
          flush=True)
    return inst


def score_sample(inst: Instance, text: str, timeout: int) -> dict:
    edits, parse_error = parse_edits(text)
    gold = sorted({m.group(1) for m in re.finditer(
        r"^diff --git a/(\S+) b/\S+", inst.record["patch"], re.MULTILINE)})
    edited = sorted({e.path for e in edits})
    row = {
        "n_edits": len(edits),
        "parse_error": parse_error,
        "edited_files": edited,
        "gold_files": gold,
        # Localization is scored on what the model *chose*, which exists
        # even when the edit fails to apply — and with resolve rates this
        # low it is the metric with any resolution at all.
        "located_gold_file": bool(set(edited) & set(gold)),
        "only_gold_files": bool(edited) and set(edited) <= set(gold),
        "applied": False,
        "apply_status": parse_error or "not attempted",
        "resolved": False,
        "p2p_ok": None,
    }
    if not edits:
        return row

    originals, status = apply_edits(inst.tree, edits)
    row["apply_status"] = status
    if status != "applied":
        return row
    row["applied"] = True
    try:
        ok, out = run_tests(inst.tree, inst.f2p_command, inst.python, timeout)
        row["resolved"] = ok
        if not ok:
            row["output_tail"] = out[-800:]
        elif inst.p2p:
            p2p_ok, p2p_out = run_tests(
                inst.tree, inst.p2p, inst.python, timeout)
            row["p2p_ok"] = p2p_ok
            if not p2p_ok:
                row["p2p_output_tail"] = p2p_out[-800:]
    finally:
        _restore(inst.tree, originals)
    return row


# --------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", default=str(_HERE / "generations.jsonl"))
    ap.add_argument("--instances", default=str(_HERE / "instances.json"))
    ap.add_argument("--tasks", default=str(
        _ROOT / "evaluation" / "swebench_tasks.json"))
    ap.add_argument("--out", default=str(_HERE / "results_stage3.json"))
    ap.add_argument("--workspace", default=None,
                    help="Where checkouts and virtualenvs live. A temp "
                         "directory by default; name one to keep them "
                         "between runs.")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", default=None)
    ap.add_argument("--controls-only", action="store_true",
                    help="Run every instance's controls and stop, without "
                         "scoring anything. This is the pre-flight: it "
                         "provisions each repository, checks the target "
                         "test fails before the fix and that the "
                         "ground-truth patch resolves it without breaking "
                         "the previously-passing set. An instance that "
                         "fails here cannot score any arm, and finding "
                         "that out after a GPU run costs the GPU run — "
                         "which is how the truncated PASS_TO_PASS ids "
                         "were caught.")
    args = ap.parse_args()

    records = {r["instance_id"]: r for r in
               json.loads(Path(args.instances).read_text())["instances"]}
    if args.controls_only:
        gens = [{"instance_id": iid, "arm": "-", "sample": 0, "output": ""}
                for iid in sorted(records)]
    else:
        gens = [json.loads(line) for line in
                Path(args.generations).read_text().splitlines() if line.strip()]
    tasks = {t["task_id"]: GistifyTask(**t) for t in
             json.loads(Path(args.tasks).read_text())["tasks"]}
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        gens = [g for g in gens
                if any(w in g["instance_id"] for w in wanted)]

    by_instance: Dict[str, List[dict]] = defaultdict(list)
    for gen in gens:
        by_instance[gen["instance_id"]].append(gen)

    tmp = None
    if args.workspace:
        workspace = Path(args.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="stage3_")
        workspace = Path(tmp.name)

    rows: List[dict] = []
    controls: Dict[str, dict] = {}
    start = time.time()
    try:
        for task_id in sorted(by_instance):
            try:
                inst = prepare(records[task_id], tasks[task_id], workspace,
                               Path(args.cache_dir) if args.cache_dir
                               else None, args.timeout)
            except ProvisionError as exc:
                controls[task_id] = {"error": f"provisioning failed: {exc}"}
                continue
            controls[task_id] = inst.controls
            if args.controls_only:
                c = inst.controls
                ok = (c["target_test_fails_before_fix"]
                      and c["ground_truth_patch_resolves"])
                print(f"{task_id:<24} {'PASS' if ok else 'FAIL'}  "
                      f"fails-before={c['target_test_fails_before_fix']} "
                      f"gold-resolves={c['ground_truth_patch_resolves']} "
                      f"p2p={c['p2p_tests_run']} run"
                      + (f", {len(c['p2p_ids_not_collectable'])} not "
                         f"collectable" if c["p2p_ids_not_collectable"]
                         else ""), flush=True)
                continue
            for gen in sorted(by_instance[task_id],
                              key=lambda g: (g["arm"], g["sample"])):
                row = score_sample(inst, gen["output"], args.timeout)
                row.update({"instance_id": task_id, "arm": gen["arm"],
                            "sample": gen["sample"]})
                rows.append(row)
                flag = ("RESOLVED" if row["resolved"] else
                        "applied" if row["applied"] else row["apply_status"])
                print(f"  {gen['arm']:<13} #{gen['sample']} {flag}",
                      flush=True)
    finally:
        if tmp is not None:
            tmp.cleanup()

    payload = {
        "controls": controls,
        "summary": summarize(rows, controls),
        "rows": rows,
        "elapsed_seconds": time.time() - start,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def summarize(rows: List[dict], controls: Dict[str, dict]) -> dict:
    """Per-arm rates, with resolve@1 and resolve@any kept apart.

    Only instances whose two controls passed are counted. An instance the
    rig could not score is excluded from every arm at once, so the arms
    stay comparable.
    """
    scoreable = {t for t, c in controls.items()
                 if c.get("target_test_fails_before_fix")
                 and c.get("ground_truth_patch_resolves")}
    out: Dict[str, dict] = {"scoreable_instances": sorted(scoreable),
                            "arms": {}}
    for arm in sorted({r["arm"] for r in rows}):
        got = [r for r in rows
               if r["arm"] == arm and r["instance_id"] in scoreable]
        if not got:
            continue
        by_inst: Dict[str, List[dict]] = defaultdict(list)
        for row in got:
            by_inst[row["instance_id"]].append(row)
        out["arms"][arm] = {
            "n_samples": len(got),
            "n_instances": len(by_inst),
            # Named, not just counted. An arm can cover fewer instances
            # than another for a legitimate reason — the oracle arm is
            # empty wherever the gold file is too large to show whole —
            # and a bare rate over a smaller set invites comparing two
            # arms that were not asked the same questions.
            "instances": sorted(by_inst),
            "parse_rate": sum(1 for r in got if not r["parse_error"]) / len(got),
            "apply_rate": sum(1 for r in got if r["applied"]) / len(got),
            "localization_rate": sum(
                1 for r in got if r["located_gold_file"]) / len(got),
            "resolve_at_1": sum(1 for r in got if r["resolved"]) / len(got),
            "resolve_any": sum(
                1 for rs in by_inst.values()
                if any(r["resolved"] for r in rs)) / len(by_inst),
            "localize_any": sum(
                1 for rs in by_inst.values()
                if any(r["located_gold_file"] for r in rs)) / len(by_inst),
        }
    return out


if __name__ == "__main__":
    sys.exit(main())
