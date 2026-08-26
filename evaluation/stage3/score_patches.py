"""Stage 3: score model patches by running the repository's own tests.

A patch is judged by execution, never by resemblance to the ground
truth: apply it to the original repository at the instance's commit, run
the test the instance says must go from failing to passing, and if it
does, run the tests that were already passing to check they still are.

Three controls run before any model output is scored, per instance:

  * the *unpatched* tree must FAIL the target test. If it passes, the
    checkout is not the pre-fix commit and every score for that instance
    is meaningless.
  * the *ground-truth* patch must PASS it. If it does not, the harness
    is broken — wrong command, wrong environment, wrong tree — and a row
    of zeros would look exactly like a model that could not fix
    anything.

  * the ground-truth patch, re-expressed as SEARCH/REPLACE edits and
    pushed through the same parser and the same anchor rules a sample
    goes through (`--gold-as-edits`). Applying it as a diff proves only
    that git works; this is what rules out a broken parser or an anchor
    rule nothing can satisfy.

The last two are the ones this project has learned to insist on. Its
whole benchmark has been voided seven times by measuring something other
than what the number claimed, and "every arm scored 0" is the most
comfortable-looking form that failure takes.

An instance whose controls fail is not scored at all — not merely left
out of the summary. A ground-truth patch that will not reverse leaves
the fix in the tree, and every sample run against that tree resolves.

The model is also held to the one rule the prompt states and could not
previously enforce: an edit aimed at a file the graded test suite owns
is refused and counted, never applied.

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
    orphaned = 0
    for match in _BLOCK.finditer(text):
        before = text[last_end:match.start()]
        paths = _PATH.findall(before)
        if not paths:
            # No path since the previous block: models often emit several
            # edits to one file under a single heading.
            if edits:
                paths = [edits[-1].path]
            else:
                orphaned += 1
                last_end = match.end()
                continue
        edits.append(Edit(paths[-1], match.group(1), match.group(2)))
        last_end = match.end()
    if not edits:
        # Kept apart on purpose. A model that wrote the format and forgot
        # the filename has failed at something quite different from one
        # that answered in prose, and rolling both into "no block found"
        # hides which of the two the prompt needs to fix.
        return [], ("SEARCH/REPLACE block with no file path"
                    if orphaned else "no SEARCH/REPLACE block found")
    return edits, None


# ------------------------------------------------------ applying edits

# A directory component of `test`/`tests`, or a filename pytest would
# collect. Deliberately broader than the instance's own test_patch,
# because a model does not have to touch the graded file to game the
# grade: django-17087 was handed `tests/postgres_tests/models.py`, which
# no test_patch names and which the graded test imports.
_TEST_PATH = re.compile(
    r"(?:^|/)tests?/"
    r"|(?:^|/)(?:test_[^/]*|[^/]*_test|conftest)\.py$")


def is_test_path(rel: str) -> bool:
    """True for a path the graded test suite owns.

    An edit here is refused rather than applied, and the refusal is
    counted. The prompt already says not to edit tests; nothing enforced
    it, and the prompt embeds the failing test's source verbatim, so the
    model has the anchor in front of it. Three of the eighty generations
    on disk emit an edit against the exact FAIL_TO_PASS file — deleting
    one assertion in any of them would have been scored as a fix.
    """
    return bool(_TEST_PATH.search(rel))


def apply_edits(tree: Path, edits: List[Edit]) -> Tuple[Dict[str, str], str]:
    """Apply edits in place; return the originals and a status.

    Exact text match only, and no fallback of any kind — not a
    whitespace-insensitive retry, not a nearest-match. A fuzzy match
    would let an edit land somewhere the model did not mean, and "the
    patch applied" would stop meaning what it says.

    That strictness is what keeps the reduced arm's particular failure
    mode visible: its SEARCH block is copied from a tree the reducer cut
    lines out of, so an anchor spanning a deletion cannot exist in the
    original. It has to be reported as not-applying and counted, because
    an arm whose edits land approximately is not an arm anyone can
    score.

    Edits are applied in order and rolled back as a group, so a run of
    edits across several files either all land or none do — half a patch
    would otherwise be scored as the model's whole answer.

    Bytes in, bytes out. The obvious `read_text(errors="replace")` is
    wrong twice over here: it substitutes U+FFFD for anything that will
    not decode, and it collapses CRLF to LF — and since the restore path
    writes that text back as "the original", the tree is permanently
    altered for every later sample of the instance, not just this one.
    """
    originals: Dict[str, bytes] = {}
    for edit in edits:
        if is_test_path(edit.path):
            _restore(tree, originals)
            return {}, f"edit targets a test file: {edit.path}"
        path = tree / edit.path
        if not path.is_file():
            _restore(tree, originals)
            return {}, f"no such file: {edit.path}"
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            _restore(tree, originals)
            return {}, f"file is not utf-8: {edit.path}"
        if edit.path not in originals:
            originals[edit.path] = raw
        if edit.search not in text:
            _restore(tree, originals)
            return {}, "SEARCH block not found in file"
        if text.count(edit.search) > 1:
            _restore(tree, originals)
            return {}, "SEARCH block is ambiguous"
        path.write_bytes(
            text.replace(edit.search, edit.replace, 1).encode("utf-8"))
    return originals, "applied"


def _restore(tree: Path, originals: Dict[str, bytes]) -> None:
    for rel, raw in originals.items():
        (tree / rel).write_bytes(raw)


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


@dataclass
class P2PPlan:
    """How the previously-passing set will actually be run.

    `n_tests` is counted, never inferred from the length of a command.
    It used to be `len(command) - 3`, which is off by one against both
    shapes of command and quietly published one more test than ran.
    """
    commands: List[List[str]] = field(default_factory=list)
    n_tests: int = 0
    dropped: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.commands)


def p2p_plan(tree: Path, test_ids: List[str], python: str,
             timeout: int) -> P2PPlan:
    """The commands that run every previously-passing test, and what drops.

    Batched rather than run one at a time: 248 separate pytest startups
    cost minutes per sample, and the question here is only whether any of
    them broke.

    A repository can need *both* shapes at once. django-17084 lists 93
    unittest labels alongside 14 bare names, and returning a single
    command meant one bare name resolving through the index was enough to
    take the pytest branch and silently throw all 93 labels away. The
    plan holds a command per shape instead, and everything it could not
    place goes into `dropped` under its original id — a name that
    quietly evaporates is a control reporting coverage it does not have.
    """
    if not test_ids:
        return P2PPlan()

    # One walk of the tree, not one per name: sympy names 105 bare test
    # functions and `locate` AST-parses every test file on each call.
    index: Dict[str, List[str]] = defaultdict(list)
    if any("::" not in t and not _LABEL.match(t) for t in test_ids):
        for node_id in _walk(tree):
            index[node_id.rsplit("::", 1)[-1]].append(node_id)

    labels: List[Tuple[str, str]] = []
    node_ids: List[str] = []
    dropped: List[str] = []
    for test_id in test_ids:
        if "::" in test_id:
            node_ids.append(test_id)
        elif _LABEL.match(test_id):
            labels.append((test_id, _LABEL.match(test_id).group(2)))
        else:
            found = index.get(test_id, [])
            if len(found) == 1:
                node_ids.append(found[0])
            else:
                # Matches no test function, or several. SWE-bench prints
                # a unittest method's *docstring* in place of its name
                # wherever it has one, so most of these are prose:
                # 34 of django-17029's 43 entries look like
                # "If single element in __path__, use it".
                dropped.append(test_id)

    plan = P2PPlan()
    if labels:
        # Guarded rather than assumed. `tests/runtests.py` is django's,
        # and ingest_tasks checks for it before emitting it; a unittest
        # repository without one would otherwise get a command that
        # always fails and read as a broken previously-passing set.
        if (tree / "tests" / "runtests.py").is_file():
            plan.commands.append(["python3", "tests/runtests.py",
                                  *[lab for _, lab in labels], "-v", "0"])
            plan.n_tests += len(labels)
        else:
            dropped.extend(orig for orig, _ in labels)
    if node_ids:
        real = collectable(tree, python, timeout)
        keep = [n for n in node_ids if n in real]
        dropped.extend(n for n in node_ids if n not in real)
        if keep:
            plan.commands.append(["python3", "-m", "pytest", *keep, "-q"])
            plan.n_tests += len(keep)
    plan.dropped = dropped
    return plan


def run_plan(tree: Path, plan: P2PPlan, python: str,
             timeout: int) -> Tuple[bool, str]:
    """Every command in the plan must succeed."""
    tails = []
    for command in plan.commands:
        ok, out = run_tests(tree, command, python, timeout)
        if not ok:
            tails.append(f"$ {' '.join(command[:4])} ...\n{out}")
    return not tails, "\n\n".join(tails)


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
    p2p: Optional[P2PPlan] = None

    @property
    def scoreable(self) -> bool:
        """Both controls held, so a number from this instance means something.

        Checked before any sample is scored, not only when the results
        are summarised. If the ground-truth patch fails to *reverse*, the
        tree still has the fix in it — and every sample scored against
        that tree resolves, lands in the published rows marked
        `resolved`, and looks exactly like a model that solved the
        instance fifteen times.
        """
        return bool(self.controls.get("target_test_fails_before_fix")
                    and self.controls.get("ground_truth_patch_resolves"))


def prepare(record: dict, task: GistifyTask, workspace: Path,
            cache_dir: Optional[Path], timeout: int) -> Instance:
    """Check out, install, and prove the instance is scoreable."""
    task_id = record["instance_id"]
    print(f"[{task_id}] preparing...", flush=True)
    source = _ensure_repo(task, cache_dir=cache_dir)
    tree = workspace / task_id
    # The tree is replaced rather than reused, even in a named
    # workspace. Scoring writes model edits into it and reverses them
    # again, and a run that died between those two leaves a tree whose
    # contents nobody can vouch for — which is not a starting point for
    # a measurement. Copying it back is seconds.
    #
    # The virtualenv beside it survives, which is what --workspace is
    # actually saving: `provision` runs again, but against an
    # environment whose packages are already present, so pip re-checks
    # rather than downloads.
    if tree.exists():
        shutil.rmtree(tree)
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
    plan = p2p_plan(tree, p2p_ids, spec.python, timeout)
    inst.p2p = plan

    passes_before, _ = run_tests(tree, f2p, spec.python, timeout)
    gold_ok = p2p_ok = False
    p2p_out = "not attempted"
    if apply_diff(tree, record["patch"]):
        gold_ok, gold_out = run_tests(tree, f2p, spec.python, timeout)
        # The ground-truth patch must also leave the previously-passing
        # tests passing. If it does not, the fault is in this rig's P2P
        # command, not in the patch — and scoring a model against it
        # would call every arm a regression.
        if plan:
            p2p_ok, p2p_out = run_plan(tree, plan, spec.python, timeout)
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
        "p2p_ids_listed": len(p2p_ids),
        "p2p_tests_run": plan.n_tests,
        "p2p_ids_not_collectable": plan.dropped,
        "p2p_output_tail": "" if p2p_ok else p2p_out[-1500:],
        # The instance may name several tests that must go from failing
        # to passing while the manifest carries only one command. Six of
        # the seven list exactly one, so this is empty today — but a
        # patch that fixed one of two would otherwise be published as a
        # fix, and nothing in the row would say so.
        "f2p_ids_not_run": _f2p_uncovered(record, f2p),
    }
    print(f"  fails before fix: {not passes_before}   "
          f"gold patch resolves: {gold_ok}   "
          f"p2p usable: {bool(inst.p2p)}"
          f" ({plan.n_tests}/{len(p2p_ids)} run)", flush=True)
    if plan.dropped:
        print(f"  {len(plan.dropped)} P2P ids could not be placed, "
              f"e.g. {plan.dropped[0]!r}", flush=True)
    return inst


def _f2p_uncovered(record: dict, command: List[str]) -> List[str]:
    """FAIL_TO_PASS entries the manifest's single command does not run."""
    ids = record.get("FAIL_TO_PASS") or []
    if isinstance(ids, str):
        ids = json.loads(ids)
    joined = " ".join(command)
    out = []
    for test_id in ids:
        name = (_LABEL.match(test_id).group(1) if _LABEL.match(test_id)
                else test_id.rsplit("::", 1)[-1])
        if name not in joined:
            out.append(test_id)
    return out


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
            p2p_ok, p2p_out = run_plan(
                inst.tree, inst.p2p, inst.python, timeout)
            row["p2p_ok"] = p2p_ok
            if not p2p_ok:
                row["p2p_output_tail"] = p2p_out[-800:]
    finally:
        _restore(inst.tree, originals)
    return row


# ----------------------------------------- the ground-truth positive control

def gold_as_reply(patch: str) -> Tuple[str, List[str]]:
    """Re-express a unified diff as the reply the prompt asks a model for.

    This is the positive control, and it is deliberately routed through
    the *same* parse → apply → run path a sample takes rather than
    applied as a diff. A rig that can run `git apply` proves only that
    git works; what a zero from every arm needs ruled out is a broken
    parser, an anchor rule nothing can satisfy, or a test command that
    never passes. Handing it the right answer in the model's own format
    is the only check that covers all three.

    Returns the reply text and the reasons any hunk had to be skipped.
    """
    out, skipped = [], []
    path: Optional[str] = None
    search: List[str] = []
    replace: List[str] = []

    def flush() -> None:
        if path is None or not (search or replace):
            return
        if not search:
            # An empty SEARCH matches everywhere; a pure insertion cannot
            # be expressed in this format and must be named, not dropped.
            skipped.append(f"{path}: hunk is a pure insertion")
            return
        out.append(f"### {path}\n<<<<<<< SEARCH\n"
                   + "\n".join(search) + "\n=======\n"
                   + "\n".join(replace) + "\n>>>>>>> REPLACE\n")

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            flush()
            search, replace, path = [], [], None
        elif line.startswith("--- "):
            continue                        # the a/ header, not a deletion
        elif line.startswith("+++ "):
            path = line[6:].strip() if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            flush()
            search, replace = [], []
        elif line.startswith("\\"):          # "\ No newline at end of file"
            continue
        elif line.startswith("-"):
            search.append(line[1:])
        elif line.startswith("+"):
            replace.append(line[1:])
        elif line.startswith(" ") or line == "":
            search.append(line[1:] if line else "")
            replace.append(line[1:] if line else "")
    flush()
    return "".join(out), skipped


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
    ap.add_argument("--gold-as-edits", action="store_true",
                    help="Score the ground-truth patch as if it were a "
                         "sample: re-expressed as SEARCH/REPLACE edits "
                         "and pushed through the same parser, the same "
                         "anchor rules and the same test command. This is "
                         "the positive control, and it belongs in code "
                         "rather than in a sentence about a check "
                         "somebody once ran.")
    ap.add_argument("--resume", action="store_true",
                    help="Keep the instances already in --out and score "
                         "only the rest. Scoring provisions six "
                         "repositories and runs their suites; losing an "
                         "hour of that to a laptop lid is a cost this "
                         "project has already paid twice.")
    args = ap.parse_args()

    if args.gold_as_edits and args.out == str(_HERE / "results_stage3.json"):
        # Its own file by default. The control and the run answer
        # different questions, and one overwriting the other is how a
        # results file comes to hold something nobody meant to publish.
        args.out = str(_HERE / "results_gold_as_edits.json")

    records = {r["instance_id"]: r for r in
               json.loads(Path(args.instances).read_text())["instances"]}
    if args.controls_only or args.gold_as_edits:
        gens = [{"instance_id": iid, "arm": "gold_as_edits", "sample": 0,
                 "output": ""} for iid in sorted(records)]
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
    unknown = set()
    for gen in gens:
        if gen["instance_id"] not in records or gen["instance_id"] not in tasks:
            # Named by the generations file and absent from the manifest.
            # Collected rather than raised: a KeyError here used to
            # abandon every instance already scored.
            unknown.add(gen["instance_id"])
            continue
        by_instance[gen["instance_id"]].append(gen)
    if unknown:
        print(f"WARNING: no manifest for {', '.join(sorted(unknown))} — "
              f"skipped", flush=True)

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
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        prior = json.loads(out_path.read_text())
        controls = prior.get("controls", {})
        rows = prior.get("rows", [])
        done = set(controls)
        for task_id in sorted(done & set(by_instance)):
            del by_instance[task_id]
        if done:
            print(f"Resuming: {len(done)} instance(s) already scored, "
                  f"{len(by_instance)} to go", flush=True)

    def save() -> dict:
        """Publish what is known so far, after every instance.

        The results file used to be written once, after the loop. A
        timeout, a missing manifest entry or a kill therefore threw away
        every instance already finished, and the temporary workspace
        went with it.
        """
        payload = {
            "controls": controls,
            "summary": summarize(rows, controls, quiet=True),
            "rows": rows,
            "unknown_instances": sorted(unknown),
            "elapsed_seconds": time.time() - start,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        return payload

    try:
        for task_id in sorted(by_instance):
            try:
                inst = prepare(records[task_id], tasks[task_id], workspace,
                               Path(args.cache_dir) if args.cache_dir
                               else None, args.timeout)
            except (ProvisionError, subprocess.TimeoutExpired,
                    OSError, ValueError) as exc:
                controls[task_id] = {
                    "error": f"{type(exc).__name__}: {exc}"[:400]}
                print(f"  [{task_id}] not scoreable: {type(exc).__name__}",
                      flush=True)
                save()
                continue
            controls[task_id] = inst.controls
            if args.controls_only:
                c = inst.controls
                print(f"{task_id:<24} {'PASS' if inst.scoreable else 'FAIL'}  "
                      f"fails-before={c['target_test_fails_before_fix']} "
                      f"gold-resolves={c['ground_truth_patch_resolves']} "
                      f"p2p={c['p2p_tests_run']}/{c['p2p_ids_listed']} run",
                      flush=True)
                save()
                continue
            if not inst.scoreable:
                # Not merely excluded from the summary — not run at all.
                # A failed reverse leaves the fix in the tree, and every
                # sample scored against it resolves.
                print(f"  [{task_id}] controls failed; no sample scored",
                      flush=True)
                save()
                continue
            for gen in sorted(by_instance[task_id],
                              key=lambda g: (g["arm"], g["sample"])):
                text = gen["output"]
                if args.gold_as_edits:
                    text, skipped = gold_as_reply(inst.record["patch"])
                    if skipped:
                        print(f"  [{task_id}] {len(skipped)} hunk(s) not "
                              f"expressible: {skipped[0]}", flush=True)
                row = score_sample(inst, text, args.timeout)
                row.update({"instance_id": task_id, "arm": gen["arm"],
                            "sample": gen["sample"]})
                rows.append(row)
                flag = ("RESOLVED" if row["resolved"] else
                        "applied" if row["applied"] else row["apply_status"])
                print(f"  {gen['arm']:<13} #{gen['sample']} {flag}",
                      flush=True)
            save()
    finally:
        if tmp is not None:
            tmp.cleanup()

    payload = save()
    payload["summary"] = summarize(rows, controls)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def summarize(rows: List[dict], controls: Dict[str, dict],
              quiet: bool = False) -> dict:
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
            # Every reason an edit did not land, counted. A single
            # apply_rate says how often the arm worked; this says what
            # went wrong when it did not, and it is the only place the
            # refused test-file edits are visible.
            "apply_status": {
                status: sum(1 for r in got if r["apply_status"] == status)
                for status in sorted({r["apply_status"] for r in got})},
            "edits_refused_as_test_files": sum(
                1 for r in got
                if r["apply_status"].startswith("edit targets a test file")),
        }

    # Two arms asked different questions do not compare, and nothing
    # about the table above says so — every rate is a clean fraction
    # whatever it was computed over. A generation run that ran out of
    # session time partway through leaves exactly this shape: the arms
    # generated first are complete, the last one is not, and the last
    # one looks like it simply did worse.
    #
    # The oracle arm is a legitimate exception, since it cannot be built
    # where the gold file exceeds the budget, so the warning names the
    # difference rather than refusing to report.
    coverage = {arm: set(data["instances"]) for arm, data in
                out["arms"].items()}
    widest = max(coverage.values(), key=len, default=set())
    uneven = {arm: sorted(widest - seen)
              for arm, seen in coverage.items() if seen != widest}
    if uneven:
        out["uneven_coverage"] = uneven
        if not quiet:
            print("\nWARNING: arms do not cover the same instances, so their "
                  "rates are not directly comparable:")
            for arm, missing in sorted(uneven.items()):
                print(f"  {arm} is missing {', '.join(missing)}")

    samples = {arm: data["n_samples"] for arm, data in out["arms"].items()}
    if len(set(samples.values())) > 1:
        out["uneven_sample_counts"] = samples
        if not quiet:
            print(f"\nWARNING: arms have different sample counts: {samples}")

    return out


if __name__ == "__main__":
    sys.exit(main())
