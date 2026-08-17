"""Stage 3: build the context each arm hands the model.

The experiment behind this file is one question. A repair model cannot
read a 457,000-line repository, so something has to choose which few
thousand lines it sees. The standard answer is BM25 retrieval against
the issue text — SWE-bench's own published baseline. autoMRE proposes a
different answer: reduce the repository first, then retrieve from what
survives.

So all three arms get the *same* token budget and the same prompt, and
differ only in which tree the budget is filled from:

  full_bm25     BM25 over the original repository (the baseline)
  reduced_bm25  BM25 over the autoMRE-reduced tree (the proposal)
  oracle        the files the ground-truth patch edits (the ceiling)

`oracle` is not a competitor; it is the calibration. If reduced_bm25
lands near it, retrieval has stopped being the bottleneck, and knowing
that is the difference between a result and a number.

## What reduced_bm25 can and cannot be scored on

It measures **localisation**, not repair, and that is structural rather
than a weakness of the model. The reducer keeps whatever reproduces the
failure, which is not the same as whatever is needed to fix it.
django-17029's gold patch adds one line to `clear_cache`; in the
reduced tree that method is

    def clear_cache(self):
        pass

— sound, because the test asserts a cache was *not* cleared and an
empty body fails it identically. A model reading that can name the
method and still cannot write a SEARCH block that exists in the
original file, nor reconstruct the body it was never shown. Across the
seven instances, three have no gold-patch anchor line surviving at all.

Two ways of recovering applicability were built and measured, and both
are **worse than the baseline at the thing that bounds everything
else** — getting the buggy file into the budget:

  rank on reduced, show original text            0.190 mean recall
  rank on reduced, original where it fits        0.333
  full_bm25 (baseline)                           0.429
  reduced_bm25                                   0.929

The cause is the same both times, and it is not a tuning problem: the
ranking is good — the gold file lands at rank 1 to 5 — but four
full-size original files fill 16,000 tokens before the fifth is
reached. Showing original text and fitting the budget are in direct
conflict, which is the project's own thesis arriving from the other
side. Neither arm is generated, because a negative result this
consistent does not need a GPU to confirm.

Note what is measurable here with no GPU at all: whether the file the
ground-truth patch edits is even *present* in the context. A model
cannot fix what it was not shown, so retrieval recall bounds every
downstream score, and it is the cheapest honest thing this experiment
produces.

Usage:
    python evaluation/stage3/build_contexts.py \
        --reduced evaluation/reduced --out evaluation/stage3/contexts.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "evaluation"))

from gistify_runner import GistifyTask, _ensure_repo  # noqa: E402

_HERE = Path(__file__).resolve().parent

# Qwen2.5-Coder-7B-Instruct is the model the Kaggle side loads; budgets
# measured with any other tokenizer would describe a context that does
# not fit the window they are supposed to fit.
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Tokens of *code*. The prompt's instructions and the issue text sit on
# top of this, and generation needs room after it. 16k against a 32k
# window leaves both, and sits inside the 13k–50k band SWE-bench's own
# retrieval baselines used.
DEFAULT_BUDGET = 16_000


# ----------------------------------------------------------------- BM25

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def code_tokens(text: str) -> List[str]:
    """Words, plus the pieces of every compound identifier.

    An issue says "clear_cache" and the file says `def clear_cache` — but
    it may equally say "clearing the cache" while the file says
    `clearCache`. Splitting on underscores and camel-case boundaries and
    keeping *both* the whole identifier and its parts lets either match,
    at the cost of a slightly longer posting list.
    """
    out: List[str] = []
    for word in _WORD.findall(text):
        low = word.lower()
        out.append(low)
        parts = [p for chunk in word.split("_") if chunk
                 for p in _CAMEL.split(chunk)]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


class BM25:
    """Okapi BM25, ~40 lines, so the experiment adds no dependency.

    Standard parameters (k1=1.2, b=0.75). Nothing here is novel; it is
    here to be the *baseline*, and a baseline nobody can inspect is not
    worth reporting against.
    """

    def __init__(self, docs: Dict[str, str], k1: float = 1.2,
                 b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: Dict[str, Counter] = {
            name: Counter(code_tokens(text)) for name, text in docs.items()}
        self.length = {name: sum(c.values()) for name, c in self.tf.items()}
        n_docs = len(self.tf) or 1
        self.avg_len = (sum(self.length.values()) / n_docs) or 1.0
        df: Counter = Counter()
        for counts in self.tf.values():
            df.update(counts.keys())
        self.idf = {
            term: math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()}

    def rank(self, query: str) -> List[Tuple[str, float]]:
        terms = code_tokens(query)
        scored = []
        for name, counts in self.tf.items():
            length = self.length[name]
            score = 0.0
            for term in terms:
                freq = counts.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * length / self.avg_len)
                score += self.idf[term] * freq * (self.k1 + 1) / denom
            scored.append((name, score))
        # Ties broken by path so a rebuild produces the same context; a
        # dict-order tiebreak would make two runs incomparable for a
        # reason that has nothing to do with retrieval.
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored


# ------------------------------------------------------------- the tree

_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist",
              ".tox", ".mypy_cache", "node_modules"}


def collect_python(tree: Path) -> Dict[str, str]:
    """Every readable .py file in the tree, keyed by relative path."""
    files: Dict[str, str] = {}
    for path in sorted(tree.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.relative_to(tree).parts):
            continue
        if any(part.endswith(".egg-info")
               for part in path.relative_to(tree).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Undecodable sources are the reducer's known blind spot too;
            # skipping them here keeps both arms describing one universe.
            continue
        files[str(path.relative_to(tree))] = text
    return files


def gold_files(patch: str) -> List[str]:
    """The files the ground-truth patch edits, from its diff headers."""
    return sorted({m.group(1) for m in
                   re.finditer(r"^diff --git a/(\S+) b/\S+", patch,
                               re.MULTILINE)})


def failing_test_name(test_id: str) -> str:
    """The function name out of any of the three identifier shapes.

    pytest node ids, django's "test_x (module.Class.test_x)" labels and
    sympy's bare names all end in the function name; only the punctuation
    around it differs.
    """
    head = test_id.split(" ")[0].strip("()")
    return head.split("::")[-1].split(".")[-1]


def extract_test_source(files: Dict[str, str], test_ids: List[str],
                        limit: int = 400) -> str:
    """The source of the failing tests, and only those.

    Whole test files are not usable here: seaborn's is two thousand
    lines, which would be an eighth of the budget spent on tests nobody
    asked about. The named function is the reproduction.
    """
    import ast
    wanted = {failing_test_name(t) for t in test_ids}
    chunks: List[str] = []
    for name, text in files.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in wanted):
                src = ast.get_source_segment(text, node) or ""
                lines = src.splitlines()[:limit]
                chunks.append(f"### {name}\n```python\n"
                              + "\n".join(lines) + "\n```\n")
    return "\n".join(chunks)


# ------------------------------------------------------------- packing

@dataclass
class Context:
    text: str
    included: List[str]
    tokens: int
    considered: int          # files the arm could have drawn from


def render(name: str, text: str) -> str:
    return f"### {name}\n```python\n{text}\n```\n"


def token_costs(files: Dict[str, str], count_tokens_batch) -> Dict[str, int]:
    """Token cost of every file, measured once.

    Measured rather than estimated, and measured up front rather than
    inside the packing loop: django's tree is 2,800 files and the loop
    walks all of them, so tokenizing on demand would tokenize half a
    million lines to decide the last few hundred tokens.
    """
    names = list(files)
    bodies = [render(n, files[n]) for n in names]
    return dict(zip(names, count_tokens_batch(bodies)))


def pack(ranked: List[str], files: Dict[str, str], budget: int,
         costs: Dict[str, int]) -> Context:
    """Fill the budget with whole files, best-ranked first.

    Whole files, never half of one: a file cut off mid-function reads to
    a model as a file whose rest does not exist, and an arm that fills
    its last 400 tokens with a truncated class has spent them making the
    context wrong rather than smaller. When the next file does not fit,
    the loop keeps going — a later, smaller file may.
    """
    chunks: List[str] = []
    included: List[str] = []
    used = 0
    for name in ranked:
        cost = costs[name]
        if used + cost > budget:
            continue
        chunks.append(render(name, files[name]))
        included.append(name)
        used += cost
    return Context("\n".join(chunks), included, used, len(files))


# -------------------------------------------------------------- prompt

_INSTRUCTION = """\
You are fixing a bug in a Python repository.

Below is an issue report, the test that currently fails because of it,
then the parts of the repository you have been given. The repository is
larger than what is shown; assume anything not shown is unchanged and
correct.

Work out which shown code is responsible for the issue and repair it.

Reply with one or more edits, and nothing else. Each edit is exactly:

### path/to/file.py
<<<<<<< SEARCH
the exact lines to replace, copied verbatim from the file above
=======
the lines to put there instead
>>>>>>> REPLACE

Rules:
- The SEARCH block must appear verbatim in the file shown, including
  indentation. It is located by exact text match, so an approximation
  will not apply.
- Keep SEARCH blocks small: enough lines to be unique, no more.
- Do not edit test files, and do not explain yourself.
"""


def build_prompt(problem: str, failing_test: str, context: str) -> str:
    return (f"{_INSTRUCTION}\n"
            f"## Issue\n{problem.strip()}\n\n"
            f"## Failing test\n{failing_test}\n"
            f"## Repository code\n{context}")


# ---------------------------------------------------------------- main

def load_reduced(reduced_root: Path, task_id: str) -> Optional[Path]:
    tree = reduced_root / task_id / "tree"
    return tree if tree.is_dir() else None


def build(instance: dict, task: GistifyTask, reduced_root: Path,
          budget: int, count_tokens, count_tokens_batch,
          cache_dir: Optional[Path]) -> List[dict]:
    task_id = instance["instance_id"]
    problem = instance["problem_statement"]
    gold = gold_files(instance["patch"])
    f2p_ids = (json.loads(instance["FAIL_TO_PASS"])
               if isinstance(instance["FAIL_TO_PASS"], str)
               else instance["FAIL_TO_PASS"])

    print(f"[{task_id}] checking out the original tree...", flush=True)
    original = _ensure_repo(task, cache_dir=cache_dir)
    full = collect_python(original)

    reduced_tree = load_reduced(reduced_root, task_id)
    if reduced_tree is None:
        raise SystemExit(
            f"{task_id}: no reduced tree at {reduced_root / task_id}. "
            f"Run gistify_runner.py --save-reduced first.")
    reduced = collect_python(reduced_tree)

    # The confound this removes would have decided the whole experiment.
    # The reduced tree *keeps* the failing test — the reducer protects
    # it, it is the question being asked — so the reduced arm would
    # arrive holding the reproduction while BM25 over the full tree,
    # ranking against an issue report, usually would not. The reduced arm
    # would then win for a reason that has nothing to do with reduction.
    #
    # So the test files leave both universes, and the failing test's
    # source is prepended to every arm instead. Every arm gets the
    # reproduction, which is the premise of this whole project anyway:
    # you have a test that fails, and you want the bug fixed.
    test_files = gold_files(instance["test_patch"])
    failing_test = extract_test_source(
        {n: t for n, t in full.items() if n in test_files}, f2p_ids)
    for name in test_files:
        full.pop(name, None)
        reduced.pop(name, None)

    rows = []
    for arm, files in (("full_bm25", full), ("reduced_bm25", reduced)):
        ranked = [name for name, _ in BM25(files).rank(problem)]
        ctx = pack(ranked, files, budget,
                   token_costs(files, count_tokens_batch))
        # Where the answer ranked, not just whether it fit. Recall alone
        # is decided at the budget boundary — on pylint the buggy file
        # ranked 13th of 2,187 and the budget held 11 — so a rank makes
        # the retrieval comparison survive a different budget, and stops
        # this from being a report on one arbitrary number.
        rank_of = {name: i for i, name in enumerate(ranked)}
        rows.append((arm, files, ctx,
                     {g: rank_of.get(g) for g in gold if g in files}))

    # The ceiling: the files the answer is in, and nothing else.
    oracle_files = {g: full[g] for g in gold if g in full}
    rows.append(("oracle", oracle_files,
                 pack(sorted(oracle_files), oracle_files, budget,
                      token_costs(oracle_files, count_tokens_batch)),
                 {g: i for i, g in enumerate(sorted(oracle_files))}))

    # A gold file can be too large to show *at all*: sympy's
    # `core/numbers.py` is 35,182 tokens, which is not merely over this
    # budget, it is over the whole 32,768-token context window of the
    # model. No retriever and no budget puts that file in front of the
    # model; only cutting it down does, and after reduction it is 11,512.
    #
    # That is the finding, but it also breaks the ceiling: the oracle arm
    # for such an instance packs nothing and would be scored as a model
    # that answered badly rather than an arm that cannot be built. So it
    # is recorded per row and the scorer is expected to set it aside.
    full_costs = token_costs({g: full[g] for g in gold if g in full},
                             count_tokens_batch)
    oversize = sorted(g for g, c in full_costs.items() if c > budget)

    out = []
    for arm, files, ctx, ranks in rows:
        hit = [g for g in gold if g in ctx.included]
        prompt = build_prompt(problem, failing_test, ctx.text)
        out.append({
            "instance_id": task_id,
            "arm": arm,
            "prompt": prompt,
            "context_tokens": ctx.tokens,
            "failing_test_tokens": count_tokens(failing_test),
            "prompt_tokens": count_tokens(prompt),
            "included_files": ctx.included,
            "files_considered": ctx.considered,
            "tree_lines": sum(len(t.splitlines()) for t in files.values()),
            "gold_files": gold,
            "gold_files_present_in_tree": [g for g in gold if g in files],
            "gold_files_in_context": hit,
            "gold_file_ranks": ranks,
            "recall": len(hit) / len(gold) if gold else 0.0,
            "gold_files_oversize": oversize,
            "context_empty": not ctx.included,
        })
        best = min([r for r in ranks.values() if r is not None],
                   default=None)
        note = ""
        if not ctx.included:
            note = ("   EMPTY — no file fits whole"
                    + (f"; {len(oversize)} gold file(s) exceed the budget"
                       if oversize else ""))
        print(f"  {arm:<13} {ctx.tokens:>6} tok  "
              f"{len(ctx.included):>3}/{ctx.considered} files  "
              f"recall {len(hit)}/{len(gold)}"
              f"   best gold rank "
              f"{'-' if best is None else best + 1}{note}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default=str(_HERE / "instances.json"))
    ap.add_argument("--tasks", default=str(
        _ROOT / "evaluation" / "swebench_tasks.json"))
    ap.add_argument("--reduced", default=str(
        _ROOT / "evaluation" / "reduced"))
    ap.add_argument("--out", default=str(_HERE / "contexts.jsonl"))
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--only", default=None,
                    help="Comma-separated instance-id substrings.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def count_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def count_tokens_batch(texts: List[str]) -> List[int]:
        if not texts:
            return []
        return [len(ids) for ids in
                tok(texts, add_special_tokens=False)["input_ids"]]

    instances = json.loads(Path(args.instances).read_text())["instances"]
    tasks = {t["task_id"]: GistifyTask(**t) for t in
             json.loads(Path(args.tasks).read_text())["tasks"]}
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        instances = [i for i in instances
                     if any(w in i["instance_id"] for w in wanted)]

    rows: List[dict] = []
    for inst in instances:
        rows.extend(build(
            inst, tasks[inst["instance_id"]], Path(args.reduced),
            args.budget, count_tokens, count_tokens_batch,
            Path(args.cache_dir) if args.cache_dir else None))

    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(rows)} contexts to {args.out}")

    for arm in ("full_bm25", "reduced_bm25", "oracle"):
        got = [r for r in rows if r["arm"] == arm]
        if not got:
            continue
        print(f"{arm:<13} mean recall "
              f"{sum(r['recall'] for r in got) / len(got):.3f}  "
              f"gold file present in "
              f"{sum(1 for r in got if r['recall'] > 0)}/{len(got)}  "
              f"empty context {sum(1 for r in got if r['context_empty'])}")

    too_big = {r["instance_id"]: r["gold_files_oversize"]
               for r in rows if r["gold_files_oversize"]}
    if too_big:
        print("\nGold files that do not fit the budget whole — no arm can "
              "show these unreduced:")
        for iid, names in sorted(too_big.items()):
            print(f"  {iid:<24} {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
