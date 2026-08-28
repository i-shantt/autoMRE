# autoMRE

**Automatically builds a minimal reproducible example (MRE) from a
Python project.** Give it your project and one test; it hands back the
smallest version of that project where the test still behaves exactly the
same way.

Everything the test does not need is deleted — whole files, then
definitions, then individual statements. After every single deletion the
test is run again, and the deletion is kept only if the output is
unchanged. Nothing is guessed: every line that survives has been proven
necessary by re-running your own test.

```
psf/requests, reduced to keep one test passing

  36 files, 11,209 lines   →   10 files, 1,166 lines
                               of which 1,016 are the test file itself,
                               which is never touched

  so: 10,193 lines of library  →  150 lines
```

---

## Why this exists

When you hit a bug in a large codebase, the useful thing to hand a
maintainer — or a bug tracker, or a language model — is not the
codebase. It is the twenty lines that actually go wrong. Producing those
twenty lines by hand means repeatedly guessing what is irrelevant and
deleting it, and checking each time that you have not accidentally
deleted the bug.

That loop is mechanical, so it can be automated. The technique is called
*delta debugging* (Zeller & Hildebrandt, 2002) and it is well understood
for single files. autoMRE applies it across a whole project, using
coverage data to decide what to try deleting first.

What comes out is a **minimal reproducible example** — the same thing a
maintainer means when they ask you to attach one, except produced by
measurement rather than by guessing.

---

## How it works

The whole method is one loop:

> Delete something. Run the test. If the output is byte-for-byte what it
> was before, keep the deletion. Otherwise put it back.

Everything else is about making that loop cheap enough to run thousands
of times, and about making sure it cannot be fooled.

**One rule matters more than the rest.** The test you name is never
modified. It is the *question* — the definition of the behavior being
preserved — not part of the answer. Left unprotected, the loop finds a
much easier solution: delete the assertions. The test then passes
trivially, and the entire library can be deleted after it. That is not a
hypothetical; see [The reducer gutted the test](#the-reducer-gutted-the-test-instead-of-the-code).

### The five phases

| Phase | What it does |
|---|---|
| **1 — Analysis** | Run the test once under `coverage.py`. Record which lines ran. Parse imports into a dependency graph. Label every file `EXECUTED`, `IMPORTED_ONLY`, or `UNREACHABLE`. |
| **2 — Delete files** | Delete every `UNREACHABLE` file at once and check. If that breaks the test, fall back to one file at a time. Then probe each `IMPORTED_ONLY` file. |
| **3 — Inlining** | Try folding modules into their importers. *Abandoned after three consecutive failures* — see below. |
| **4 — Shrink files** | The bulk of the work. Drop definitions with zero coverage in one query per file (4a), then delta-debug what survives, statement by statement (4b). |
| **5 — Final sweep** | Retry deleting whole files. Phase 2 asked while everything still imported everything; Phase 4 is what removed the last import. |

Phase 3 is kept but almost never fires. Inlining a module of an
installed package cannot work — folding `src/flask/app.py` into
`__init__.py` breaks every `from flask.app import X` — and on flask all
twenty attempts rolled back at one wasted query each. Rather than
hard-code a rule that might be wrong elsewhere, it gives up after three
failures in a row.

**Phase 4b is ~97% of all queries**, so that is where the cost lives.
It works in two stages: a batch pass asks "can all of this go at once?"
and halves on refusal, stripping bulk cheaply; then a per-unit pass
walks every remaining candidate. The halving stops at groups of 8,
which is not an arbitrary constant — see [the measurement lesson
below](#the-halving-descent).

---

## Try it

### In half a minute

```bash
pip install -e ".[dev]"      # plain `pip install -e .` if you don't want the tests
./entrypoint.sh
```

Runs the test suite, then reduces a bundled four-file example down to
the six lines that still raise the same `TypeError`. It does not
reproduce the benchmark below — that clones three repositories and takes
hours — it prints the command for it.

### In a browser

`web/` contains a small web app: drop in a zipped project, name the
test, watch it shrink, download the result.

**There is no hosted instance yet** — run it locally with two commands
(worker, then UI), both in [`web/README.md`](web/README.md). The UI is a
Next.js app built to deploy to Vercel, but the engine cannot go there:
a reduction runs thousands of subprocesses over minutes to hours, and
Vercel functions cap at 800 s with an ephemeral filesystem. So the UI
ships to Vercel and the engine ships as a container to somewhere that
can hold a long job. Both halves are written and tested; neither is
deployed, because deploying a service that executes uploaded code needs
a sandbox this does not yet have.

### On the command line

```bash
pip install -e .

automre reduce-project ./my_project \
  -c "python3 -m pytest tests/test_bug.py::test_foo -x -q" \
  -v
```

Your project is copied first; the original is not touched. Use
`--in-place` if you want the opposite.

Use the `automre` console script rather than `python3 -m automre.src.cli`.
The repo root holds an `automre.py` shim next to the `automre/` package
directory, which has no `__init__.py` and is therefore a namespace
package — and a namespace package loses to a same-named module on the
same path entry. From the repo root the module form fails with
"`automre` is not a package"; the console script has no such ambiguity.

### The rest

```bash
# Tests (127, ~55 s). pytest comes from the [dev] extra above; it is not
# a runtime dependency, because autoMRE runs your reproduction command
# in your environment rather than in its own.
python3 -m pytest automre/tests/

# Full benchmark. Provisions a pinned venv on first run.
python3 evaluation/gistify_runner.py

# Or one repo's worth
python3 evaluation/gistify_runner.py --only requests
```

---

## What it needs

Worth stating plainly, because two of these are the opposite of what
people usually assume about a tool with an `ml/` directory in it.

| | |
|---|---|
| **GPU** | None. Not for anything. The optional learned oracle is a scikit-learn model that runs on CPU in microseconds. |
| **CPU** | **One core.** Sampled during a live run: the parent process sits at 0% waiting, and exactly one child test process holds one core at 100%. |
| **Memory** | ~13 MB for autoMRE itself, plus whatever your own test needs — that subprocess is the only thing here with a real footprint. |
| **Disk** | A copy of your project, plus a virtualenv if one is being provisioned for it. |
| **Python** | `requires-python = ">=3.10"`. Everything in this file was measured on 3.13, which is the only version currently exercised. |

**A bigger machine will not make one reduction faster.** Reduction is a
strictly sequential loop — delete, run the test, decide, repeat — and it
saturates a single core no matter how many are available. Sixteen cores
finish one reduction in exactly the time one core does. What extra cores
*do* buy is more reductions at once, which is why the benchmark harness
runs tasks back to back rather than trying to split one. (A speculative
parallel Phase 4b exists on the `parallel-oracle` branch and reached
1.33× peak; it is not merged.)

### The cost is time, and it is arithmetic rather than a spec

Two numbers decide everything, and only one of them is about your
hardware:

```
queries      ≈  0.11 to 0.48  ×  (lines of Python in your project)
wall clock   ≈  queries  ×  (how long your test takes to run once)
```

Both ranges are measured across the ten benchmark tasks below. `requests`
sits at the bottom (0.11 queries/line), `tomlkit` at the top (0.48) —
an eightfold spread on the same tool, which is why autoMRE reports
progress as a range and never as a countdown.

Worked: 11,209 lines of `requests` × 0.11 ≈ 1,200 queries, and its test
runs in 0.18 s → **3.6 minutes**, which is what it actually takes.

**The multiplier is the thing to look at before you start.** Your test's
runtime is paid roughly a thousand times over. A 0.2-second test on a
10,000-line project is about four minutes; the same project with a
5-second test is about two hours. If your reproduction command is slow,
make it faster before reducing — narrowing `pytest tests/` down to a
single `::test_name` is usually the whole fix, and it is free.

### One real constraint on the test itself

Your reproduction command runs **thousands of times**, in a copy of your
project, concurrently with nothing. It therefore has to be safe to
repeat: no dependence on a network service that will rate-limit you, no
writes to a shared database, no reliance on state left behind by a
previous run. A test that behaves differently on its second run is
rejected before reduction starts rather than quietly producing nonsense —
see [the readiness gate](#pointing-it-at-a-repository-it-has-never-seen).

---

## Results

Ten pytest tasks across `psf/requests v2.32.3`, `pallets/flask 3.0.3`
and `python-poetry/tomlkit 0.13.2`
(`evaluation/gistify_tasks.json`), on an M4 MacBook Air.

| task | lines | reduction | files | queries | timeouts |
|------|------:|----------:|:-----:|--------:|---------:|
| requests-super_len_partial   | 11209 → 1166 | 89.6% | 36 → 10 | 1184 | 0 |
| requests-guess_json_utf      | 11209 → 1185 | 89.4% | 36 → 10 | 1221 | 0 |
| requests-content_disposition | 11209 → 1179 | 89.5% | 36 → 10 | 1202 | 0 |
| requests-cookie_utils        | 11209 → 1190 | 89.4% | 36 → 10 | 1248 | 0 |
| flask-request_ctx_basic      | 17565 → 2427 | 86.2% | 82 → 13 | 2489 | 6 |
| flask-blueprint_registration | 17565 → 1896 | 89.2% | 82 → 17 | 3037 | 6 |
| tomlkit-write_backslash      | 8621 → 746  | 91.3% | 28 → 14 | 2647 | 10 |
| tomlkit-parse_examples       | 8621 → 958  | 88.9% | 28 → 11 | 2572 | 14 |
| tomlkit-array_items          | 8621 → 1798 | 79.1% | 28 → 14 | 3172 | 52 |
| tomlkit-document_dict        | 8621 → 2301 | 73.3% | 28 → 15 | 4159 | 94 |

**87.03% aggregate line reduction, 10/10 execution fidelity.** Raw data
in `evaluation/results_gistify_decorators.json` (requests, flask) and
`evaluation/results_gistify_tomlkit_fixed.json` (tomlkit).

Re-run in one pass after the reduction pipeline moved onto the shared
[readiness gate](#pointing-it-at-a-repository-it-has-never-seen):
`evaluation/results_gistify_provision.json`. Every row above came back
identical — the same surviving lines, the same file counts, the same
22,931 queries and the same 182 timeouts. That is the point of running
it. A refactor that touches how the environment is built and how the
baseline is captured is exactly the kind that moves numbers quietly,
and the only way to know it did not is to spend the 3.3 hours and diff
the table.

### What the last two fixes bought

Every task above was re-measured after the [decorated-definition
fix](#decorated-code-could-never-be-removed) and the [timeout
fix](#a-hung-query-cost-two-minutes), against the same ten tasks on the
same machine:

| | before | after |
|---|---:|---:|
| aggregate reduction | 86.53% | **87.03%** |
| reducible-only | 95.23% | **95.77%** |
| surviving lines | 15,413 | **14,846** |
| queries | 23,326 | **22,931** |
| wall clock | 6.97 h | **3.34 h** |
| execution fidelity | 10/10 | 10/10 |
| timeouts | *not counted* | 182 |

**The two fixes cannot be told apart from this table**, and it would be
convenient but wrong to imply otherwise: they shipped together, so the
567-line improvement is not attributable to either alone without an
ablation that has not been run.

What *can* be said is directional. The decorator fix has a mechanism for
improving reduction — it makes code removable that previously was not.
The timeout fix has no such mechanism, and can only ever make reduction
worse: a shorter limit means a candidate that used to finish at 90 s now
gets refused. Since reduction improved rather than degraded, the
decorator fix is doing the work and the tighter limit cost nothing
measurable. That is a weaker claim than an ablation, and it is stated as
one.

The 52% wall-clock cut is unambiguous, and it is the timeout fix.

### Reading these numbers

The headline percentage **undersells the tool**, because most of what
survives is the test file, which is deliberately never touched. On
`requests-super_len_partial`, 1,016 of the 1,166 surviving lines are the
test and its conftest.

Measured over the code actually eligible for removal, reduction is
**95.8%**:

| repo | tasks | aggregate | reducible-only | fidelity |
|---|:-:|---:|---:|:-:|
| `psf/requests`        | 4 | 89.5% | **98.4%** | 4/4 |
| `pallets/flask`       | 2 | 87.7% | 96.7% | 2/2 |
| `python-poetry/tomlkit` | 4 | 83.2% | **91.4%** | 4/4 |

Both numbers are worth having: the aggregate says what the project looks
like afterwards, the reducible-only figure says what the tool did.

### Why tomlkit is here

It is not one of the paper's repos. It was added so that
leave-one-repo-out has three folds instead of two, and it was chosen to
have a different shape: a flat package layout rather than `src/`, and
tests driven by real `.toml` fixture files.

It earns its place by **disagreeing** with the other two. It reduces
**7 points worse than requests** (91.4% against 98.4% reducible-only),
and the spread *within* tomlkit is wider than the spread across all six
requests and flask tasks combined. With two repos, ~97.5% looked like a
property of the method. The third fold said it was partly a property of
the codebases the method had been pointed at.

The disagreement survived both recent fixes, which is the useful part.
tomlkit gained the most of any repo from them — 1.3 points of aggregate
reduction, against 0.1 on requests — and it is still the outlier. So the
gap is not an artifact of the two defects; something about a
hand-written recursive-descent parser resists this technique in a way
`requests` does not.

Nothing was tuned for tomlkit or away from it: the pipeline is
byte-identical across all ten tasks. That is the single most useful
thing the third repo bought, and it is an argument for a fourth.

### Reproducibility

Repeated runs of a task are bit-identical, in output and in query count,
with no pinned hash seed. That is a recent property rather than an
assumed one — it took [finding a soundness
bug](#the-oracle-judged-code-that-was-not-on-disk) to get there.

One caveat, new: the per-query time limit is now derived from how long
this project's queries actually take, so it is machine-dependent where
it used to be a fixed constant. In practice this changes no verdicts —
with a 10-second floor against a sub-second query, the only thing that
crosses the limit is a candidate that stopped terminating, and those do
not finish on a faster machine either.

---

## Where the time goes

Reduction is the expensive part: ~2,290 validation queries per task,
22,931 across the benchmark. Timing every function on the reduction path
accounts for that time to within half a percent:

| | requests-guess_json_utf | flask-request_ctx_basic |
|---|---:|---:|
| reduction wall | 262.2 s | 1288.5 s |
| in the oracle subprocess | 258.0 s (**98.4%**) | 1281.4 s (**99.4%**) |
| in the coverage tracer | — | 5.3 s (0.4%) |
| mean per subprocess | 195.6 ms | 490.6 ms |
| everything else | 3.19 ms/query | 0.69 ms/query |

*(Measured on the pre-fix pipeline; the shares are structural and have
not moved, the absolute figures have — flask is now 582 s.)*

**The reducer's own work is a rounding error.** What differs between
repos is the price of a single question: 490 ms on flask against 196 ms
on requests, because every query pays a fresh interpreter start and a
fresh `import flask`. Asking fewer questions, or making one cheaper, is
therefore the entire lever. There is no hot spot in the reducer to go
find.

Breaking down one flask query on the unreduced tree, median of seven:
11 ms to start the interpreter, 62 ms once `pytest` is imported, 103 ms
once `flask` is too, 202 ms after collection — and 212 ms to actually
run the test. **The test body is about 5% of the query.** The other 95%
is identical work repeated thousands of times. That makes a persistent
worker the obvious idea and also a dangerous one: reusing a process
means reusing `sys.modules`, which is the [stale-bytecode
failure](#the-oracle-judged-code-that-was-not-on-disk) again in a form
`PYTHONDONTWRITEBYTECODE` cannot fix. Not attempted.

Raw per-function timings: `evaluation/profile_reduction_paths.json`.

---

## Why these numbers are trustworthy

**A minimizer is extremely easy to fool, including by accident.** Its
whole job is to delete things and then check whether anything broke — so
any flaw in *how it checks* shows up as a spectacular reduction score
rather than as an error.

Seven earlier versions of this project reported numbers that were wrong.
Each is written up below, because between them they explain most of the
design. If you only read one, read the first: it is the one that would
have made every number in this file meaningless.

### The reduced code was never under test

Both original repos use a `src/` layout, and the benchmark installed the
*cache* checkout — so `import requests` resolved past the copy being
reduced. Deleting the entire `src/requests` package still passed 13/13.
Every "reduction" was of files nothing imported.

The harness now installs the work copy and refuses to run if the package
under test resolves outside it.

### …and that fix was not enough

The same bug came back on a hosted runtime. A notebook worked around a
broken `ensurepip` by adding `--system-site-packages` to the benchmark
venv, and Colab and Kaggle preinstall `requests` and `flask` — so a
second copy sat on `sys.path` *after* the work copy's.

At baseline the work copy was authoritative and the resolve-check passed
honestly. But with a `src/` layout an editable install is a plain path
entry, so the instant the reducer deleted the package the entry pointed
at nothing and the import fell through to the host's copy. It scored
**6/6 fidelity having deleted flask down to its test file in 56
queries**.

The lesson: *asking where imports resolve is the wrong question*,
because it is asked before the state that breaks it exists. The harness
now runs a **positive control** after capturing the baseline — rename
the package aside, the reducer's own most destructive move, and require
the test to stop passing. A tree that still reproduces the bug with its
implementation missing is refused.

Building the regression test for it took three attempts, and the two
failures are instructive: a flat-layout fixture cannot exhibit the bug
at all (setuptools gives it a strict finder that hard-fails rather than
falling through), and a venv created from another venv with
`--system-site-packages` exposes the *base* interpreter's site-packages,
not the parent's — so there was no shadowing copy to find.

### The single-file path never ran the candidate either

The same category error, in a different door, and it survived four
audits because every one of them went through `reduce-project`.

`automre reduce <file> -c "<command>"` wrote the candidate to a
temporary file and then ran the command — which reads the file the
command *names*, i.e. the original. Nothing ever executed the candidate,
so every candidate matched and an entirely empty file was an acceptable
reduction. The multi-file path was unaffected because its validator
writes candidates to the real file, which is why nobody noticed.

The candidate now replaces the target file for the duration of the run
and is restored afterwards, and a command supplied with no target file
is **refused** rather than answered from the original.

Making that path work made a second case reachable: reducing *the test
file itself* against a passing command has the same degenerate solution.
The CLI now refuses that too and points at `reduce-project`. A failing
command on its own file (`python bug.py` reducing `bug.py`) stays
allowed — there the oracle compares a traceback naming the line that
raised, which anchors it.

### The oracle accepted a different bug

`error_type` compares only the exception class, so a reduction that
turned a str/list concatenation `TypeError` into a `NoneType` arithmetic
`TypeError` was accepted. The default is now `output_match`, which
compares full normalized output.

Making that usable required normalizing traceback line numbers — every
removal shifts them, so the same failure at a different offset read as a
different behavior and capped one example at 27% reduction.

### The reducer gutted the test instead of the code

For a *passing* test, "output still matches" has a degenerate solution:
delete the assertions. pytest still prints `1 passed`, so the reducer
was free to stub the test body to `pass`, stub every fixture to `pass`,
and then delete the library nothing depended on any more. That produced
a "99.6% reduction" whose surviving tree was two files of empty stubs.

This is a category error rather than a bug in a heuristic: the test file
is the *oracle*. Files named in the reproduction command — plus any
`conftest.py` above them — are now protected, but only for test-runner
commands. Under `python main.py` the script *is* the subject, and that
case is self-protecting anyway.

**A later gap in the same rule:** protection matched only `.py`
arguments, so `pytest tests/` or a bare `pytest -k foo` protected
nothing at all and put the whole failure back within reach. Directory
arguments now protect the files beneath them, and a test-runner command
that protects nothing is refused outright before any work is done.

### The oracle judged code that was not on disk

CPython decides a cached `.pyc` is fresh by comparing the source's
`(mtime, size)` — and the mtime it stores has one-second resolution. The
reducer rewrites the same file several times per second, so two
candidates for one file that happen to share a byte length (trivially
common: swap one statement for another of equal width) are
indistinguishable to that check. Python then imports the *previous*
candidate's bytecode, and the validator reports on code that no longer
exists.

That is worse than noise. A candidate that genuinely breaks the bug can
be accepted because stale bytecode still holds the working version. It
also made runs unreproducible — the same task, with the hash seed
pinned, took 1449 / 1407 / 1407 queries on three runs. The benchmark's
own fidelity check had the same hole.

Every subprocess that must observe the tree as written now goes through
`validator.oracle_env()`, which sets `PYTHONDONTWRITEBYTECODE`; the
fidelity check additionally purges `__pycache__` before scoring, so the
verifier does not depend on a setting chosen by the code it verifies.

Suppressing writes is not sufficient on its own, which took a second
pass to notice: it says nothing about bytecode that was *already there*.
Reduction now purges `__pycache__` when it begins, so the invariant is
one the reducer establishes rather than one it inherits.

The first failures inflated the reduction figures. This one did not:
reduction stayed within a line or two, and what moved was the query
count, down 4.7%, because false rejections had been forcing wasted
re-subdivision. Its cost was correctness and reproducibility rather than
headline numbers — which is exactly why it survived three rounds of
auditing the headline numbers.

### Removals were cut in the wrong place on non-ASCII files

tree-sitter reports node positions as offsets into a file's UTF-8
*bytes*; everything downstream sliced a Python `str`, which is indexed
by *characters*. The two agree exactly until the first non-ASCII
character and diverge by one index per extra byte thereafter.

`psf/requests` ships `src/requests/certs.py` with a single em dash in
its module docstring, and that one character was enough: the import
statement sliced as `om certifi import where`. Nine of the 118 `.py`
files across requests and flask have non-ASCII in them.

Both halves of that cost something, and neither is visible from outside.
A mis-sliced candidate is garbage, so the syntax check rejected it
without spending a query — and the unit it came from could then *never*
be removed. A span running past the end removes *nothing*, so the
candidate equals its input: it validates, the pass records a removal,
the source does not shrink, and the loop comes round again on identical
input. That spent 100 of the 1,319 queries of one task asking one
question over and over.

Fixing it cut queries 4.5% and *improved* reduction by 44 lines. The
fields are now `start_char`/`end_char`, because the name is what made
the mistake easy to write and invisible to read, and `_candidates` drops
any span that would remove nothing so the next such bug is cheap rather
than expensive. It was found by hashing the whole tree at every oracle
call and counting duplicate states — not by reading the code.

### Decorated code could never be removed

tree-sitter puts a definition's decorators in a `decorated_definition`
wrapper *around* it. Nothing here had a rule for that node, and the
consequences ran in three directions, each of which looked like
something else:

- **At module level, decorated functions were invisible.** The unit
  extractor dropped the wrapper for not being removable and took its
  whole subtree with it. 111 such definitions on flask were never
  offered as candidates at all, and the reducer reported convergence
  with them still in place.
- **Inside a class, they were unremovable.** The inner definition was
  reachable, but removing it cut between the decorators and their
  target, so every attempt was a syntax error — rejected locally at no
  cost, which is exactly why it never showed up as anything. 497
  decorated methods on flask could never be removed by any pass.
- **Phase 4a lost whole files.** The pruner took the inner definition
  and stranded the decorators, and since a prune strips every uncovered
  unit in a file at once, *one* dead decorated function made the entire
  file's pruned source unparseable. Downstream that is indistinguishable
  from "coverage lied": the query fails, the prune rolls back, and every
  unit in the file is rediscovered one at a time. 50 of flask's 82
  files, 12 of requests' 36 and 10 of tomlkit's 28 contain a decorated
  definition.

`decorated_definition` is now the removable unit and the bare definition
inside it is not, so exactly one candidate covers a decorated def while
the statements in its body stay reachable. The pruner judges it by the
inner function's *body* — the decorator line runs at import time, so
judging the whole node would mark every decorated function covered.

**This is the first defect in this project's history that made the
numbers too pessimistic rather than too optimistic.** Every other entry
here inflated a score; this one suppressed one. Candidate counts rose
3.8% on requests, 9.1% on flask and 2.8% on tomlkit, and across the ten
tasks the re-run removed 567 more lines while spending 395 fewer
queries — though it shipped alongside the timeout fix, so see [what the
last two fixes bought](#what-the-last-two-fixes-bought) for what can and
cannot be attributed to it.

The per-repo pattern is what you would predict from the exposure counts
and is worth recording as a weak confirmation: requests, with the fewest
decorated definitions in surviving code, moved 3 lines per task; flask,
with 43% of its functions decorated, moved 55 and 71.

### A hung query cost two minutes

Queries cluster tightly — on `tomlkit-write_backslash` the median is
0.137 s and p99 is 0.404 s. But ten queries out of 2,699 hit the 120 s
ceiling, and **those ten were 76% of that task's entire query time.**

Two plausible explanations for tomlkit's cost were wrong before
instrumenting every query settled it. It is not the target test's own
cost (on the unreduced tree those commands take 0.14–0.49 s, neither the
right ordering nor the right spread), and it is not the per-collection
work in `tests/conftest.py` (that predicts cost tracking test count, and
the ordering inverts).

A mean is simply the wrong statistic for a distribution with that tail,
and every "seconds per query" figure in earlier versions of this file
was one.

`--timeout` is now a **ceiling** rather than the value used; the working
limit is derived from what this project's queries actually cost, over a
sliding window so one slow outlier cannot ratchet it back up. Timeouts
are counted and reported separately.

**A correction to what this file previously said.** An earlier revision
called this "a correctness bug wearing a performance bug's clothes" and
guessed it explained why tomlkit reduced worse. That was wrong on the
first count. A timed-out candidate is one whose tree does not reproduce
the bug in bounded time; refusing it *keeps* the code, which is the
conservative and correct verdict. A tighter limit can cost reduction; it
cannot make the oracle unsound. What was genuinely wrong was the cost,
and the invisibility — a hang was indistinguishable from an ordinary
rejection, which is how it hid.

And it was never a tomlkit phenomenon. Re-measuring found **six timeouts
in each flask task**, worth roughly 700 s of a 1265 s run — 57% of that
task's wall clock, sitting unremarked in every previously published
flask number. Across all ten tasks there were **182 timeouts** that no
earlier run counted. Fixing it cut the benchmark's wall clock from 6.97
hours to 3.34 hours.

**The prediction this file made was half right.** Before the run it said
tomlkit's reduction should be unchanged and its wall clock should fall
~70%. Wall clock fell 62%. Reduction did *not* stay flat — it improved
1.3 points — but that is the decorator fix, which shipped in the same
run, and not a sign the timeout policy was refusing valid reductions.
See [what the last two fixes bought](#what-the-last-two-fixes-bought)
for why the two cannot be separated from this data.

**The policy helps very unevenly, and that is worth stating.** The limit
is `20 ×` the slowest query in a recent window, so it adapts to the
project — but a project with genuinely slow queries gets a
correspondingly slack limit. Per task, the effect ranges from a 69% cut
(`tomlkit-document_dict`, 176.9 → 54.2 min) to 25%
(`tomlkit-array_items`, 115.7 → 86.9 min), and it is the task with 52
timeouts that improved least. Working backwards from its wall clock, its
limit settled near 90 s where `document_dict`'s settled near 28 s,
because `array_items` has multi-second legitimate queries pulling the
window maximum up.

Using a high percentile rather than the maximum would clamp that case
much harder. It is the obvious next lever and it has not been measured,
so it has not been done.

### The profile blamed the reducer, and the profile was what was broken

The others are defects in the tool. This one is a defect in a
*measurement* of it, recorded because that failure mode is the more
common of the two.

Instrumenting only `MultiFileValidator._run` and subtracting from wall
time attributed 64% of the flask run to the reducer's own work — a
striking 275 ms per query against 2.9 ms on requests, superlinear in
tree size, textbook algorithmic defect. It was written up as one, in
this file.

There is no such defect. Timing the whole reduction path instead of one
function puts flask at 99.4% subprocess with a 0.1% residual. The tell
was there in the original table and went unread: flask's mean query came
out *cheaper* than requests', which cannot be true when the query is
`import flask` versus `import requests`.

Two habits come out of it. **Account for the whole of an interval before
believing a subtraction** — an unexplained majority is a broken
measurement until shown otherwise, not a finding. And **a correction is
a claim like any other**: this one overturned a *correct* statement and
cost more than the error it was fixing.

### The same hole was open in the web app

Every entry above is about the benchmark, because that is where the
checks lived. When they were moved into a shared module and the web
worker was pointed at them, the worker turned out to have the very first
failure on this list, still open.

A job naming a test that does not exist ran to **"done"**. pytest exited
4, the oracle adopted *exiting 4* as the behavior to preserve, and the
reducer deleted everything not needed to keep exiting 4. The download
contained `mylib.py` reduced to five blank lines. The summary read
"11 → 9 lines", because blank lines are still lines.

Reduction cannot catch this from the inside — every candidate genuinely
does preserve the behavior it was given; the behavior was worthless. The
check has to happen before the first query, which is what the readiness
gate is for. The benchmark had one and the app did not, and that is the
argument for the two of them sharing a module rather than each growing
its own.

### Three repositories could not tell it what was broken

Every entry above was found on `requests`, `flask` or `tomlkit`. Pointing
the reducer at SWE-bench instances broke it three more ways within the
first two repositories it had never seen, and none of the three is
reachable from the benchmark, because all three of its repos are tidy in
exactly the ways that matter:

| the reducer assumed | broken by | what it looked like |
|---|---|---|
| source files are UTF-8 | one latin-1 file in pylint | crash, **reported as 0% reduction and fidelity 0** |
| paths stay inside the project | any symlink leaving the tree | crash; an outside file sitting in the delete set |

| tests run under pytest | django, sympy | the oracle **entirely unprotected** |

The first is the one worth dwelling on. `pylint` ships
`tests/functional/i/implicit/implicit_str_concat_latin1.py` — one file in
2,189 — and it killed the run in `_line_count`, the very first thing done
to a tree, before a single query was spent. It did not present as an
encoding problem. It presented as a row in a results table reading 0%,
which reads as "this repository is hard to reduce". With the crash fixed,
pylint reduces **97.96%**, better than anything in the benchmark. A bug
that disguises itself as a result is the worst kind this project keeps
finding.

The symlink one needs a caveat, because the exposure is narrower than
the row suggests and overstating it would be the same sin as the rest of
this section. Both the CLI and the benchmark copy a project with
`shutil.copytree`, whose default dereferences symlinks into ordinary
files inside the copy, so neither can reach outside through one. What
was actually broken is a direct `reduce_project` call on a tree that
still has its symlinks: the outside file entered the candidate set, and
nothing outside was ever damaged only because `relative_to` raised
several phases later and killed the run. A crash was standing in for a
boundary. The boundary is now a boundary.

The third is the one that mattered most. `_is_test_runner` knew pytest,
py.test, unittest and nose2; django's suite runs only through
`tests/runtests.py`, so the command read as "python some_script.py" —
the case where the script *is* the subject — and `_protected_files`
returned nothing. django reduced with **0 lines protected**. That
survived only because the target test fails, so its output pins the test
body; the same command with a *passing* test is the flask stub-out
above, arriving through a door nobody had checked.

None of these were findable by reading. The benchmark passes all three
gates on every run, and always will, because its three repositories do
not contain a non-UTF-8 file, a symlink, or a test suite that declines to
be pytest. That is the argument for ingest that the reduction numbers do
not make: the value was never a fourth repository to score against, it
was a repository that could disagree.

### The check that catches these is a vacuity probe, not a size assertion

Size cannot distinguish a good reduction from a destroyed one.
`automre/tests/test_no_vacuous_reduction.py` sabotages the library on
purpose and requires the test to notice. A reduction that survives its
own dependency being broken was never exercising it.

Verified end-to-end on `psf/requests`: reducing for `TestGuessJSONUTF`
keeps `src/requests/utils.py` with its real BOM-detection body and the
test with its parametrized assertions, and injecting a bogus `return`
into `guess_json_utf` makes the reduced test fail.

Re-checked on tomlkit, because execution fidelity alone cannot tell a
real reduction from a gutted test. `tomlkit-write_backslash` was re-run
with the work tree preserved: it reproduced the benchmark's query count
exactly, all four test functions survived with their assertions, and
`_utils.escape_string` — the code the target test actually exercises —
kept its full body and escape tables. 84 of the 115 surviving defs are
stubbed to `pass`, which is the body-stubbing pass doing its job rather
than the reducer emptying the thing it is measured on.

### The halving descent

Not a bug, but the sharpest measurement lesson here.

Unbounded, the batch pass costs one probe per node of a binary tree over
the spans — exactly 2n−1 for n spans, where asking per unit costs n —
and every singleton it probes is a unit the per-unit pass then probes
again. On isolated modules it looked like a 35–40% win. In the pipeline
it *cost* 14–475 queries per task and bought one line, because Phase 4a
has already dropped the zero-coverage definitions and what survives is
mostly live.

**Cost scales with removable density, not candidate count.** Stopping
the descent at groups of 8 makes it a real win: 11,676 queries against
12,171 with no batch pass and 13,066 unbounded, at equal or better
output. Comment stripping keeps the unbounded descent, because it has no
per-unit pass behind it and singleton probes are the only thing that can
isolate one live `# noqa`.

---

## Withdrawn results

Two earlier experiments were benchmarked and reported. **Both
measurements are withdrawn**: they were taken while the benchmark was
reducing code that was never under test, so they describe nothing.

- **Open-source LLM prioritizer** (Qwen 2.5 Coder 0.5B/1.5B, branch
  `ml-prioritizer`). Reported as producing byte-identical output at
  7–12× overhead. That "convergence" was every ordering reaching the
  same place because no removal affected the test. Not re-run; the code
  is retained, the number is not.
- **Learned removability oracle** (`ml/`). Reported as cutting queries
  55%. Re-measured — see below.

Also void: `evaluation/results_ddmin.json`, `results_random.json` and
`results_syntax.json`, which were produced through the single-file
validator described [above](#the-single-file-path-never-ran-the-candidate-either)
and therefore measured nothing.

## The learned oracle, re-measured

Retrained from scratch on **9,423 validated removal attempts** across 8
tasks. The old rows were not merely stale, they were wrong per row: they
were harvested while the reducer was gutting its own oracle, so they
recorded destruction rather than dead code. The signature is in the
class balance, which inverted from 7.0% safely-removable to **63.9%**
once the test file was protected.

Leave-one-repo-out, so no fold scores a repo it trained on:

| held out | rows | AUC |
|---|-----:|----:|
| `pallets/flask` | 5184 | 0.850 |
| `psf/requests`  | 4239 | 0.963 |
| pooled          | 9423 | **0.912** |

These are **two** folds. tomlkit was added after this was measured, so
the third fold does not exist yet — and the gap between the two that do
(0.850 against 0.963) is exactly why a third one matters.

Precision is 95.1% at p>0.9 and **79.5% at p<0.1**. The second number is
the one that matters, because p<0.1 is what the reducer skips: roughly a
fifth of skips are wrong, and a wrong skip leaves code that later passes
must re-examine.

Held out end to end it buys **5.7% fewer queries and gives up 6 lines**.
That is the wrong trade for this project — complete minimization is the
goal — so it is **off by default**, behind `--use-learned-oracle`.

Both arms of that comparison predate several later fixes. The baseline
has since moved by more than the oracle claimed to save, and the fixes
that moved it *improved* reduction where the oracle gives lines up. On
those numbers its margin is likelier to have vanished than merely shrunk.

Worth saying what the model learned: `coverage_ratio` dominates the
permutation importances, with `body_coverage_ratio` second. It largely
rediscovers what Phase 4a already encodes. An honest 6.5% → 5.7%,
shrinking as the reducer around it got better, is the result.

---

## Comparison to Gistify

`evaluation/gistify_runner.py` borrows its evaluation shape from Gistify
(Lee et al., ICLR 2026, arXiv 2510.26790) — clone a real repo, capture
baseline output, reduce, verify execution fidelity — because it needed a
yardstick, not because the tasks are the same.

**It is not a head-to-head.** Their 25-test list isn't public, so this is
a custom subset. More importantly the mechanisms differ: Gistify asks an
LLM to *generate* a single mimicking file, which is a much harder
problem than deleting dead code under empirical validation. This tool
also scores 0/6 on their Self-Containment metric, and no attempt is made
to chase it.

---

## Pointing it at a repository it has never seen

The ten benchmark tasks are hand-written. So are the eighteen the learned
oracle trains on. That is not an accident of effort — it is the cost of
adding a repository: someone has to work out how to install it, which
test to name, and whether the result is worth measuring. It is also why
the oracle has two cross-validation folds and why the fold scores
disagree (0.850 against 0.963), which is the number that most wants a
third repository behind it.

`automre/src/provision/` is that step, in three pieces:

| | |
|---|---|
| `provision()` | a virtualenv per project: coverage, then the project, then its test extra, then pytest **only if** the first three did not already supply one |
| `discover()` | pytest node ids read out of the source, without running pytest |
| `check()` | the readiness gate: the named test exists, the command runs, it does the same thing twice, and removing the package changes the result |

The install order in `provision()` is the whole design. Putting pytest
last is what lets a project keep its own pin without anything parsing a
`pyproject.toml` to find it — and flask is why that matters, since its
conftest reads a sentinel removed in pytest 9.1.

`evaluation/ingest_tasks.py` runs all three over a list of
`(repo, commit, failing test)` records and writes the survivors as a
manifest the benchmark runner already reads. Rejected instances are
written out **with the reason**, since an instance that disappears
quietly is how a benchmark starts measuring nothing again.

### Running it on instances of your own

The input is a JSON or JSONL file of records. Nothing is tied to
SWE-bench — three hand-written dicts work — but its export happens to be
exactly this shape:

```json
[{"instance_id": "psf__requests-1234",
  "repo": "psf/requests",
  "base_commit": "abc123",
  "FAIL_TO_PASS": ["tests/test_a.py::test_x"],
  "test_patch": "--- a/tests/test_a.py\n+++ b/..."}]
```

`test_patch` is optional and usually necessary: 62.6% of SWE-bench
Verified names a test the checkout does not contain until it is applied.
SWE-bench Verified itself needs no library to fetch — it is served as
plain JSON, 100 rows at a time:

```bash
curl -s "https://datasets-server.huggingface.co/rows?dataset=princeton-nlp/SWE-bench_Verified&config=default&split=test&offset=0&length=100" \
  | python3 -c "import json,sys; print(json.dumps([r['row'] for r in json.load(sys.stdin)['rows']]))" \
  > instances.json
```

Then ingest, and reduce whatever survives the gate:

```bash
python3 evaluation/ingest_tasks.py --instances instances.json --limit 15
python3 evaluation/gistify_runner.py \
    --tasks evaluation/tasks_ingested.json --provision-per-task
```

`--provision-per-task` is not optional for an ingested manifest. The
benchmark shares one virtualenv across its three repositories because
they were chosen to coexist; twelve repositories at twelve pinned
commits have conflicting dependency sets, and one environment cannot
hold them.

Run the ingest before budgeting anything. It is cheap — fifteen
instances took about five minutes, most of it cloning — and it is where
you find out how many of them your machine can actually build.

One consequence worth stating: SWE-bench-shaped instances are pinned at
the commit *before* the fix, so their `FAIL_TO_PASS` test **fails**. For
a minimal reproducer that is the ordinary case rather than a problem —
the question is what the smallest project that still crashes this way is.
The gate demanded a passing command until this landed, which would have
rejected every such instance; that assumption now belongs to the
Gistify-style benchmark, which is the one that really does preserve a
passing test.

### What happened when it was pointed at SWE-bench

Fifteen instances of SWE-bench Verified, chosen to span all twelve of its
repositories, none of which autoMRE had been measured on:

| | |
|---|---:|
| accepted by the gate | **7 / 15** |
| rejected, environment cannot be built on Python 3.13 | 8 / 15 |
| ingest wall clock | ~5 min |

Three properties of that dataset, measured across all 500 instances,
decide whether ingest works at all:

| | share | needs |
|---|---:|---|
| test exists only after `test_patch` is applied | **62.6%** | apply the patch at checkout |
| identifier is a pytest node id | 38.8% | works directly |
| identifier is a unittest label (Django) | 41.6% | the repo's own `tests/runtests.py` |
| identifier is a bare function name (sympy) | 19.6% | resolve it to a node id |

The first is the one that matters most, and it is not an edge case: a
SWE-bench instance is pinned before the fix *and before the test exists*,
so nearly two thirds of the dataset names a test the checkout does not
contain. Ingesting one without applying its patch produces a gate
rejection that is perfectly accurate and completely misleading.

**The eight rejections are all the same shape, and none of them are
autoMRE's doing.** These are 2022–2023 commits: `imghdr` was removed in
Python 3.13 (sphinx), `np.unicode_` in NumPy 2.0 (xarray),
`soft_unicode` from markupsafe (requests), `setuptools.dep_util`
(astropy); flask 2.3 calls `pkgutil.get_loader` with
`filterwarnings = error`, and pytest 7.2 calls `ast.Str`. Every one is a
pinned-dependency problem, which is exactly why SWE-bench ships a Docker
image per instance. Reaching the rest of the dataset means per-instance
interpreters, not a change to this code.

#### All seven, reduced end to end

Every instance the gate accepted was reduced with `--provision-per-task`,
on one laptop, one at a time. None of these five repositories had been
measured before; four of them are between 8× and 40× larger than
anything in the benchmark.

| instance | lines | → | raw | reducible-only | fid. | queries | q/line | hours |
|---|---:|---|---:|---:|:-:|---:|---:|---:|
| `mwaskom__seaborn-3187` | 54,019 | 4,156 | 92.31% | 96.31% | 1/1 | 5,228 | 0.097 | 1.0 |
| `pylint-dev__pylint-8898` | 112,008 | 2,288 | 97.96% | 98.29% | 1/1 | 6,400 | 0.057 | 0.4 |
| `django__django-17029` | 456,901 | 2,575 | 99.44% | 99.73% | 1/1 | 23,664 | 0.052 | 1.8 |
| `django__django-17087` | 457,222 | 3,602 | 99.21% | 99.61% | 1/1 | 24,144 | 0.053 | 1.9 |
| `django__django-17084` | 457,257 | 9,455 | 97.93% | 98.58% | 1/1 | 35,229 | 0.077 | 2.6 |
| `sympy__sympy-24562` | 685,418 | 5,931 | 99.13% | 99.48% | 1/1 | 42,656 | 0.062 | 9.7 |
| `sympy__sympy-24661` | 687,383 | 1,908 | 99.72% | 99.80% | 1/1 | 43,186 | 0.063 | 5.5 |
| **aggregate** | **2,910,208** | **29,915** | **98.97%** | **99.37%** | **7/7** | **180,507** | **0.062** | **22.9** |

Reducible-only excludes the protected lines from both sides, and is the
fair comparison with the benchmark's 95.77%: the named test is the
question, so it is never removable. That unseen repositories land *above*
the tuned benchmark is not evidence of quality — it says a 457k-line
Django checkout carries more that one test does not touch than a
17k-line requests checkout does. The number to be pleased about is
7/7 fidelity, which says every one of those trees still reproduces.

The interesting figure is none of the reduction rates. It is **0.052 to
0.097 queries per line, entirely below the 0.11–0.48 range** measured on
the benchmark's 8k–17k-line repositories — at up to forty times the size.

That was the open question this work existed to answer, and it settles
it in the useful direction. The worry was that cost grows with the
repository, making large instances unaffordable: at 0.48 queries per
line a 457k-line Django checkout would have been 219,000 queries and
about a week. It came in at 23,664 and under two hours. Cost does not
scale with the tree, because Phases 1–3 delete most of a large repository
in bulk before Phase 4 ever walks it, and Phase 4 is ~97% of the queries.

Two caveats on that, both visible in the table:

**The estimate was 17% low.** Seven instances were predicted at ~19
core-hours and took **22.9**. Predicting from queries alone under-counts,
because per-query cost is the repository's own test command and is not
constant across repositories.

**sympy is where that shows.** Its two instances ran 42,656 and 43,186
queries — within 1.2% of each other — in **9.7 h and 5.5 h**. Nearly the
same work, nearly double the time. Query count is a portable measure of
the algorithm; wall clock is a measure of somebody else's test suite,
and only one of those two is worth optimising.

The seven manifests the gate produced are committed as
`evaluation/swebench_tasks.json`, and the results above are
`evaluation/results_swebench.json`, so this is re-runnable without
re-ingesting:

```bash
python3 evaluation/gistify_runner.py \
    --tasks evaluation/swebench_tasks.json --provision-per-task \
    --output /tmp/swebench.json
```

Budget a day. It clones five repositories at seven pinned commits and
builds an environment per instance; sympy alone is 15 of the 23 hours.

### Does a smaller repository help a model find the bug?

That is the question the reduction is for, and it is answerable before
any model is loaded. A repair model cannot read a 457,000-line
repository, so something has to choose the few thousand lines it sees.
The standard answer is BM25 over the issue text — SWE-bench's own
published baseline. autoMRE proposes reducing first and retrieving from
what survives.

Three arms, one 16,000-token budget, one prompt, differing only in
which tree the budget is filled from (`evaluation/stage3/`):

| arm | gold file in context | mean recall |
|---|:---:|---:|
| `full_bm25` — BM25 over the original repository | 3 / 7 | 0.429 |
| `reduced_bm25` — BM25 over the reduced tree | **7 / 7** | **0.929** |
| `oracle` — the files the ground-truth patch edits | 5 / 7 | 0.714 |

A model cannot fix what it was not shown, so this bounds every
downstream score, and it costs no GPU to measure.

The oracle arm scoring *below* the proposal is not a paradox, and it is
the most interesting row in the table. **`sympy/core/numbers.py` is
35,182 tokens — larger than the model's entire 32,768-token context
window.** django's `sql/query.py` is 22,944. No retriever at any budget
can put those files in front of the model, so the arm that consists of
exactly the right files cannot be built for those two instances.
Reduced, they are 11,512 and 2,749 tokens, and both fit. On pylint the
whole reduced repository — all 42 files — fits inside the budget, where
BM25 over the original showed 11 files of 2,187 and ranked the buggy one
14th.

Ranks are recorded alongside recall, because recall alone is decided at
the budget boundary and would report something different at 20,000
tokens.

One property of the instances is worth stating before the scores, since
it cuts both ways. Checking every added line of each ground-truth patch
against its issue text: **django-17029's entire fix appears verbatim in
the issue** ("I propose to add: `self.get_swappable_settings_name.
cache_clear()` line to def clear_cache"), and django-17084 and
sympy-24661 leak one line each of six and twenty-one. The other three
leak nothing. For a benchmark of *reasoning* that would be
contamination. For this one it is close to a controlled condition: when
the repair is given, the only remaining variable is whether the model
was shown the file to apply it to, which is precisely the variable under
test.

Both halves of the rig are checked against the ground truth before any
model output is judged by them. Every instance must fail its target test
before the fix and resolve with the ground-truth patch applied as a
*diff*; then the same patch is re-expressed as SEARCH/REPLACE edits — the
format the model is asked for — and pushed through the same parser, the
same anchor rules and the same test command a sample gets. A zero from
the model is therefore the model's zero, and not a broken parser.

That second control is `--gold-as-edits`, and it is a flag rather than a
paragraph on purpose: an unreproducible positive control is the one
claim in a results section that most needs code behind it.

```bash
python3 evaluation/stage3/score_patches.py --gold-as-edits
```

An instance is scored only if **both** controls hold. Failing the first
means the checkout is not the pre-fix commit; failing the second can
leave the fix still applied to the tree, in which case every sample
scored against it resolves and reads as a model that solved the instance
fifteen times.

That pre-flight has already earned itself once. SWE-bench exports
PASS_TO_PASS comma-joined, so pylint's
`test_csv_regex_comma_in_quantifier[foo,bar-expected1]` arrives truncated
at the comma inside its own parameter. pytest cannot collect it, the
whole batch errors, and *the ground-truth patch* read as breaking twelve
previously-passing tests — which would have scored every arm as a
regression. Uncollectable ids are now dropped and named — as is every
`PASS_TO_PASS` entry that resolves to no test at all, which is not a
rare case: SWE-bench prints a unittest method's *docstring* in place of
its name wherever it has one, so 34 of django-17029's 43 entries are
English sentences. Those used to vanish silently, leaving the control
reporting a clean set of 43 while running 9.

The graded test files are also off limits to the model. The prompt has
always said so; nothing enforced it, and the prompt shows the failing
test's source verbatim, so the anchor is in front of it. Three of the
eighty generations on disk aim an edit at the exact `FAIL_TO_PASS` file,
and one deleted assertion in any of them would have been published as a
fix. Such an edit is now refused and counted per arm.

### What the model did with it

80 generations from Qwen2.5-Coder-7B-Instruct at fp16 — five samples per
context — scored by execution, never by resemblance
(`evaluation/stage3/results_stage3.json`). One instance is out on an
environment fault, leaving six.

| arm | instances | parses | edit applies | names a gold file | resolves |
|---|:---:|---:|---:|:---:|:---:|
| `full_bm25` | 6 | 0.67 | 0.17 | 2 / 6 | **0** |
| `reduced_bm25` | 6 | 0.43 | 0.00 | 2 / 6 | **0** |
| `oracle` | 4 | 0.45 | 0.10 | 3 / 4 | **0** |

Nothing resolved, in any arm, on any sample. That is the result, and the
rig is not what produced it: the same six ground-truth patches,
re-expressed as SEARCH/REPLACE edits and pushed through the same parser,
the same anchor rules and the same test command, score **6 of 6**
(`--gold-as-edits`,
`evaluation/stage3/results_gold_as_edits.json`). Parse 1.00, apply 1.00.
A zero here is a 7-billion-parameter model's zero.

**The retrieval advantage did not survive contact with the model.** The
reduced arm puts the gold file in the context 7 times out of 7 against
the full arm's 3, and both then name a gold file on 2 instances out of 6.
Being shown the answer is not the same as choosing it — an obvious thing
to say and a different thing to measure, and it is the honest limit of
what the recall table above can be taken to mean.

**The reduced arm's apply rate is 0.00, exactly as predicted.** Its
SEARCH blocks are copied from a tree the reducer cut lines out of, so an
anchor spanning a deletion cannot exist in the original file. Eleven of
its thirty samples parse and then fail to apply for precisely that
reason. This is not a bug in the scorer; it is the finding, and the next
section is about it.

**Five samples tried to edit the graded test.** Two of them applied —
sympy-24562 twice, once rewriting `assert Rational('0.5', '100') ==
Rational(1, 200)` to expect the buggy `Rational(1, 100100)` instead.
Neither flipped a verdict, because the graded test asserts a loop of
other equalities before that line and the bug breaks those too. So the
hole was real and reachable, and on this data it happened not to pay.
Such edits are now refused and counted rather than left to chance.

**What this cannot claim.** The reducer preserves what *reproduces* a
failure, which is not the same as what is needed to *repair* it.
django-17029's patch adds one line to `Apps.clear_cache`; in the reduced
tree that method is `def clear_cache(self): pass` — sound, because the
test asserts a cache was not cleared and an empty body fails it
identically. A model reading that can name the method and still cannot
write an edit that applies to the original file. Three of the seven
instances have no ground-truth anchor line surviving at all. So the
reduced arm is scored on **localisation**, and resolution is reported
against the full-repository arm; neither stands in for the other.

**And the two file universes are not identical.** Only the files the
`test_patch` names are removed from both trees. Everything else the
reducer deleted for never executing — other tests, fixtures, doc
modules — is still in the full tree and still competes for the budget:
39% of the full arm's context slots hold a test or doc file, against 12%
of the reduced arm's. pylint spends 5 of its 11 slots on
`tests/regrtest_data/`, `doc/data/messages/` and the like, and scores
recall 0. That is reduction doing exactly what it is for, so it is not
subtracted out — but it means the 0.429-vs-0.929 gap is reduction
*including* its removal of dead test and doc files, not reduction of
live source alone.

Two ways of recovering applicability were built and measured before
being deleted. Ranking on the reduced tree while showing original text
scored 0.190; showing the original only where it fits scored 0.333.
Both are *worse than the baseline*, and for the same reason: the ranking
was fine — the gold file lands at rank 1 to 5 — but four full-size
original files fill 16,000 tokens before the fifth is reached. Showing
original text and fitting the budget are in direct conflict, which is
this project's own argument arriving from the other side. Neither arm
was run; a negative result that consistent does not need a GPU.

### Where the idea came from

[SWE-Hub](https://arxiv.org/abs/2603.00575) (2026) is a production system
for manufacturing executable software-engineering tasks: an Env Agent
that turns a repository snapshot into a reproducible environment, a Test
Agent that finds the entrypoint, and a verification gate before anything
is called a task. Its code and data are not released, so nothing here
uses it — what is taken is the observation that those three steps belong
together, which is precisely what was wrong in this repository, where two
of them already existed in files that could not see each other.

The relationship is worth stating plainly, because it runs the other way
from the borrowing. SWE-Hub *generates* executable tasks and argues that
agent progress is bottlenecked on them being brittle and expensive.
autoMRE takes one task and produces the smallest environment that still
exhibits it. Same problem, opposite end.

## Structure

```
automre/src/
  parser.py                 tree-sitter AST + CodeUnit model
  tracer.py                 coverage.py wrapper (interpreter-aware)
  validator.py              behavior oracle + output normalization
  reducer.py                delta debugging, stubbing, comment/blank passes
  multi_file/
    dependency_analyzer.py    Phase 1
    multi_file_validator.py   whole-project oracle + query timing
    import_inliner.py         Phase 3
    coverage_pruner.py        Phase 4a
    multi_file_debugger.py    orchestrator, oracle-file protection, progress
  provision/
    environment.py            a virtualenv per project
    discovery.py              pytest node ids, read not run
    gate.py                   is this worth reducing at all
  ml/                       learned removability oracle (optional extra)
automre/tests/              soundness, vacuity, and unit tests
automre/examples/           small projects with real bugs, used by
                            entrypoint.sh and by the tests
automre.py                  convenience shim; see the note above on why
                            `python3 -m automre.src.cli` fails beside it
entrypoint.sh               tests plus one example reduction, ~30 s
evaluation/
  gistify_runner.py         benchmark harness
  gistify_tasks.json        10-task manifest (requests, flask, tomlkit)
  ingest_tasks.py           (repo, commit, failing test) -> a manifest,
                            everything that fails the gate written out
                            with its reason
  cloud_bench.py            same harness for Colab/Kaggle, plus the
                            --ablation coverage-prune A/B
  swebench_tasks.json       the seven SWE-bench Verified manifests the
                            gate accepted, plus results_swebench.json
  stage3/                   does a smaller repository help a model find
                            the bug
    build_contexts.py       BM25, budgeting, and the three arms
    instances.json          the seven full SWE-bench records
    contexts.jsonl          one row per (instance, arm): the prompt and
                            what retrieval put in it
    controls.json           which instances are scoreable, and why the
                            one that is not was excluded
    score_patches.py        parse edits, apply, run F2P then P2P
    kaggle_cells.md         the generation notebook, as reviewable text
    push_kernel.py          turns that into a notebook and pushes it
  results_gistify_*.json    results per configuration
notebooks/
  automre_benchmark.ipynb   driver for cloud_bench.py on a hosted runtime
web/
  frontend/                 Next.js UI (Vercel)
  worker/                   FastAPI + autoMRE in a container
```

### Benchmark notes

The benchmark pins `pytest==9.0.0` in `.gistify_venv`. flask's conftest
reads `_pytest.monkeypatch.notset`, a private sentinel removed in 9.1,
so under a newer pytest every flask task dies during collection — and
the harness would then faithfully preserve *the collection error*.

Wall clock is not comparable across sessions; see the `_provenance` note
in the results files.

---

## References

- Zeller & Hildebrandt (2002) — `ddmin`, the algorithm every delta
  debugger descends from.
- Misherghi & Su (2006) — Hierarchical Delta Debugging: structure-aware
  reduction.
- Lee et al. (ICLR 2026) — Gistify, source of the evaluation shape.

## License

MIT.
