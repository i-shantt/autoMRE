# autoMRE

**Give it a Python project and one test. It hands back the smallest
version of that project where the test still behaves exactly the same
way.**

Everything the test does not need is deleted — whole files, then
definitions, then individual statements. After every single deletion the
test is run again, and the deletion is kept only if the output is
unchanged.

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

The output is an **MRE**: a minimal reproducible example.

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
pip install -e .
./entrypoint.sh
```

Runs the test suite, then reduces a bundled four-file example down to
the six lines that still raise the same `TypeError`. It does not
reproduce the benchmark below — that clones three repositories and takes
hours — it prints the command for it.

### In a browser

`web/` contains a small web app: drop in a zipped project, name the
test, watch it shrink, download the result. The UI deploys to Vercel;
the engine has to run somewhere that can hold a job for minutes to
hours, so it ships as a container. Setup is in
[`web/README.md`](web/README.md).

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
# Tests (70, ~21 s)
python3 -m pytest automre/tests/

# Full benchmark. Provisions a pinned venv on first run.
python3 evaluation/gistify_runner.py

# Or one repo's worth
python3 evaluation/gistify_runner.py --only requests
```

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
  cloud_bench.py            same harness for Colab/Kaggle, plus the
                            --ablation coverage-prune A/B
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
