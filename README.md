# AutoRepro-Min

Automated multi-file bug reproduction minimization for Python projects,
via coverage-guided empirical delta debugging.

Given a project directory and a reproduction command, AutoRepro-Min
iteratively removes files, definitions and statements, validating after
each candidate removal that the command still produces the same
behavior. What remains is a small subset of the original codebase that
still exhibits the bug.

## Results

Six pytest tasks across `psf/requests v2.32.3` and `pallets/flask 3.0.3`
(`evaluation/gistify_tasks.json`), on an M4 MacBook Air:

| task | lines | reduction | files | queries |
|------|------:|----------:|:-----:|--------:|
| requests-super_len_partial   | 11209 → 1169 | 89.6% | 36 → 10 | 1186 |
| requests-guess_json_utf      | 11209 → 1188 | 89.4% | 36 → 10 | 1223 |
| requests-content_disposition | 11209 → 1182 | 89.5% | 36 → 10 | 1193 |
| requests-cookie_utils        | 11209 → 1193 | 89.4% | 36 → 10 | 1224 |
| flask-request_ctx_basic      | 17565 → 2482 | 85.9% | 82 → 13 | 2613 |
| flask-blueprint_registration | 17565 → 1967 | 88.8% | 82 → 17 | 3181 |

**88.5% aggregate line reduction, 6/6 execution fidelity.** Raw data in
`evaluation/results_gistify_charoffsets.json`.

Runs are reproducible: repeated runs of a task are bit-identical, in
output and in query count, with no pinned hash seed. That is a recent
property rather than an assumed one — see the fourth entry below.

That headline number undersells the tool, because 79% of what survives
is the test file, which is deliberately never touched — it is the
statement of what must remain true. Measured over the code actually
eligible for removal, reduction is **97.5%**: `psf/requests` goes from
10,193 reducible lines to 153–177, and `flask-request_ctx_basic` from
15,515 to 432.

Reduction is the expensive part: ~1770 validation queries per task,
roughly an hour for all six. Timing every function on the reduction
path — the validator, the reducer, the parser, the coverage tracer —
accounts for where that time goes to within half a percent:

| | requests-guess_json_utf | flask-request_ctx_basic |
|---|---:|---:|
| reduction wall | 262.2 s | 1288.5 s |
| in the oracle subprocess | 258.0 s (**98.4%**) | 1281.4 s (**99.4%**) |
| in the coverage tracer | — | 5.3 s (0.4%) |
| mean per subprocess | 195.6 ms | 490.6 ms |
| everything else | 3.19 ms/query | 0.69 ms/query |

The reducer's own work is a rounding error on both repos. What differs
is the price of a single question: 490 ms on flask against 196 ms on
requests, because every query pays a fresh interpreter start and a
fresh `import flask`. Asking fewer questions, or making one cheaper, is
therefore the entire lever — there is no third option and no hot spot
in the reducer to go find.

Raw per-function timings: `evaluation/profile_reduction_paths.json`.

Breaking down one flask query on the unreduced tree, median of seven:
11 ms to start the interpreter, 62 ms once `pytest` is imported, 103 ms
once `flask` is too, 202 ms after collection — and 212 ms to actually
run the test. **The test body is about 5% of the query.** The other 95%
is identical work repeated 2,612 times. That makes a persistent worker
the obvious idea and also a dangerous one: reusing a process means
reusing `sys.modules`, which is the stale-bytecode failure again in a
form that `PYTHONDONTWRITEBYTECODE` cannot fix. Not attempted.

Two caveats on that table. It is one run per repo, so the millisecond
figures carry ordinary machine-load noise; the shares do not, since
they are internal ratios. And an earlier revision of this section
reported the flask column as 35.6% subprocess and blamed a superlinear
defect in the reducer — that measurement left 64% of the run
unexplained, which was reason to distrust it rather than to publish it.
See the sixth entry below.

## Why these numbers are trustworthy

A minimizer is easy to fool, including by accident. Five earlier
versions of this benchmark reported reduction numbers that were wrong,
and a sixth entry covers a performance measurement that was wrong. Each
is worth stating because it shapes how the tool is now built and how it
is now measured.

**The reduced code was never under test.** Both target repos use a
`src/` layout, and the harness installed the *cache* checkout, so
`import requests` resolved past the copy being reduced. Deleting the
entire `src/requests` package still passed 13/13. Every "reduction" was
of files nothing imported. The harness now installs the work copy and
refuses to run if the package under test resolves outside it.

**The oracle accepted a different bug.** `error_type` compares only the
exception class, so a reduction that turned a str/list concatenation
`TypeError` into a `NoneType` arithmetic `TypeError` was accepted. The
default is now `output_match`, which compares full normalized output.
Making that usable required normalizing traceback line numbers — every
removal shifts them, so the same failure at a different offset read as
different behavior and capped one example at 27% reduction.

**The reducer gutted the test instead of the code.** For a *passing*
test, "output still matches" has a degenerate solution: delete the
assertions. pytest still prints `1 passed`, so the reducer was free to
stub the test body to `pass`, stub every fixture to `pass`, and then
delete the library nothing depended on any more. That produced a
"99.6% reduction" whose surviving tree was two files of empty stubs.

That third one is a category error rather than a bug in a heuristic:
the test file is the *oracle*, the statement of what must remain true,
not code to be reduced. Files named in the reproduction command — plus
any `conftest.py` above them — are now protected, but only for
test-runner commands. Under `python main.py` the script *is* the
subject, and that case is self-protecting anyway, since the oracle
compares the traceback including the line that raised.

**The oracle judged code that was not on disk.** CPython decides a
cached `.pyc` is fresh by comparing the source's `(mtime, size)`, and
the mtime it stores has one-second resolution. The reducer rewrites the
same file several times per second, so two candidates for one file that
happen to share a byte length — trivially common, swap one statement
for another of equal width — are indistinguishable to that check.
Python then imports the *previous* candidate's bytecode, and the
validator reports on code that no longer exists.

That is worse than noise. A candidate that genuinely breaks the bug can
be accepted because stale bytecode still holds the working version. It
also made runs unreproducible, since whether two writes land in the same
clock tick is pure timing — the same task, with the hash seed pinned,
took 1449 / 1407 / 1407 queries on three runs. The benchmark's own
fidelity check had the same hole: it scored the tree in the directory
the reducer had just rewritten, so a tree that did not actually
reproduce could still score 1.

Every subprocess that must observe the tree as written now goes through
`validator.oracle_env()`, which sets `PYTHONDONTWRITEBYTECODE`; the
fidelity check additionally purges `__pycache__` before scoring, so the
verifier does not depend on a setting chosen by the code it verifies.
`autorepro_min/tests/test_oracle_sees_current_source.py` rewrites a
module to a same-length body and fails if the change goes unobserved.

Suppressing writes is not sufficient on its own, which took a second
pass to notice: it says nothing about bytecode that was already there
when reduction started, and the benchmark leaves some, because it runs
the reproduction command once to check the baseline is healthy before
handing the tree over. Reduction now purges `__pycache__` when it
begins, so the invariant is one the reducer establishes rather than one
it inherits.

The first three failures inflated the reduction figures. This one did
not: reduction is within a line or two of what it was, and what actually
moved was the query count, down 4.7%, because false rejections had been
forcing wasted re-subdivision. Its cost was correctness and
reproducibility rather than headline numbers, which is exactly why it
survived three rounds of auditing the headline numbers.

**Removals were cut in the wrong place on any file containing a
non-ASCII character.** tree-sitter reports node positions as offsets
into a file's UTF-8 *bytes*; everything downstream sliced a Python
`str`, which is indexed by *characters*. The two agree exactly until the
first character that is not ASCII, and diverge by one index per extra
byte from there on.

`psf/requests` ships `src/requests/certs.py` with a single em dash in
its module docstring, and that one character was enough. The import
statement sliced as `om certifi import where`. The last statement in the
file began at character 414 of a 427-character string and ended at 428.
Nine of the 118 `.py` files across the two repos have non-ASCII in them.

Both halves of that cost something, and neither is visible from the
outside. A mis-sliced candidate is garbage, so the syntax check rejects
it without spending a query — and the unit it came from can then never
be removed, which made reduction quietly worse on exactly those nine
files. A span running past the end removes *nothing*, so the candidate
equals its input: it validates, because nothing changed and the bug
still reproduces; the pass records a removal; the source does not
shrink; and the loop comes round again on identical input until
`max_iterations` stops it. That spent 100 of the 1,319 queries of
`requests-guess_json_utf` asking one question over and over.

Fixing it cut queries 4.5% and *improved* reduction by 44 lines, which
is the tell that it was never only a performance bug. The fields are now
`start_char`/`end_char`, because the name is what made the mistake easy
to write and invisible to read, and `_candidates` drops any span that
would remove nothing so the next such bug is cheap rather than
expensive. It was found by hashing the whole tree at every oracle call
and counting duplicate states — not by reading the code.

**The profile blamed the reducer, and the profile was the thing that was
broken.** The five above are all defects in the tool. This one is a
defect in a measurement of it, recorded here because the failure mode is
the more common of the two. Instrumenting only `MultiFileValidator._run`
and subtracting from wall time attributed 64% of the flask run to the
reducer's own work — a striking 275 ms per query against 2.9 ms on
requests, superlinear in tree size, textbook algorithmic defect. It was
written up as one, in this file.

There is no such defect. Timing the whole reduction path instead of one
function puts flask at 99.4% subprocess with a 0.1% residual. The
earlier figure was a subprocess total that covered part of a run divided
by a wall time and a query count that covered all of it. The tell was
there in the original table and went unread: flask's mean query came out
*cheaper* than requests' (154 ms against 160 ms), which cannot be true
when the query is `import flask` versus `import requests`. It is
actually 490 ms against 196 ms.

Two habits come out of this. Account for the whole of an interval before
believing a subtraction — an unexplained majority is a broken
measurement until shown otherwise, not a finding. And a correction is a
claim like any other: this one overturned a *correct* statement about
where the time went, and cost more than the error it was fixing.

**The check that catches the first three is a vacuity probe, not a size
assertion.** Size cannot distinguish a good reduction from a destroyed
one. `autorepro_min/tests/test_no_vacuous_reduction.py` sabotages the
library on purpose and requires the test to notice; a reduction that
survives its own dependency being broken was never exercising it.

Verified end-to-end on `psf/requests`: reducing for `TestGuessJSONUTF`
keeps `src/requests/utils.py` with its real BOM-detection body and the
test with its parametrized assertions, and injecting a bogus `return`
into `guess_json_utf` makes the reduced test fail.

## Pipeline

**Phase 1 — Analysis.** Run the reproduction command once under
`coverage.py` to record which files and lines executed, parse `import`
statements into a project-local dependency graph, and classify every
file as `EXECUTED`, `IMPORTED_ONLY` or `UNREACHABLE`. The trace must run
under the same interpreter as the reproduction command; tracing flask
under a different pytest version collected 2 files instead of 23 and
sent the whole run to zero.

**Phase 2 — File deletion.** Bulk-delete every `UNREACHABLE` file and
validate, falling back to per-file deletion if the batch breaks the bug.
Then probe each `IMPORTED_ONLY` file individually.

**Phase 3 — Selective inlining.** Walk the surviving graph leaves-first,
inlining used definitions into importers. Abandoned after three
consecutive rollbacks: inlining a module of an installed package cannot
work — folding `src/flask/app.py` into `__init__.py` breaks every
`from flask.app import X` — and re-learning that twenty times costs
twenty queries.

**Phase 4 — Intra-file reduction.**
- 4a: coverage-based bulk prune — drop definitions with zero executed
  lines in one query per file, rolling back if the bug breaks.
- 4b: delta debugging over AST units, in two stages. First a batch pass
  asks whether all top-level units can go at once and halves on refusal,
  ddmin-style, to strip bulk cheaply. Then a per-unit pass walks every
  remaining candidate once and accumulates the removals that stick,
  repeating until a pass changes nothing. Unparseable candidates are
  rejected locally rather than spending a subprocess on an edit no
  oracle could accept.

  The halving descent is bounded, and the reason is the sharpest
  measurement lesson in this project. Unbounded, a set with nothing
  removable costs one probe per node of a binary tree over it — exactly
  2n−1 for n spans, where asking per unit costs n — and every singleton
  it probes is a unit the per-unit pass then probes again. On isolated
  modules the pass looked like a 35–40% win; in the pipeline it *cost*
  14–475 queries per task and bought one line, because Phase 4a has
  already dropped the zero-coverage definitions and what survives is
  mostly live. Cost scales with removable *density*, not candidate
  count. Stopping the descent at groups of 8 makes it a real win:
  11,676 queries against 12,171 with no batch pass and 13,066
  unbounded, at equal or better output. Those three were measured
  together on the pre-`PYTHONDONTWRITEBYTECODE` pipeline, so they
  compare fairly with each other but none is the current baseline — the
  bounded configuration now costs 10,620. Comment stripping keeps the
  unbounded descent, because it has no per-unit pass behind it and
  singleton probes are the only thing that can isolate one live
  `# noqa`.
- Then: collapse function bodies to `pass` where the body turns out not
  to matter (deletion can never empty a body — Python needs one
  statement — so a docstring would otherwise be pinned in place), strip
  comments by ddmin-style halving, and collapse the blank lines that
  byte-range removal leaves behind.

**Phase 5 — Final file sweep.** Retry deleting every surviving file,
repeating until a round finds nothing. Phase 2 asks this while
everything still imports everything; Phase 4 is what removes the last
import. Without a second look those modules stay as orphans — on
requests that was ten files.

## Withdrawn results

Two earlier experiments were benchmarked and reported. **Both
measurements are withdrawn**: they were taken while the benchmark was
reducing code that was never under test, so they describe nothing.

- **Open-source LLM prioritizer** (Qwen 2.5 Coder 0.5B/1.5B, branch
  `ml-prioritizer`). Reported as producing byte-identical output at
  7–12× overhead. That "convergence" was every ordering reaching the
  same place because no removal affected the test.
- **Learned removability oracle** (`ml/`). Reported as cutting queries
  55% with identical output, held out. Its 10,074 training rows were
  harvested while nothing being removed was under test, so almost every
  skip looked free. **Re-measured — see below.**

The LLM prioritizer has not been re-run. The code is retained; the
number is not.

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

Precision is 95.1% at p>0.9 and **79.5% at p<0.1**. The second number is
the one that matters, because p<0.1 is what the reducer skips: roughly a
fifth of skips are wrong, and a wrong skip leaves code that later passes
must re-examine.

Held out end to end, against the identical reducer, it buys **5.7% fewer
queries and gives up 6 lines** (11,676 → 11,006 queries; 9,227 → 9,233
lines). That is the wrong trade for this project — complete minimization
is the goal — so it is **off by default**, behind
`--use-learned-oracle`.

Both arms of that comparison predate two later fixes. The baseline has
since moved to 10,620 queries on its own — more than the 670 the oracle
claimed to save — and the offset fix that got it there also *improved*
reduction, where the oracle gives lines up. The oracle arm has not been
re-run, so its margin against the current pipeline is unmeasured; on
those numbers it is likelier to have vanished than merely shrunk.

It is also worth saying what the model learned: `coverage_ratio`
dominates the permutation importances, with `body_coverage_ratio`
second. It largely rediscovers what Phase 4a already encodes, and the
AST-shape features are what it adds on top. An honest 6.5% → 5.7%,
shrinking as the reducer around it got better, is the result.

## Install and run

```bash
pip install -e .

# Reduce a project against a failing test
python -m autorepro_min.src.cli reduce-project ./my_project \
  -c "python3 -m pytest tests/test_bug.py::test_foo -x -q" \
  -v

# Tests (32, ~7s)
python -m pytest autorepro_min/tests/

# Benchmark. Provisions a pinned venv on first run; ~1 h for 6 tasks
python evaluation/gistify_runner.py

# Same benchmark with the learned oracle (off by default)
python evaluation/gistify_runner.py --use-learned-oracle
```

The benchmark pins `pytest==9.0.0` in `.gistify_venv`. flask's conftest
reads `_pytest.monkeypatch.notset`, a private sentinel removed in 9.1,
so under a newer pytest every flask task dies during collection — and
the harness would then faithfully preserve *the collection error*.

## Structure

```
autorepro_min/src/
  parser.py                 tree-sitter AST + CodeUnit model
  tracer.py                 coverage.py wrapper (interpreter-aware)
  validator.py              behavior oracle + output normalization
  reducer.py                delta debugging, stubbing, comment/blank passes
  multi_file/
    dependency_analyzer.py    Phase 1
    multi_file_validator.py   whole-project oracle
    import_inliner.py         Phase 3
    coverage_pruner.py        Phase 4a
    multi_file_debugger.py    orchestrator + oracle-file protection
  ml/                       learned removability oracle (optional extra)
  tests/                    soundness, vacuity, and unit tests
evaluation/
  gistify_runner.py         benchmark harness
  gistify_tasks.json        6-task manifest
  results_gistify_*.json    results per configuration
```

## Comparison to Gistify

`evaluation/gistify_runner.py` borrows its evaluation shape from Gistify
(Lee et al., ICLR 2026, arXiv 2510.26790) — clone a real repo, capture
baseline output, reduce, verify execution fidelity — because it needed a
yardstick, not because the tasks are the same.

It is not a head-to-head. Their 25-test list isn't public, so this is a
custom 6-test subset. More importantly the mechanisms differ: Gistify
asks an LLM to *generate* a single mimicking file, which is a much
harder problem than deleting dead code under empirical validation. This
tool also scores 0/6 on their Self-Containment metric, and no attempt is
made to chase it.

## References

- Zeller & Hildebrandt (2002) — `ddmin`, the algorithm every delta
  debugger descends from.
- Misherghi & Su (2006) — Hierarchical Delta Debugging: structure-aware
  reduction.
- Lee et al. (ICLR 2026) — Gistify, source of the evaluation shape.

## License

MIT.
