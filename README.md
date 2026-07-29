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
| requests-super_len_partial   | 11209 → 1181 | 89.5% | 36 → 10 | 1464 |
| requests-guess_json_utf      | 11209 → 1200 | 89.3% | 36 → 10 | 1504 |
| requests-content_disposition | 11209 → 1193 | 89.4% | 36 → 10 | 1456 |
| requests-cookie_utils        | 11209 → 1204 | 89.3% | 36 → 10 | 1509 |
| flask-request_ctx_basic      | 17565 → 2485 | 85.9% | 82 → 13 | 2866 |
| flask-blueprint_registration | 17565 → 1968 | 88.8% | 82 → 17 | 3372 |

**88.5% aggregate line reduction, 6/6 execution fidelity.** Raw data in
`evaluation/results_gistify_v5_sound.json`.

Reduction is the expensive part: ~2000 validation queries per task at
roughly 170 ms each. Around 99% of wall time is the subprocess running
the target test; the reducer's own work is about 0.5 ms per query. The
only lever that matters is asking fewer questions.

## Why these numbers are trustworthy

A minimizer is easy to fool, including by accident. Three earlier
versions of this benchmark reported better numbers that were wrong, and
each failure is worth stating because it shapes how the tool is now
built.

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

The last one is a category error rather than a bug in a heuristic: the
test file is the *oracle*, the statement of what must remain true, not
code to be reduced. Files named in the reproduction command — plus any
`conftest.py` above them — are now protected, but only for test-runner
commands. Under `python main.py` the script *is* the subject, and that
case is self-protecting anyway, since the oracle compares the traceback
including the line that raised.

**The check that catches all three is a vacuity probe, not a size
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
- 4b: delta debugging over AST units. Each pass walks every candidate
  once and accumulates the removals that stick, repeating until a pass
  changes nothing. Unparseable candidates are rejected locally rather
  than spending a subprocess on an edit no oracle could accept.
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
- **Learned removability oracle** (branch history on `master`, `ml/`).
  Reported as cutting queries 55% with identical output, held out. Its
  10,074 training rows were harvested from the broken pipeline and
  describe a reducer that no longer exists.

Both need re-running before any claim is made. The code is retained; the
numbers are not.

## Install and run

```bash
pip install -e .

# Reduce a project against a failing test
python -m autorepro_min.src.cli reduce-project ./my_project \
  -c "python3 -m pytest tests/test_bug.py::test_foo -x -q" \
  -v

# Tests (~8s)
python -m pytest autorepro_min/tests/

# Benchmark. Provisions a pinned venv on first run; ~65 min for 6 tasks
python evaluation/gistify_runner.py
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
