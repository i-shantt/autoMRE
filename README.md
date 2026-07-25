# AutoRepro-Min

Automated multi-file bug reproduction minimization for Python projects,
via coverage-guided empirical delta debugging.

Given a project directory and a reproduction command (typically a
failing pytest test), AutoRepro-Min iteratively removes files,
imports, and definitions, validating after each candidate removal
that the reproduction command still produces the same output. What
remains is the smallest subset of the original codebase that still
exhibits the bug.

## Pipeline

The reducer runs four phases. Each one either eliminates candidates
in bulk (fast) or one-at-a-time under validation (safe fallback).

**Phase 1 — Analysis.** Run the reproduction command once with
`coverage.py` to record which files and lines executed. Parse each
file's `import` statements to build a project-local dependency
graph. Classify every file as `EXECUTED`, `IMPORTED_ONLY`, or
`UNREACHABLE`.

**Phase 2 — File deletion.**
- 2a: bulk-delete every `UNREACHABLE` file and validate. If the
  batch broke the bug (dynamic import etc.), restore and fall back
  to per-file validated deletion.
- 2b: probe every `IMPORTED_ONLY` file individually; keep the
  deletion if the bug still fires.

**Phase 3 — Selective inlining.** Walk the surviving dependency
graph leaves-first. For each file, try to inline its used top-level
definitions into every importer. With `--aggressive-inline`, retry
even for files that have top-level side effects and let empirical
validation catch the failures.

**Phase 4 — Intra-file reduction.**
- 4a: coverage-based bulk prune (see below).
- 4b: hierarchical delta debugging (HDD-E) on what remains.

## Coverage-based pruning (Phase 4a)

Phase 1's per-line coverage tells us exactly which lines executed
in the original run. After Phases 2 and 3, we re-run coverage
against the current tree (inlining shifts line numbers and copies
code into new files), then in each surviving file:

- If a top-level definition has zero executed lines, drop it.
- If a class is partially covered, descend into its methods and
  drop the ones whose bodies have zero coverage.
- Never touch statements inside a live function — control-flow
  surgery is HDD-E's job.

One validation call per file replaces up to N per-unit queries
that HDD-E would otherwise burn discovering the same dead code.
When the removal breaks the bug (decorator side effects, dynamic
dispatch, metaclass wiring — things static coverage can miss), the
file is rolled back and HDD-E processes it normally.

The pass is on by default. Disable per-run with
`--no-coverage-prune` on `reduce-project` or the Gistify runner.

## Gistify-style benchmark

`evaluation/gistify_runner.py` runs the same evaluation shape as
the Gistify paper (Lee et al., ICLR 2026, arXiv 2510.26790):
clone a real repo, capture baseline output, reduce, verify
execution fidelity. The current manifest is 6 pytest tests across
`psf/requests v2.32.3` and `pallets/flask 3.0.3`.

| Configuration              | Fidelity | Single-file | Avg queries | Avg time/task |
|----------------------------|:--------:|:-----------:|:-----------:|:-------------:|
| heuristic (Phase 4a off)   | 6/6      | 0/6         | 241         | 51.3s         |
| heuristic + coverage prune | 6/6      | 0/6         | 244         | 52.0s         |
| Gistify paper best         | 58.7%    | (their SC)  | —           | —             |

**Read these numbers carefully.** The Gistify paper number is a
directional comparison, not a formal head-to-head:

- Their test list (25 tests) isn't publicly released, so ours is
  a custom 6-test subset over the same repos. Ours are pure-Python,
  hand-picked, and probably easier on average.
- Their approach *generates* a mimicking single file with an LLM;
  ours *deletes dead code* under empirical validation. Different
  mechanism, same execution-fidelity metric.
- We score 0/6 on their Self-Containment (single-file output)
  metric because we can only collapse to one file when the reduced
  project happens to fit in one.

The head-to-head is honest only on Execution Fidelity itself — and
even there, the sample sizes and test selection differ.

## What we tried and dropped

Two earlier iterations of the "prioritization" idea were built,
benchmarked, and removed. Both stories are here because a portfolio
project should show the discipline of rejecting things that didn't
work, not just the ones that did.

**Open-source LLM prioritizer (Qwen 2.5 Coder, 0.5B / 1.5B).**
Idea: use a code LLM to rank HDD-E's removal candidates so the
debugger tries the highest-probability-removable ones first. Result:
byte-identical output to the pure heuristic on all 6 tasks, at 7–12×
wall-clock overhead. The empirical validator converges on the same
fixed point regardless of removal order; the LLM saved ~30% of
validation queries but each query got expensive enough to more than
wipe out the savings. Code removed on this branch. Prior branch:
`ml-prioritizer`.

**Coverage-based Phase 4a pruning (this branch).** Idea above.
Result: correctness preserved (still 6/6 fidelity, byte-identical
output), but slightly *worse* wall-clock (+~1s/task) and slightly
more queries (+~4/task). Same convergence story: HDD-E was already
going to reach the same minimum, and the pruner's failed attempts
(fixture / decorator machinery that coverage can't see) each cost
one wasted validation. Retained as an off-switch (`--no-coverage-
prune`) rather than deleted because it's safe and could be a win
on non-test-file codebases where import-time machinery is less
prevalent.

## Install and run

```bash
pip install -e .

# Reduce a project against a specific failing test
python -m autorepro_min.src.cli reduce-project ./my_project \
  -c "python3 -m pytest tests/test_bug.py::test_foo -x -q" \
  --strategy output_match \
  --aggressive-inline \
  -v

# Rerun the Gistify benchmark (~5 min on M4)
python evaluation/gistify_runner.py

# Ablation: with vs without Phase 4a
python evaluation/gistify_runner.py
python evaluation/gistify_runner.py --no-coverage-prune
```

## Structure

```
autorepro_min/src/
  parser.py               tree-sitter AST + CodeUnit model
  tracer.py               coverage.py wrapper
  validator.py            behavior oracle (exact / output_match / error_type / error_message)
  reducer.py              single-file HDD-E
  multi_file/
    dependency_analyzer.py  Phase 1: coverage + import graph + classification
    multi_file_validator.py whole-project oracle
    import_inliner.py       Phase 3: inline defs into importers
    coverage_pruner.py      Phase 4a: coverage-based bulk prune
    multi_file_debugger.py  Phase 1-4 orchestrator
  cli.py                  reduce / reduce-project / validate / trace commands
evaluation/
  gistify_runner.py       Gistify-style benchmark harness
  gistify_tasks.json      6-task manifest
  results_gistify_*.json  benchmark results per configuration
```

## References

- Zeller & Hildebrandt (2002) — `ddmin`, the algorithm every delta
  debugger descends from.
- Misherghi & Su (2006) — Hierarchical Delta Debugging: structure-
  aware reduction (what HDD-E extends).
- Lee et al. (ICLR 2026) — Gistify, the benchmark we compare against.

## License

MIT.
