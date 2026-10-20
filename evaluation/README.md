# `evaluation/`

The benchmark harness, and a provenance record for every `results_*.json`
in this directory. There are 26 of them and only one is current, so the
point of this file is to stop a reader quoting a number that was
withdrawn.

Validity is adjudicated in the top-level [README](../README.md#why-these-numbers-are-trustworthy),
which documents each measurement bug and the date it was fixed. This file
only says which file is which.

## The harness

| script | what it does |
|---|---|
| `gistify_runner.py` | The benchmark behind the headline numbers. `--only <repo>` runs one repo's tasks, `--resume` continues an interrupted run. |
| `cloud_bench.py` | The same benchmark packaged for a hosted GPU/CPU runner, with the local reference printed alongside for comparison. |
| `ingest_tasks.py` | Turns SWE-bench instances into benchmark tasks, recording which were rejected and why. |
| `generate_oracle_data.py` | Harvests labelled removal attempts to train the optional learned oracle. |
| `stage3/` | The SWE-bench patch experiment: `build_contexts.py` builds the prompts, `score_patches.py` applies and grades the replies, `push_kernel.py` ships it to a hosted runner. |

## The results files

Chronological. **Everything before 2026-07-30 is withdrawn** — those runs
were measured through a reducer that was gutting its own test file, and
through an oracle that judged bytecode rather than the code on disk. The
top-level README walks through both.

| file | date | tasks | queries | aggregate | status |
|---|---|---:|---:|---:|---|
| `results_hdd-e.json` | 07-23 | 2 | – | – | **void** — single-file validator [never ran the candidate](../README.md#the-single-file-path-never-ran-the-candidate-either) |
| `results_ddmin.json` | 07-23 | 2 | – | – | **void**, same cause. The ddmin baseline. |
| `results_syntax.json` | 07-23 | 2 | – | – | **void**, same cause. Syntax-guided baseline. |
| `results_random.json` | 07-23 | 2 | – | – | **void**, same cause. Random baseline. |
| `results_mvp.json` | 07-24 | 2 | – | – | **void**, same cause. |
| `results_gistify_heuristic.json` | 07-28 | 6 | 1,444 | 95.04% | **void** — pre-audit; the 1,444 queries are the tell, a sound run of these six costs ~10,000 |
| `results_gistify_heuristic_coverage.json` | 07-28 | 6 | 1,466 | 95.04% | **void**, same run with coverage pruning on |
| `results_gistify_baseline_pinned.json` | 07-28 | 6 | 3,710 | 52.44% | **void** |
| `results_gistify_heuristic_oracle.json` | 07-28 | 6 | 3,449 | 34.03% | **void** — the learned-oracle arm whose 55% query saving [did not survive re-measurement](../README.md#the-learned-oracle-re-measured) |
| `results_gistify_v3_all_fixes.json` | 07-28 | 6 | 10,425 | 94.38% | **void** — [the reducer gutted the test](../README.md#the-reducer-gutted-the-test-instead-of-the-code) |
| `results_gistify_v4.json` | 07-29 | 6 | 12,594 | 97.24% | **void**, same cause |
| `results_gistify_v5_sound.json` | 07-29 | 6 | 12,171 | 88.46% | **void** — named "sound" before [the stale-bytecode bug](../README.md#the-oracle-judged-code-that-was-not-on-disk) was known |
| `results_gistify_baseline_real.json` | 07-29 | 6 | 13,066 | 88.46% | **void**, same cause |
| `results_gistify_oracle_heldout.json` | 07-29 | 6 | 12,212 | 88.46% | **void**, same cause |
| `results_gistify_heuristic_bounded.json` | 07-29 | 6 | 11,676 | 88.46% | **void**, same cause |
| `results_gistify_oracle_bounded.json` | 07-29 | 6 | 11,006 | 88.45% | **void**, same cause |
| `results_gistify_postbytecode.json` | 07-30 | 6 | 11,126 | 88.46% | first run after the bytecode fix; superseded |
| `results_gistify_charoffsets.json` | 07-30 | 6 | 10,620 | 88.52% | superseded — after [the byte-vs-char offset fix](../README.md#removals-were-cut-in-the-wrong-place-on-non-ascii-files) |
| `results_gistify_threerepo.json` | 08-01 | 10 | 23,326 | 86.53% | superseded — first run with [tomlkit as a third fold](../README.md#why-tomlkit-is-here) |
| `results_gistify_decorators.json` | 08-09 | 6 | 10,381 | 88.69% | superseded — after [decorated code became removable](../README.md#decorated-code-could-never-be-removed) |
| `results_gistify_tomlkit_fixed.json` | 08-10 | 4 | 12,550 | 83.17% | tomlkit only, after its hung-query fix |
| `results_gistify_provision.json` | 08-11 | 10 | 22,931 | 87.03% | superseded |
| `results_gistify_provision2.json` | 08-11 | 10 | 22,931 | 87.03% | repeat of the above — see determinism below |
| `results_gistify_afterfixes.json` | 08-15 | 10 | 22,931 | 87.03% | repeat of the above — see determinism below |
| `results_swebench.json` | 08-15 | 7 | 180,507 | 98.97% | current, and the only SWE-bench run — [what happened](../README.md#what-happened-when-it-was-pointed-at-swe-bench) |
| **`results_gistify_pruneblocks.json`** | **08-27** | **10** | **21,065** | **87.02%** | **current.** This is the file the [results table](../README.md#results) is built from. |

### The three repeats are a determinism check, not clutter

`provision`, `provision2` and `afterfixes` look redundant and report the
same 22,931 queries and 87.03%. They are three separate runs, and their
**per-task query counts and final line counts are identical in all
three** — the work is bit-reproducible. What is *not* reproducible is
wall clock: the same work took 3.32 h, 2.97 h and 4.20 h.

That 1.4× spread on identical work is why this repository quotes queries
and not hours. Roughly 98% of reduction time is the oracle subprocess, so
a loaded machine stretches the clock without changing a single decision.

## Four modules in here are not wired to anything

`baselines.py`, `metrics.py`, `simple_runner.py` and `benchmark_runner.py`
are MVP-era and nothing in the repository calls them. They are kept for
the record, but **do not run `simple_runner.py`**, and do not read numbers
out of what it produces:

* It calls the reducer with `test_command=None`. There is then no failure
  to preserve, so every candidate validates, nothing is removed, and the
  run still reports `success: true` with a 100% success rate after
  burning 46 queries. Measured on `bug_original.py`: 45 lines in,
  45 lines out, against 45 → 7 from the supported CLI on the same file.
* Its `behavior_preserved` metric is assigned from `result.success`, so
  it cannot ever contradict the reducer, and `minimized_executed_lines`
  is hardcoded to `0`. Both are the [unprotected-oracle
  pattern](../README.md#the-check-that-catches-these-is-a-vacuity-probe-not-a-size-assertion)
  this project has fixed elsewhere.
* It writes its output over `automre/examples/simple_single_file/bug_minimized.py`
  and `bug2_minimized.py`, which are **tracked files**. Since the
  reduction is a no-op, running it replaces the hand-written expected
  output with the unreduced original.

Passing a real `test_command` instead does not rescue it: the validator
now refuses that call shape outright, because a command with no
`target_file` would run against a file that is not the candidate and
every candidate would pass. The harness predates that guard.

So the ddmin / syntax-guided / random comparison this directory was meant
to provide does not currently exist. Re-establishing it means driving
those baselines through the multi-file path in `gistify_runner.py`, not
reviving this harness.

## Reproducing the current numbers

```bash
python3 evaluation/gistify_runner.py --only requests --out /tmp/requests.json
```

Four tasks, roughly three minutes each. Query counts should match the
`requests` rows of the results table exactly; wall clock will not.
