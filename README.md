# AutoRepro-Min: Automated Bug Reproduction Minimization

A Python tool for automatically minimizing bug-triggering code while preserving reproduction capability. Works on both **single files** and **whole multi-file projects**, with an optional **local, open-source LLM** to guide the reduction.

## Overview

When developers receive bug reports, they often contain large amounts of code spread across multiple files. Manually identifying the minimal subset that reproduces the bug is time-consuming. AutoRepro-Min automates this process with two layered algorithms:

- **HDD-E** (Hybrid Delta Debugging with Execution guidance) — the single-file core: AST-aware hierarchical reduction using tree-sitter, prioritized by coverage.py execution data.
- **MF-HDD-E** — the multi-file extension: analyzes a project's import graph, deletes unreachable files, inlines chained dependencies, then applies HDD-E per surviving file with whole-project validation.

Both algorithms use a pluggable **prioritizer** — the strategy that decides which candidate code unit to try removing next. Two ship today:

- `--prioritizer=heuristic` (default) — cold-code-first, deterministic, zero dependencies.
- `--prioritizer=llm --model={tiny,small,medium,large,alt}` — a locally-run open-source coder LLM (Qwen2.5-Coder family + CodeGemma-2B) reads the error and ranks candidates by "least likely relevant to the bug." No API keys, no billing — you pick a model that fits your hardware.

## Quick Start

### Installation

```bash
git clone https://github.com/i-shantt/autorepro-min
cd autorepro-min

# Core install (heuristic prioritizer, no ML dependencies)
pip install .

# Add local open-source LLM prioritizer
pip install .[llm]

# Add benchmarking + matplotlib comparison charts
pip install .[bench]
```

The `[llm]` extra pulls in `torch`, `transformers`, and `accelerate`. The core install is intentionally lightweight so the tool is usable even on machines that can't run local LLMs.

### Usage

```bash
# Single-file reduction (heuristic — default)
python autorepro_min.py reduce bug.py -o bug.min.py -v

# Whole-project reduction (multi-file MF-HDD-E)
python autorepro_min.py reduce-project my_project/ -c "python main.py" -v

# Same, but with an LLM-guided prioritizer (needs pip install .[llm])
python autorepro_min.py reduce-project my_project/ \
    -c "python main.py" \
    --prioritizer llm --model small -v

# Trace / validate helpers
python autorepro_min.py trace bug.py
python autorepro_min.py validate bug.min.py -r bug.py
```

### Python API

```python
from autorepro_min.src.reducer import HybridDeltaDebugger

# Create reducer
reducer = HybridDeltaDebugger(verbose=True)

# Reduce code
with open('bug.py') as f:
    source_code = f.read()

result = reducer.reduce(source_code)

print(f"Original: {result.stats.original_size} lines")
print(f"Minimized: {result.stats.final_size} lines")
print(f"Reduction: {result.stats.reduction_rate*100:.1f}%")

# Save minimized code
with open('bug.min.py', 'w') as f:
    f.write(result.minimized_code)
```

## How It Works

1. **Parse**: Build AST and identify removable units (functions, classes, statements)
2. **Trace**: Run code with coverage to identify executed lines
3. **Prioritize**: Sort units by execution count (cold first) and size
4. **Reduce**: Iteratively remove units and validate behavior preservation
5. **Output**: Return minimized, self-contained reproduction

## Example

**Original Code (45 lines)**:
```python
# Unused import
import json

# Unused function
def helper(x):
    return x * 2

# Unused class
class DataProcessor:
    def process(self, data):
        return data.upper()

# The bug
def trigger_bug():
    x = 42
    return len(x)  # TypeError: object of type 'int' has no len()

if __name__ == "__main__":
    trigger_bug()
```

**Minimized Code (9 lines)**:
```python
def trigger_bug():
    x = 42
    return len(x)

if __name__ == "__main__":
    trigger_bug()
```

**Reduction**: 80% fewer lines while preserving the same TypeError.

## Algorithms

AutoRepro-Min separates the *search structure* (how to explore candidate reductions) from the *prioritization strategy* (which candidate to try next). Both are pluggable.

**Search structures:**

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `hdd-e` (default) | Hybrid Delta Debugging with Execution guidance | Single-file bugs |
| `mf-hdd-e` (`reduce-project`) | Multi-file: import-graph analysis + inlining + per-file HDD-E | Whole projects |
| `ddmin` | Vanilla line-level delta debugging | Maximum reduction, slow |

**Prioritization strategies:**

| Strategy | Description | Extras |
|----------|-------------|--------|
| `heuristic` (default) | Cold-code-first, size-descending. Deterministic, zero deps. | none |
| `llm` | Local open-source coder LLM ranks candidates from stack trace | `pip install .[llm]` |

**LLM model tiers** (all locally-run open weights, no API keys):

| Tier | Model | Params | Default precision on CUDA | Loaded size |
|------|-------|--------|---------------------------|-------------|
| `tiny` | Qwen2.5-Coder-0.5B-Instruct | 0.5B | fp16 (full) | ~1.2 GB |
| `small` | Qwen2.5-Coder-1.5B-Instruct | 1.5B | fp16 (full) | ~3.2 GB |
| `medium` | Qwen2.5-Coder-3B-Instruct | 3B | **4-bit NF4** | ~2.1 GB |
| `large` | Qwen2.5-Coder-7B-Instruct | 7B | **4-bit NF4** | ~4.5 GB |
| `alt` | CodeGemma-2B-it | 2B | **4-bit NF4** | ~1.6 GB |

**Quantization policy**: the small tiers (`tiny`, `small`) already fit in 4 GB at fp16, so we run them full-precision — quantization loss hurts small models more than it helps. The bigger tiers (`medium`, `large`, `alt`) default to 4-bit NF4 via `bitsandbytes` on CUDA so a 6 GB VRAM card fits every option. Pass `--no-quantize` to force full precision on the quantized tiers. On Apple Silicon (MPS) and CPU, `bitsandbytes` isn't supported so every tier runs at fp16/fp32 respectively.

### Hardware guidance

- **Apple Silicon Mac (M1/M2/M3/M4):** `tiny` and `small` are snappy; `medium` is usable (10–20s per ranking); `large` only if you have M2 Max or better. Uses MPS automatically.
- **NVIDIA GTX 1660 Ti / RTX 20-series (6 GB):** every tier fits at 4-bit. Recommended: `small` for iteration, `medium` for real runs.
- **RTX 30-series and later (8 GB+):** any tier, no compromises.
- **CPU only:** `tiny` is fine, `small` is slow but usable, `medium+` will exhaust your patience.

If `.[llm]` isn't installed or a requested model can't be loaded, the tool logs a warning and falls back to the heuristic — the reducer never fails because the ML side failed.

## Evaluation

The repo ships a small, always-reproducible multi-file benchmark (three synthetic bugs under `autorepro_min/examples/multi_file/`) plus a harness for running any prioritizer against it.

```bash
# Heuristic baseline
python evaluation/bugsinpy_runner.py --prioritizer heuristic

# Local LLM path (requires pip install .[llm])
python evaluation/bugsinpy_runner.py --prioritizer llm --model small

# Full matrix across all prioritizer configs + comparison table + chart
python evaluation/model_comparison.py
```

`model_comparison.py` emits `evaluation/results_bench_comparison.md` (a markdown table) and, with matplotlib available, `evaluation/results_bench_comparison.png` (a bar chart of line-reduction / queries / wall-time by prioritizer).

### Preliminary results on the built-in benchmark (Apple M4, 16 GB, MPS)

Full matrix from `python evaluation/model_comparison.py` on a base-model MacBook Air (M4, 16 GB unified memory, MPS backend, fp16 for all LLM tiers since `bitsandbytes` is CUDA-only):

| Prioritizer | Success | Median file red. | Median line red. | Median queries | Median time /bug |
|-------------|---------|------------------|------------------|----------------|------------------|
| heuristic | 3/3 | 75.0% | 61.0% | 9.0 | 0.28s |
| tiny (Qwen 0.5B) | 3/3 | 75.0% | 61.0% | 9.0 | 1.38s |
| small (Qwen 1.5B) | 3/3 | 75.0% | 61.0% | 9.0 | 1.82s |
| medium (Qwen 3B) | 3/3 | 75.0% | 61.0% | 9.0 | 4.77s |
| alt (CodeGemma 2B) | — | — | — | — | — |

**Honest read of the numbers:** on this tiny 3-project built-in benchmark, all prioritizers produce identical output — the LLM adds no reduction quality. This is expected: Phase 4 (per-file HDD-E) barely runs after Phases 1–3 do most of the work on small projects. **The one real signal**: `small` used 11 queries on project3 (side-effects test) vs. heuristic's 14 — a ~20% query saving on the hardest of the three bugs. Wall time scales with model size.

Meaningful discrimination requires a larger, harder benchmark (see Roadmap — Gistify head-to-head).

**Notes on `alt` (CodeGemma-2B-it)**: CodeGemma is a **gated** model on Hugging Face. To include it in the comparison you must (a) accept the license at `https://huggingface.co/google/codegemma-2b-it`, (b) create a token at `https://huggingface.co/settings/tokens`, and (c) `export HF_TOKEN=...` before running. Without a token the tool falls back to heuristic and marks the result row `⚠️FALLBACK` so you don't confuse it with real LLM data.

### Legacy single-file baselines

Single-file HDD-E vs. classic algorithms on the two-bug README example set:

| Algorithm | Success | Reduction | Time | Queries |
|-----------|---------|-----------|------|---------|
| hdd-e | 100% | 36.5% | 1.06s | 11.5 |
| ddmin | 100% | 87.6% | 25.09s | 635.5 |
| syntax | 100% | 36.5% | 0.42s | 8.5 |
| random | 100% | 19.2% | 0.59s | 8.5 |

## Project Structure

```
autorepro_min/
├── src/
│   ├── parser.py            # tree-sitter AST parsing + CodeUnit
│   ├── tracer.py            # coverage.py execution tracing
│   ├── validator.py         # single-file behavior oracle
│   ├── reducer.py           # HDD-E and vanilla ddmin
│   ├── cli.py               # `reduce` / `reduce-project` / `trace` / ...
│   ├── multi_file/          # MF-HDD-E multi-file extension
│   │   ├── dependency_analyzer.py    # Phase 1: import graph + coverage
│   │   ├── multi_file_validator.py   # whole-project oracle
│   │   ├── import_inliner.py         # Phase 3: inline used defs
│   │   └── multi_file_debugger.py    # Phase 1-4 orchestrator
│   └── ml/                  # Prioritization strategy layer
│       ├── prioritizers.py           # Heuristic / LLM prioritizers
│       ├── llm_backend.py            # transformers-based inference
│       └── model_registry.py         # tiny/small/medium/large/alt
├── examples/
│   ├── simple_single_file/  # single-file example bugs
│   └── multi_file/          # 3 multi-file benchmark bugs
└── tests/
    └── test_prioritizers.py # unit tests (no model downloads needed)

evaluation/
├── metrics.py               # metric definitions
├── baselines.py             # ddmin / syntax / random baselines
├── simple_runner.py         # legacy single-file eval
├── benchmark_dataset.py     # curated multi-file benchmark
├── bugsinpy_runner.py       # per-config benchmark harness
└── model_comparison.py      # matrix runner + comparison chart
```

## Research Background

AutoRepro-Min is based on research in:

- **Delta Debugging** (Zeller & Hildebrandt, 2002): Systematic input minimization
- **Hierarchical DD** (Misherghi & Su, 2006): Structure-aware reduction
- **Weighted DD** (Zhou et al., 2024): Prioritization by element size
- **Gistify** (Lee et al., 2025): Codebase-level understanding

See `writing_space/literature_review.md` for full survey.

## Limitations

- **Conservative inliner**: refuses to inline modules whose top-level code has side effects, or that importers use via bare `import mod` / `from mod import *`. Preserves correctness at the cost of leaving some files intact.
- **Python only**: tree-sitter grammar swap is straightforward but not yet wired.
- **BugsInPy checkout not automated**: the benchmark harness accepts external project directories, but per-bug Python-version and dependency management is left to the user (BugsInPy's own `framework/bin/` scripts).
- **No dependency reconstruction**: after removal, we don't attempt to repair broken references.

## Roadmap

- Fine-tuned `learned` prioritizer distilled from the LLM run traces (PyTorch, planned).
- Full BugsInPy adapter with automated per-bug env management.
- Expression-level intra-statement reduction (fixing the parser's structural-child filter).
- Parallel validation for faster reduction.
- Additional language grammars.

## Contributing

Contributions welcome — issues and PRs for the roadmap items above are a great place to start.

## License

MIT License - see LICENSE file for details.

## Citation

If you use AutoRepro-Min in your research, please cite:

```bibtex
@software{autorepro_min,
  title={AutoRepro-Min: Automated Bug Reproduction Minimization},
  year={2026},
  url={https://github.com/<repository>}
}
```

## Acknowledgments

- BugsInPy benchmark (Widyasari et al., 2024)
- tree-sitter parsing library
- coverage.py tracing library
