# AutoRepro-Min: Automated Bug Reproduction Minimization

A Python tool for automatically minimizing bug-triggering code while preserving reproduction capability.

## Overview

When developers receive bug reports, they often contain large amounts of code spread across multiple files. Manually identifying the minimal subset that reproduces the bug is time-consuming. AutoRepro-Min automates this process using **Hybrid Delta Debugging (HDD-E)** - a novel algorithm that combines:

- **Hierarchical Delta Debugging**: AST-aware structural reduction
- **Execution-Guided Prioritization**: Coverage data to prioritize cold code removal
- **Multi-Granularity Reduction**: Module → Class → Function → Statement

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd autorepro-min

# Install dependencies
pip install tree-sitter tree-sitter-python coverage
```

### Usage

```bash
# Reduce a single Python file
python autorepro_min.py reduce bug.py -o bug.min.py -v

# Reduce with custom test command
python autorepro_min.py reduce test_bug.py -c "pytest test_bug.py -v"

# Trace execution
python autorepro_min.py trace bug.py

# Validate minimized code
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

AutoRepro-Min implements multiple reduction algorithms:

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **hdd-e** (default) | Hybrid Delta Debugging with Execution guidance | Balanced speed/reduction |
| **ddmin** | Vanilla line-level delta debugging | Maximum reduction (slow) |
| **syntax** | AST-guided without execution data | Fast, moderate reduction |
| **random** | Random unit removal | Sanity check baseline |

## Evaluation

Run evaluation on example bugs:

```bash
# Run single algorithm
python evaluation/simple_runner.py \
    --examples ./autorepro_min/examples/simple_single_file \
    --output results.json \
    --algorithm hdd-e

# Compare all algorithms
python evaluation/simple_runner.py \
    --examples ./autorepro_min/examples/simple_single_file \
    --output results.json \
    --compare-all
```

### Example Results

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
│   ├── parser.py        # AST parsing
│   ├── tracer.py        # Execution tracing
│   ├── validator.py     # Behavior validation
│   ├── reducer.py       # Reduction algorithms
│   └── cli.py           # Command-line interface
├── examples/            # Example bugs
└── tests/               # Unit tests

evaluation/
├── metrics.py           # Evaluation metrics
├── baselines.py         # Baseline algorithms
└── simple_runner.py     # Evaluation runner
```

## Research Background

AutoRepro-Min is based on research in:

- **Delta Debugging** (Zeller & Hildebrandt, 2002): Systematic input minimization
- **Hierarchical DD** (Misherghi & Su, 2006): Structure-aware reduction
- **Weighted DD** (Zhou et al., 2024): Prioritization by element size
- **Gistify** (Lee et al., 2025): Codebase-level understanding

See `writing_space/literature_review.md` for full survey.

## Limitations

- **Single-file focus**: Multi-file inlining not yet implemented
- **Limited BugsInPy integration**: Full benchmark requires setup scripts
- **No dependency reconstruction**: Broken references not repaired

## Contributing

Contributions welcome! Areas for improvement:
- Multi-file support
- Expression-level reduction
- Parallel validation
- Additional language support

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
