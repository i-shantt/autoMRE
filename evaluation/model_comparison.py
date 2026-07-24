"""
AutoRepro-Min: Multi-Prioritizer Comparison Runner

Runs the benchmark harness across every prioritizer configuration and
emits a comparison table (JSON + markdown) plus a matplotlib bar chart
for the README.

The full matrix is:
    heuristic
    llm × {tiny, small, medium, large, alt}

That's expensive if all five models actually run — the outer script
iterates them serially and skips any config that fails to build a
backend (e.g. because `pip install .[llm]` hasn't been run). You can
also select a subset via `--configs`.

Usage:
    python evaluation/model_comparison.py                          # all
    python evaluation/model_comparison.py --configs heuristic,small
    python evaluation/model_comparison.py --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))
sys.path.insert(0, str(_ROOT / "evaluation"))

from benchmark_dataset import load_dataset  # noqa: E402
from bugsinpy_runner import run_all, summarize  # noqa: E402


DEFAULT_CONFIGS: List[Tuple[str, Optional[str]]] = [
    ("heuristic", None),
    ("llm", "tiny"),
    ("llm", "small"),
    ("llm", "medium"),
    ("llm", "large"),
    ("llm", "alt"),
]


def _config_key(prioritizer: str, model: Optional[str]) -> str:
    return prioritizer if model is None else f"{prioritizer}_{model}"


def _config_label(prioritizer: str, model: Optional[str]) -> str:
    if prioritizer == "heuristic":
        return "heuristic"
    return model or "llm"


def parse_configs(spec: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    if not spec:
        return DEFAULT_CONFIGS
    out: List[Tuple[str, Optional[str]]] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "heuristic":
            out.append(("heuristic", None))
        elif token in {"tiny", "small", "medium", "large", "alt"}:
            out.append(("llm", token))
        else:
            raise ValueError(f"unknown config: {token!r}")
    return out


def render_markdown_table(rows: List[Dict]) -> str:
    header = ("| Prioritizer | Success | Median file red. | "
              "Median line red. | Median queries | Median time |\n"
              "|-------------|---------|------------------|"
              "------------------|----------------|-------------|")
    lines = [header]
    for r in rows:
        s = r["summary"]
        lines.append(
            f"| {r['label']} "
            f"| {s['n_success']}/{s['n_bugs']} "
            f"| {s['median_file_reduction']*100:.1f}% "
            f"| {s['median_line_reduction']*100:.1f}% "
            f"| {s['median_queries']:.1f} "
            f"| {s['median_time_seconds']:.2f}s |")
    return "\n".join(lines)


def maybe_render_chart(rows: List[Dict], output_path: Path) -> Optional[Path]:
    """Emit a bar chart if matplotlib is installed; None otherwise."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [r["label"] for r in rows]
    line_red = [r["summary"]["median_line_reduction"] * 100 for r in rows]
    queries = [r["summary"]["median_queries"] for r in rows]
    latency = [r["summary"]["median_time_seconds"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(labels, line_red, color="#4c72b0")
    axes[0].set_title("Median line reduction (%) — higher is better")
    axes[0].set_ylim(0, 100)
    axes[1].bar(labels, queries, color="#dd8452")
    axes[1].set_title("Median queries per bug — lower is better")
    axes[2].bar(labels, latency, color="#55a868")
    axes[2].set_title("Median wall time per bug (s) — lower is better")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("AutoRepro-Min prioritizer comparison "
                 "(built-in multi-file benchmark)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AutoRepro-Min benchmark across multiple "
                    "prioritizer configurations and emit a comparison.")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated subset "
                             "(heuristic,tiny,small,medium,large,alt). "
                             "Default: all six.")
    parser.add_argument("--output-dir", default=str(_ROOT / "evaluation"),
                        help="Where to write per-config JSON + summary.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = parse_configs(args.configs)
    bugs = load_dataset()
    if not bugs:
        print("No benchmark projects found.", file=sys.stderr)
        return 1

    all_rows: List[Dict] = []
    for prioritizer, model in configs:
        key = _config_key(prioritizer, model)
        label = _config_label(prioritizer, model)
        print(f"\n=== {label} ({key}) ===")
        try:
            results = run_all(bugs, prioritizer, model,
                              verbose=args.verbose)
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue
        summary = summarize(results)

        per_config_path = output_dir / f"results_bench_{key}.json"
        per_config_path.write_text(json.dumps({
            "config": {"prioritizer": prioritizer, "model": model},
            "summary": summary,
            "runs": [asdict(r) for r in results],
        }, indent=2))

        all_rows.append({
            "key": key,
            "label": label,
            "prioritizer": prioritizer,
            "model": model,
            "summary": summary,
        })

    if not all_rows:
        print("No configurations completed.", file=sys.stderr)
        return 1

    # Write consolidated comparison files.
    comp_json = output_dir / "results_bench_comparison.json"
    comp_json.write_text(json.dumps(all_rows, indent=2))

    md = render_markdown_table(all_rows)
    (output_dir / "results_bench_comparison.md").write_text(md + "\n")

    chart_path = maybe_render_chart(
        all_rows, output_dir / "results_bench_comparison.png")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(md)
    print(f"\nWritten to: {comp_json}")
    if chart_path:
        print(f"Chart:      {chart_path}")
    else:
        print("Chart:      skipped (matplotlib not installed; "
              "install with `pip install .[bench]`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
