"""
autoMRE: Automated Bug Reproduction Minimization
Simple Evaluation Runner

Runs the single-file examples through one algorithm, or all of them with
--compare-all, which is the only place ddmin, random and syntax-guided
reduction are compared against HDD-E.

Nothing in the repository calls this, and the results it last produced —
`results_ddmin.json`, `results_random.json`, `results_syntax.json` — are
listed as void in README.md: they were measured through a single-file
validator that never ran the candidate. That bug is fixed, so the
comparison is worth re-running, but until it is, do not read numbers out
of those files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'automre' / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from reducer import HybridDeltaDebugger
from baselines import VanillaDDMin, SyntaxAwareReducer, RandomReducer
from metrics import ReductionMetrics, BenchmarkResult, MetricsCollector


class SimpleBenchmarkRunner:
    """Runner for simple example-based evaluation."""

    def __init__(self, examples_dir: Path, output_path: Path,
                 algorithm: str = 'hdd-e',
                 verbose: bool = False):
        """
        Initialize runner.

        Args:
            examples_dir: Directory containing example bugs
            output_path: Path to write results
            algorithm: Algorithm to use
            verbose: Verbose output
        """
        self.examples_dir = Path(examples_dir)
        self.output_path = Path(output_path)
        self.algorithm = algorithm
        self.verbose = verbose
        self.collector = MetricsCollector()

    def run(self) -> MetricsCollector:
        """Run evaluation on example bugs."""
        # Find all bug files
        bug_files = sorted(self.examples_dir.glob('bug*_original.py'))

        print(f"Starting evaluation on {len(bug_files)} example bugs...")
        print(f"Algorithm: {self.algorithm}")
        print()

        for bug_file in bug_files:
            bug_id = bug_file.stem.replace('_original', '')
            print(f"Evaluating {bug_id}...")

            try:
                result = self._evaluate_bug(bug_file, bug_id)
                self.collector.add_result(result)

                if result.error:
                    print(f"  Error: {result.error}")
                else:
                    print(f"  Success! Reduction: {result.metrics.size_reduction_rate*100:.1f}%")
                    print(f"  Time: {result.metrics.time_seconds:.2f}s")
                    print(f"  Queries: {result.metrics.query_count}")

            except Exception as e:
                print(f"  Exception: {e}")
                import traceback
                traceback.print_exc()

        # Save results
        self.collector.export_json(self.output_path)
        print()
        print(f"Results saved to: {self.output_path}")

        # Print summary
        summary = self.collector.get_summary()
        print()
        print("=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total bugs: {summary['total_bugs']}")
        print(f"Successful: {summary['successful_reductions']}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        print(f"Avg reduction: {summary['avg_reduction_rate']*100:.1f}%")
        print(f"Avg time: {summary['avg_time_seconds']:.2f}s")
        print(f"Avg queries: {summary['avg_queries']:.1f}")

        return self.collector

    def _evaluate_bug(self, bug_file: Path, bug_id: str) -> BenchmarkResult:
        """Evaluate a single bug."""
        source_code = bug_file.read_text()

        # Create reducer
        if self.algorithm == 'hdd-e':
            reducer = HybridDeltaDebugger(verbose=self.verbose)
        elif self.algorithm == 'ddmin':
            reducer = VanillaDDMin(verbose=self.verbose)
        elif self.algorithm == 'syntax':
            reducer = SyntaxAwareReducer(verbose=self.verbose)
        elif self.algorithm == 'random':
            reducer = RandomReducer(verbose=self.verbose)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Run reduction
        result = reducer.reduce(
            source_code=source_code,
            test_command=None,
            cwd=bug_file.parent
        )

        # Save minimized version for hdd-e
        if self.algorithm == 'hdd-e':
            output_file = bug_file.parent / f"{bug_id}_minimized.py"
            output_file.write_text(result.minimized_code)

        # Compute metrics (handle both ReductionResult and BaselineResult)
        if hasattr(result, 'stats'):
            # ReductionResult from HybridDeltaDebugger
            metrics = ReductionMetrics(
                original_lines=result.stats.original_size,
                minimized_lines=result.stats.final_size,
                success=result.success,
                executable=True,
                behavior_preserved=result.success,
                time_seconds=result.stats.time_seconds,
                query_count=result.stats.queries,
                original_executed_lines=result.original_trace.total_executed_lines if result.original_trace else 0,
                minimized_executed_lines=0,
            )
        else:
            # BaselineResult from baseline algorithms
            metrics = ReductionMetrics(
                original_lines=result.original_size,
                minimized_lines=result.minimized_size,
                success=result.success,
                executable=True,
                behavior_preserved=result.success,
                time_seconds=result.time_seconds,
                query_count=result.queries,
                original_executed_lines=0,
                minimized_executed_lines=0,
            )

        return BenchmarkResult(
            bug_id=bug_id,
            project='examples',
            metrics=metrics
        )


def run_all_algorithms(examples_dir: Path, output_dir: Path):
    """Run all algorithms and compare results."""
    algorithms = ['hdd-e', 'ddmin', 'syntax', 'random']
    results = {}

    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo}...")
        print('='*60)

        output_path = output_dir / f'results_{algo}.json'
        runner = SimpleBenchmarkRunner(
            examples_dir=examples_dir,
            output_path=output_path,
            algorithm=algo,
            verbose=False
        )
        collector = runner.run()
        results[algo] = collector.get_summary()

    # Print comparison
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Success':<10} {'Reduction':<12} {'Time (s)':<12} {'Queries':<10}")
    print("-" * 60)
    for algo, summary in results.items():
        print(f"{algo:<12} "
              f"{summary['success_rate']*100:>6.1f}%    "
              f"{summary['avg_reduction_rate']*100:>6.1f}%      "
              f"{summary['avg_time_seconds']:>6.2f}       "
              f"{summary['avg_queries']:>6.1f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run autoMRE evaluation on example bugs'
    )
    parser.add_argument('--examples', required=True,
                       help='Directory containing example bugs')
    parser.add_argument('--output', required=True,
                       help='Output JSON file path')
    parser.add_argument('--algorithm', default='hdd-e',
                       choices=['hdd-e', 'ddmin', 'syntax', 'random'],
                       help='Algorithm to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--compare-all', action='store_true',
                       help='Run all algorithms for comparison')

    args = parser.parse_args()

    examples_dir = Path(args.examples)
    output_path = Path(args.output)

    if args.compare_all:
        output_dir = output_path.parent
        run_all_algorithms(examples_dir, output_dir)
    else:
        runner = SimpleBenchmarkRunner(
            examples_dir=examples_dir,
            output_path=output_path,
            algorithm=args.algorithm,
            verbose=args.verbose
        )
        runner.run()


if __name__ == '__main__':
    main()
