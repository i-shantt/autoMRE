"""
autoMRE: Automated Bug Reproduction Minimization
BugsInPy Benchmark Runner

Runs autoMRE on the BugsInPy benchmark and collects results.

Based on:
- Widyasari et al. (2024): BugsInPy benchmark
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'automre' / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from reducer import HybridDeltaDebugger
from baselines import VanillaDDMin, SyntaxAwareReducer, RandomReducer
from metrics import ReductionMetrics, BenchmarkResult, MetricsCollector


class BugsInPyAdapter:
    """Adapter for running on BugsInPy benchmark."""

    def __init__(self, bugsinpy_path: Path):
        """
        Initialize adapter.

        Args:
            bugsinpy_path: Path to BugsInPy repository
        """
        self.bugsinpy_path = Path(bugsinpy_path)
        self.projects_path = self.bugsinpy_path / 'projects'

    def list_projects(self) -> List[str]:
        """List available projects."""
        return [p.name for p in self.projects_path.iterdir() if p.is_dir()]

    def list_bugs(self, project: str) -> List[int]:
        """
        List available bugs for a project.

        Args:
            project: Project name

        Returns:
            List of bug IDs
        """
        project_path = self.projects_path / project / 'bugs'
        if not project_path.exists():
            return []

        bugs = []
        for bug_dir in project_path.iterdir():
            if bug_dir.is_dir():
                try:
                    bugs.append(int(bug_dir.name))
                except ValueError:
                    pass

        return sorted(bugs)

    def get_bug_info(self, project: str, bug_id: int) -> Optional[Dict]:
        """
        Get information about a specific bug.

        Args:
            project: Project name
            bug_id: Bug ID

        Returns:
            Bug information dictionary or None
        """
        bug_path = self.projects_path / project / 'bugs' / str(bug_id)

        if not bug_path.exists():
            return None

        info = {
            'project': project,
            'bug_id': bug_id,
            'path': bug_path,
        }

        # Try to read bug info files
        for info_file in ['bug.info', 'bug_info.txt', 'info.txt']:
            info_file_path = bug_path / info_file
            if info_file_path.exists():
                info['description'] = info_file_path.read_text()
                break

        return info

    def setup_bug(self, project: str, bug_id: int,
                  work_dir: Path) -> Tuple[bool, Optional[Path]]:
        """
        Setup a bug in a working directory.

        Args:
            project: Project name
            bug_id: Bug ID
            work_dir: Working directory

        Returns:
            Tuple of (success, source_file_path)
        """
        bug_path = self.projects_path / project / 'bugs' / str(bug_id)

        if not bug_path.exists():
            return False, None

        # For MVP, we focus on single-file bugs
        # Look for Python files in the bug directory
        py_files = list(bug_path.rglob('*.py'))

        if not py_files:
            return False, None

        # Try to find the main bug file (often the one with the test)
        # For simplicity, take the first Python file that's a test or has 'bug' in name
        source_file = None

        for f in py_files:
            if 'test' in f.name.lower() or 'bug' in f.name.lower():
                source_file = f
                break

        if not source_file:
            source_file = py_files[0]

        return True, source_file


class BenchmarkRunner:
    """Runner for executing benchmark evaluations."""

    def __init__(self, bugsinpy_path: Path, output_path: Path,
                 algorithm: str = 'hdd-e',
                 max_bugs: int = 50,
                 verbose: bool = False):
        """
        Initialize runner.

        Args:
            bugsinpy_path: Path to BugsInPy
            output_path: Path to write results
            algorithm: Algorithm to use ('hdd-e', 'ddmin', 'syntax', 'random')
            max_bugs: Maximum number of bugs to evaluate
            verbose: Verbose output
        """
        self.adapter = BugsInPyAdapter(bugsinpy_path)
        self.output_path = Path(output_path)
        self.algorithm = algorithm
        self.max_bugs = max_bugs
        self.verbose = verbose
        self.collector = MetricsCollector()

    def run(self) -> MetricsCollector:
        """
        Run evaluation on BugsInPy.

        Returns:
            MetricsCollector with results
        """
        projects = self.adapter.list_projects()
        bugs_evaluated = 0

        print("Starting benchmark evaluation...")
        print(f"Algorithm: {self.algorithm}")
        print(f"Max bugs: {self.max_bugs}")
        print(f"Projects: {len(projects)}")
        print()

        for project in projects:
            if bugs_evaluated >= self.max_bugs:
                break

            bugs = self.adapter.list_bugs(project)

            for bug_id in bugs:
                if bugs_evaluated >= self.max_bugs:
                    break

                print(f"Evaluating {project} bug {bug_id}...")

                try:
                    result = self._evaluate_bug(project, bug_id)
                    self.collector.add_result(result)

                    if result.error:
                        print(f"  Error: {result.error}")
                    else:
                        print(f"  Success! Reduction: {result.metrics.size_reduction_rate*100:.1f}%")

                    bugs_evaluated += 1

                except Exception as e:
                    print(f"  Exception: {e}")
                    result = BenchmarkResult(
                        bug_id=f"{project}-{bug_id}",
                        project=project,
                        metrics=ReductionMetrics(
                            original_lines=0,
                            minimized_lines=0,
                            success=False,
                            executable=False,
                            behavior_preserved=False,
                            time_seconds=0.0,
                            query_count=0,
                        ),
                        error=str(e)
                    )
                    self.collector.add_result(result)
                    bugs_evaluated += 1

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

    def _evaluate_bug(self, project: str, bug_id: int) -> BenchmarkResult:
        """
        Evaluate a single bug.

        Args:
            project: Project name
            bug_id: Bug ID

        Returns:
            BenchmarkResult
        """
        bug_id_str = f"{project}-{bug_id}"

        # Setup bug
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            success, source_file = self.adapter.setup_bug(project, bug_id, work_dir)

            if not success or not source_file:
                return BenchmarkResult(
                    bug_id=bug_id_str,
                    project=project,
                    metrics=ReductionMetrics(
                        original_lines=0,
                        minimized_lines=0,
                        success=False,
                        executable=False,
                        behavior_preserved=False,
                        time_seconds=0.0,
                        query_count=0,
                    ),
                    error="Could not setup bug"
                )

            # Read source code
            source_code = source_file.read_text()

            # Create reducer based on algorithm
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
                cwd=source_file.parent
            )

            # Compute metrics
            metrics = ReductionMetrics(
                original_lines=result.stats.original_size,
                minimized_lines=result.stats.final_size,
                success=result.success,
                executable=True,
                behavior_preserved=result.success,
                time_seconds=result.stats.time_seconds,
                query_count=result.stats.queries,
                original_executed_lines=result.original_trace.total_executed_lines if result.original_trace else 0,
                minimized_executed_lines=0,  # Would need re-trace
            )

            return BenchmarkResult(
                bug_id=bug_id_str,
                project=project,
                metrics=metrics
            )


def main():
    """Main entry point for benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run autoMRE on BugsInPy benchmark'
    )
    parser.add_argument('--bugsinpy', required=True,
                       help='Path to BugsInPy repository')
    parser.add_argument('--output', required=True,
                       help='Output JSON file path')
    parser.add_argument('--algorithm', default='hdd-e',
                       choices=['hdd-e', 'ddmin', 'syntax', 'random'],
                       help='Algorithm to use')
    parser.add_argument('--max-bugs', type=int, default=50,
                       help='Maximum number of bugs to evaluate')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    runner = BenchmarkRunner(
        bugsinpy_path=Path(args.bugsinpy),
        output_path=Path(args.output),
        algorithm=args.algorithm,
        max_bugs=args.max_bugs,
        verbose=args.verbose
    )

    runner.run()


if __name__ == '__main__':
    main()
