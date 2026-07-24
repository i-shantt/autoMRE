"""
AutoRepro-Min: Automated Bug Reproduction Minimization
Evaluation Metrics Module

Implements metrics for evaluating reduction quality:
- Size Reduction Rate (SRR)
- Execution Fidelity
- Line Existence Rate
- Line Execution Rate
- Time-to-Minimize

Based on:
- Alipour et al. (2016): Test-case reduction metrics
- Gistify (Lee et al., 2025): Codebase minimization metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class ReductionMetrics:
    """Metrics for evaluating a reduction result."""
    # Size metrics
    original_lines: int
    minimized_lines: int

    # Success metrics
    success: bool  # Was reduction successful?
    executable: bool  # Is minimized code executable?
    behavior_preserved: bool  # Does it preserve original behavior?

    # Time metrics
    time_seconds: float
    query_count: int

    # Coverage metrics (if trace available)
    original_executed_lines: int = 0
    minimized_executed_lines: int = 0

    @property
    def size_reduction_rate(self) -> float:
        """
        Size Reduction Rate (SRR).
        Percentage of lines removed.
        """
        if self.original_lines == 0:
            return 0.0
        return (self.original_lines - self.minimized_lines) / self.original_lines

    @property
    def compression_ratio(self) -> float:
        """
        Compression ratio (original / minimized).
        Higher is better.
        """
        if self.minimized_lines == 0:
            return float('inf')
        return self.original_lines / self.minimized_lines

    @property
    def line_execution_rate(self) -> float:
        """
        Percentage of minimized lines that were executed.
        Measures minimality (higher = less dead code).
        """
        if self.minimized_lines == 0:
            return 0.0
        return self.minimized_executed_lines / self.minimized_lines

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization."""
        return {
            'original_lines': self.original_lines,
            'minimized_lines': self.minimized_lines,
            'size_reduction_rate': self.size_reduction_rate,
            'compression_ratio': self.compression_ratio,
            'success': self.success,
            'executable': self.executable,
            'behavior_preserved': self.behavior_preserved,
            'time_seconds': self.time_seconds,
            'query_count': self.query_count,
            'original_executed_lines': self.original_executed_lines,
            'minimized_executed_lines': self.minimized_executed_lines,
            'line_execution_rate': self.line_execution_rate,
        }


@dataclass
class BenchmarkResult:
    """Result of running on a benchmark bug."""
    bug_id: str
    project: str
    metrics: ReductionMetrics
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'bug_id': self.bug_id,
            'project': self.project,
            'metrics': self.metrics.to_dict(),
        }
        if self.error:
            result['error'] = self.error
        return result


class MetricsCollector:
    """Collects and aggregates metrics from multiple reduction runs."""

    def __init__(self):
        """Initialize collector."""
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def get_summary(self) -> Dict:
        """
        Get summary statistics across all results.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self.results:
            return {}

        successful = [r for r in self.results if r.metrics.success]
        total = len(self.results)

        summary = {
            'total_bugs': total,
            'successful_reductions': len(successful),
            'success_rate': len(successful) / total if total > 0 else 0.0,
            'avg_reduction_rate': 0.0,
            'avg_time_seconds': 0.0,
            'avg_queries': 0.0,
        }

        if successful:
            summary['avg_reduction_rate'] = sum(
                r.metrics.size_reduction_rate for r in successful
            ) / len(successful)
            summary['avg_time_seconds'] = sum(
                r.metrics.time_seconds for r in successful
            ) / len(successful)
            summary['avg_queries'] = sum(
                r.metrics.query_count for r in successful
            ) / len(successful)

        return summary

    def get_project_summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics grouped by project.

        Returns:
            Dictionary mapping project names to summary stats
        """
        by_project: Dict[str, List[BenchmarkResult]] = {}

        for result in self.results:
            if result.project not in by_project:
                by_project[result.project] = []
            by_project[result.project].append(result)

        summaries = {}
        for project, results in by_project.items():
            collector = MetricsCollector()
            collector.results = results
            summaries[project] = collector.get_summary()

        return summaries

    def export_json(self, output_path: Path):
        """
        Export all results to JSON.

        Args:
            output_path: Path to write JSON file
        """
        import json

        data = {
            'summary': self.get_summary(),
            'project_summaries': self.get_project_summary(),
            'results': [r.to_dict() for r in self.results],
        }

        output_path.write_text(json.dumps(data, indent=2))


def compute_line_existence_rate(original_source: str, minimized_source: str) -> float:
    """
    Compute Line Existence Rate (from Gistify).

    Measures what percentage of lines in the minimized output
    exist in the original source (no hallucination).

    Args:
        original_source: Original source code
        minimized_source: Minimized source code

    Returns:
        Line existence rate (0.0 to 1.0)
    """
    original_lines = set(original_source.split('\n'))
    minimized_lines = minimized_source.split('\n')

    if not minimized_lines:
        return 0.0

    existing_count = sum(
        1 for line in minimized_lines if line in original_lines
    )

    return existing_count / len(minimized_lines)


def compute_metrics(original_source: str,
                   minimized_source: str,
                   success: bool,
                   time_seconds: float,
                   query_count: int,
                   original_trace=None,
                   minimized_trace=None) -> ReductionMetrics:
    """
    Compute all metrics for a reduction result.

    Args:
        original_source: Original source code
        minimized_source: Minimized source code
        success: Whether reduction was successful
        time_seconds: Time taken for reduction
        query_count: Number of validation queries
        original_trace: Original execution trace (optional)
        minimized_trace: Minimized execution trace (optional)

    Returns:
        ReductionMetrics object
    """
    original_lines = len(original_source.split('\n'))
    minimized_lines = len(minimized_source.split('\n'))

    # Count executed lines if traces available
    orig_executed = 0
    minimized_executed = 0

    if original_trace:
        orig_executed = original_trace.total_executed_lines
    if minimized_trace:
        minimized_executed = minimized_trace.total_executed_lines

    return ReductionMetrics(
        original_lines=original_lines,
        minimized_lines=minimized_lines,
        success=success,
        executable=True,  # Assumed if we got here
        behavior_preserved=success,
        time_seconds=time_seconds,
        query_count=query_count,
        original_executed_lines=orig_executed,
        minimized_executed_lines=minimized_executed,
    )
