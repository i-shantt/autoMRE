"""
AutoRepro-Min: Automated Bug Reproduction Minimization
Command-Line Interface

Usage:
    python -m autorepro_min reduce [options] <file>
    python -m autorepro_min validate [options] <file>
    python -m autorepro_min trace [options] <file>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from parser import PythonParser
from tracer import ExecutionTracer
from validator import Validator
from reducer import HybridDeltaDebugger, LineLevelDeltaDebugger
from multi_file import MultiFileDebugger
from ml import build_prioritizer


def cmd_reduce(args):
    """Execute the reduce command."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    # Read source code
    source_code = file_path.read_text()

    print(f"AutoRepro-Min: Reducing {file_path}")
    print(f"Original size: {len(source_code.split(chr(10)))} lines")
    print()

    # Create reducer
    if args.algorithm == "hdd-e":
        prioritizer = build_prioritizer(
            kind=getattr(args, "prioritizer", "heuristic"),
            model=getattr(args, "model", None),
            verbose=args.verbose,
        )
        reducer = HybridDeltaDebugger(prioritizer=prioritizer,
                                      verbose=args.verbose)
    else:
        reducer = LineLevelDeltaDebugger(verbose=args.verbose)

    # Build test command if provided
    test_command = None
    if args.command:
        test_command = args.command.split()
    elif args.pytest:
        test_command = ['pytest', str(file_path), '-v']

    # Run reduction
    result = reducer.reduce(
        source_code=source_code,
        test_command=test_command,
        cwd=file_path.parent
    )

    print()
    print("=" * 60)
    print("REDUCTION RESULTS")
    print("=" * 60)
    print(f"Original size:  {result.stats.original_size} lines")
    print(f"Minimized size: {result.stats.final_size} lines")
    print(f"Reduction rate: {result.stats.reduction_rate*100:.1f}%")
    print(f"Iterations:     {result.stats.iterations}")
    print(f"Queries:        {result.stats.queries}")
    print(f"Time:           {result.stats.time_seconds:.2f}s")
    print()

    # Write output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = file_path.with_suffix('.min.py')

    output_path.write_text(result.minimized_code)
    print(f"Minimized code written to: {output_path}")

    return 0


def cmd_validate(args):
    """Execute the validate command."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    source_code = file_path.read_text()

    print(f"AutoRepro-Min: Validating {file_path}")

    # Create validator
    validator = Validator(match_strategy=args.strategy)

    # Capture original behavior if reference provided
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            from tracer import ExecutionTracer
            tracer = ExecutionTracer()

            test_command = None
            if args.command:
                test_command = args.command.split()

            if test_command:
                trace = tracer.trace_command(test_command, cwd=ref_path.parent)
            else:
                trace = tracer.trace_python_file(ref_path)

            validator.set_original_behavior(trace.output,
                                           0 if trace.success else 1)
            print(f"Original behavior captured from: {ref_path}")
        else:
            print(f"Error: Reference file not found: {ref_path}", file=sys.stderr)
            return 1

    # Validate
    test_command = None
    if args.command:
        test_command = args.command.split()

    result = validator.validate(source_code, test_command, cwd=file_path.parent)

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Valid:                  {result.is_valid}")
    print(f"Matches original:       {result.matches_original}")
    print(f"Error type match:       {result.error_type_match}")
    print(f"Message similarity:     {result.error_message_similarity:.2f}")
    print(f"Return code:            {result.return_code}")
    print()

    if args.verbose:
        print("Output:")
        print("-" * 40)
        print(result.output)
        print("-" * 40)

    return 0 if result.is_valid else 1


def cmd_trace(args):
    """Execute the trace command."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    print(f"AutoRepro-Min: Tracing {file_path}")

    # Create tracer
    tracer = ExecutionTracer(timeout=args.timeout)

    # Trace execution
    test_command = None
    if args.command:
        test_command = args.command.split()

    if test_command:
        trace = tracer.trace_command(test_command, cwd=file_path.parent)
    else:
        trace = tracer.trace_python_file(file_path)

    print()
    print("=" * 60)
    print("EXECUTION TRACE")
    print("=" * 60)
    print(f"Success:        {trace.success}")
    print(f"Return code:    {trace.return_code}")
    print(f"Total lines:    {trace.total_executed_lines}")
    print()
    print("Executed lines by file:")
    for path, lines in trace.executed_lines.items():
        print(f"  {path}: {len(lines)} lines")

    if args.verbose:
        print()
        print("Output:")
        print("-" * 40)
        print(trace.output)
        print("-" * 40)

    return 0


def cmd_reduce_project(args):
    """Execute the reduce-project command (multi-file MF-HDD-E)."""
    project_dir = Path(args.project).resolve()

    if not project_dir.exists() or not project_dir.is_dir():
        print(f"Error: project directory not found: {project_dir}",
              file=sys.stderr)
        return 1

    if not args.command:
        print("Error: --command is required (the reproduction command, e.g. "
              "\"python main.py\")", file=sys.stderr)
        return 1

    test_command = args.command.split()

    # Work on a copy unless --in-place was requested.
    if args.in_place:
        work_dir = project_dir
    else:
        if args.output:
            work_dir = Path(args.output).resolve()
        else:
            work_dir = project_dir.parent / f"{project_dir.name}_minimized"
        if work_dir.exists():
            if args.force:
                shutil.rmtree(work_dir)
            else:
                print(f"Error: output directory already exists: {work_dir}. "
                      "Use --force to overwrite or pick another --output.",
                      file=sys.stderr)
                return 1
        shutil.copytree(project_dir, work_dir)
        print(f"Working on copy at: {work_dir}")

    prioritizer = build_prioritizer(
        kind=getattr(args, "prioritizer", "heuristic"),
        model=getattr(args, "model", None),
        verbose=args.verbose,
    )
    debugger = MultiFileDebugger(verbose=args.verbose,
                                 timeout=args.timeout,
                                 match_strategy=args.strategy,
                                 prioritizer=prioritizer)
    result = debugger.reduce_project(work_dir, test_command)

    print()
    print("=" * 60)
    print("MULTI-FILE REDUCTION RESULTS")
    print("=" * 60)
    print(f"Output directory:  {result.project_dir}")
    print(f"Files:             {result.original_file_count} -> "
          f"{result.final_file_count} "
          f"({result.file_reduction_rate*100:.1f}% removed)")
    print(f"Lines:             {result.original_line_count} -> "
          f"{result.final_line_count} "
          f"({result.line_reduction_rate*100:.1f}% removed)")
    print(f"Unreachable dropped: {len(result.unreachable_deleted)}")
    print(f"Imported-only dropped: {len(result.imported_deleted)}")
    print(f"Inlined away:      {len(result.inlined_away)}")
    print(f"Total queries:     {result.total_queries}")
    print(f"Time:              {result.time_seconds:.2f}s")
    return 0


def cmd_parse(args):
    """Execute the parse command."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    print(f"AutoRepro-Min: Parsing {file_path}")

    # Create parser
    parser = PythonParser()
    tree, source = parser.parse_file(file_path)

    # Extract units
    units = parser.extract_units(tree, source)

    print()
    print("=" * 60)
    print("PARSE RESULTS")
    print("=" * 60)
    print(f"Total top-level units: {len(units)}")
    print()

    def print_unit(unit, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{unit.node_type} (L{unit.start_line}-{unit.end_line}, "
              f"{unit.size} lines, exec={unit.execution_count})")
        for child in unit.children:
            print_unit(child, indent + 1)

    for unit in units:
        print_unit(unit)

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog='autorepro_min',
        description='Automated Bug Reproduction Minimization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reduce a Python file
  python -m autorepro_min reduce bug.py -o bug.min.py

  # Reduce with custom command
  python -m autorepro_min reduce test_bug.py -c "pytest test_bug.py -v"

  # Trace execution
  python -m autorepro_min trace bug.py

  # Validate minimized code
  python -m autorepro_min validate bug.min.py -r bug.py
        """
    )

    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    subparsers = parser.add_subparsers(dest='subcommand', help='Available commands')

    # Reduce command
    reduce_parser = subparsers.add_parser('reduce', help='Minimize bug-triggering code')
    reduce_parser.add_argument('file', help='Python file to reduce')
    reduce_parser.add_argument('-o', '--output', help='Output file path')
    reduce_parser.add_argument('-c', '--command', help='Command to reproduce bug')
    reduce_parser.add_argument('--pytest', action='store_true', help='Use pytest to run')
    reduce_parser.add_argument('-a', '--algorithm', default='hdd-e',
                              choices=['hdd-e', 'ddmin'],
                              help='Reduction algorithm (default: hdd-e)')
    reduce_parser.add_argument('--prioritizer', default='heuristic',
                              choices=['heuristic', 'llm'],
                              help='Candidate prioritization strategy '
                                   '(default: heuristic)')
    reduce_parser.add_argument('--model', default=None,
                              choices=['tiny', 'small', 'medium',
                                       'large', 'alt'],
                              help='Model tier when --prioritizer=llm '
                                   '(default: small)')
    reduce_parser.add_argument('-v', '--verbose', action='store_true',
                              help='Verbose output')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate minimized code')
    validate_parser.add_argument('file', help='Python file to validate')
    validate_parser.add_argument('-r', '--reference', help='Reference file for comparison')
    validate_parser.add_argument('-c', '--command', help='Command to execute')
    validate_parser.add_argument('-s', '--strategy', default='error_type',
                                choices=['exact', 'error_type', 'error_message'],
                                help='Validation strategy')
    validate_parser.add_argument('-v', '--verbose', action='store_true',
                                help='Verbose output')

    # Trace command
    trace_parser = subparsers.add_parser('trace', help='Trace execution')
    trace_parser.add_argument('file', help='Python file to trace')
    trace_parser.add_argument('-c', '--command', help='Command to execute')
    trace_parser.add_argument('-t', '--timeout', type=int, default=60,
                             help='Timeout in seconds')
    trace_parser.add_argument('-v', '--verbose', action='store_true',
                             help='Verbose output')

    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse Python file')
    parse_parser.add_argument('file', help='Python file to parse')
    parse_parser.add_argument('-v', '--verbose', action='store_true',
                             help='Verbose output')

    # Reduce-project command (multi-file MF-HDD-E)
    rp = subparsers.add_parser(
        'reduce-project',
        help='Minimize a multi-file Python project (MF-HDD-E)')
    rp.add_argument('project', help='Path to the project directory')
    rp.add_argument('-c', '--command', required=True,
                    help='Reproduction command, e.g. "python main.py"')
    rp.add_argument('-o', '--output',
                    help='Output directory (default: <project>_minimized/)')
    rp.add_argument('--in-place', action='store_true',
                    help='Mutate the project directory directly (dangerous)')
    rp.add_argument('--force', action='store_true',
                    help='Overwrite --output if it already exists')
    rp.add_argument('-t', '--timeout', type=int, default=60,
                    help='Per-run reproduction timeout in seconds')
    rp.add_argument('-s', '--strategy', default='error_type',
                    choices=['exact', 'error_type', 'error_message'],
                    help='Behavior-matching strategy')
    rp.add_argument('--prioritizer', default='heuristic',
                    choices=['heuristic', 'llm'],
                    help='Candidate prioritization strategy '
                         '(default: heuristic)')
    rp.add_argument('--model', default=None,
                    choices=['tiny', 'small', 'medium', 'large', 'alt'],
                    help='Model tier when --prioritizer=llm '
                         '(default: small)')
    rp.add_argument('-v', '--verbose', action='store_true',
                    help='Verbose progress output')

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        return 1

    # Dispatch to appropriate command
    commands = {
        'reduce': cmd_reduce,
        'validate': cmd_validate,
        'trace': cmd_trace,
        'parse': cmd_parse,
        'reduce-project': cmd_reduce_project,
    }

    return commands[args.subcommand](args)


if __name__ == '__main__':
    sys.exit(main())
