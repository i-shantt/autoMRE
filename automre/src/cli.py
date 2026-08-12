"""
autoMRE: Automated Bug Reproduction Minimization
Command-Line Interface

Usage:
    automre reduce [options] <file>
    automre validate [options] <file>
    automre trace [options] <file>

Invoke the `automre` console script, not `python -m automre.src.cli`.
The repo root holds an `automre.py` shim beside the `automre/` package
directory, and that directory has no `__init__.py`, so from the repo root
the shim wins and the module form fails with "automre is not a package".
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from parser import PythonParser
from tracer import ExecutionTracer
from validator import Validator
from reducer import HybridDeltaDebugger, LineLevelDeltaDebugger
from multi_file import MultiFileDebugger


def _version() -> str:
    """The installed version, or a marker that we are not installed.

    Hard-coding it here is how `--version` came to report 0.1.0 while
    pyproject.toml said 0.3.0: two places to change, one of them easy to
    forget. Ask the package metadata instead, so there is one.
    """
    try:
        from importlib.metadata import version
        return version("automre")
    except Exception:
        return "unknown (not installed)"


def _oracle_is_the_subject(file_path: Path, test_command: list):
    """Reason to refuse a single-file reduction, or None.

    `reduce -c "pytest test_bug.py" test_bug.py` asks the reducer to
    minimize the file that decides whether the reduction is any good. For
    a *passing* test that has the degenerate solution the multi-file path
    already refuses: stub the body to `pass`, watch pytest print
    "1 passed", and report a near-total reduction of nothing.

    Multi-file reduction handles this by protecting the test and reducing
    the code around it, which is what this case wants too — hence the
    pointer rather than a flag to override.

    A *failing* command is fine and is left alone: the oracle then
    compares a traceback that names the offending line, so the reduction
    cannot fake it by deleting the cause.
    """
    # Relative arguments resolve against the directory the command runs
    # in, which is the file's parent — not against our cwd.
    run_dir = file_path.parent
    named = set()
    for arg in test_command:
        candidate = arg.split("::", 1)[0]
        if not candidate.endswith(".py"):
            continue
        path = Path(candidate)
        named.add((path if path.is_absolute() else run_dir / path).resolve())
    if file_path.resolve() not in named:
        return None
    if not MultiFileDebugger._is_test_runner(test_command):
        return None

    try:
        proc = subprocess.run(test_command, cwd=file_path.parent,
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    return (f"{file_path.name} is both the file being reduced and the test "
            "the command runs, and that test passes. Emptying it keeps the "
            "command passing, so the reduction would measure nothing. "
            "Reduce the code around it instead:\n"
            f"  automre reduce-project {file_path.parent} "
            f"-c \"{' '.join(test_command)}\"")


def cmd_reduce(args):
    """Execute the reduce command."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    # Read source code
    source_code = file_path.read_text()

    print(f"autoMRE: Reducing {file_path}")
    print(f"Original size: {len(source_code.split(chr(10)))} lines")
    print()

    # Build test command if provided
    test_command = None
    if args.command:
        test_command = args.command.split()
    elif args.pytest:
        test_command = ['pytest', str(file_path), '-v']

    if test_command:
        refusal = _oracle_is_the_subject(file_path, test_command)
        if refusal:
            print(f"Error: {refusal}", file=sys.stderr)
            return 1

    # A command runs the file where it lies, so the oracle has to put each
    # candidate there and restore it afterwards — otherwise the command
    # reads the original and every candidate "passes". Reduction therefore
    # rewrites this file many times before restoring it.
    validator = Validator(target_file=file_path if test_command else None)
    if test_command:
        print(f"Note: {file_path} is rewritten in place during reduction "
              "and restored after each query.")
        print()

    # Create reducer
    if args.algorithm == "hdd-e":
        reducer = HybridDeltaDebugger(verbose=args.verbose,
                                      validator=validator)
    else:
        reducer = LineLevelDeltaDebugger(verbose=args.verbose,
                                         validator=validator)

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

    print(f"autoMRE: Validating {file_path}")

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

    print(f"autoMRE: Tracing {file_path}")

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
        # --force means "overwrite the output". If the output is the project
        # itself, or holds it, that quietly means "delete the input": the
        # tree is removed and the copy that was supposed to replace it then
        # fails with FileNotFoundError, leaving nothing at all. Refuse.
        if work_dir == project_dir or work_dir in project_dir.parents:
            print(f"Error: --output {work_dir} is the project being reduced, "
                  "or a directory containing it. Reducing into it would "
                  "destroy the input. Pick a path outside the project, or "
                  "use --in-place to reduce it where it lies.",
                  file=sys.stderr)
            return 1
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

    debugger = MultiFileDebugger(
        verbose=args.verbose,
        timeout=args.timeout,
        match_strategy=args.strategy,
        aggressive_inline=getattr(args, "aggressive_inline", False),
        use_coverage_prune=not getattr(args, "no_coverage_prune", False),
        use_learned_oracle=getattr(args, "use_learned_oracle", False),
        oracle_model_path=(Path(args.oracle_model)
                           if getattr(args, "oracle_model", None) else None),
        python=getattr(args, "python", None),
    )
    try:
        result = debugger.reduce_project(work_dir, test_command)
    except ValueError as exc:
        # The pre-flight refusals — an unprotected test-runner command,
        # mainly. They fire before the tree is touched, and every other
        # user error in this CLI prints a line and exits; this one should
        # not be the single exception that answers with a stack trace.
        # The copy is removed because it holds nothing, and leaving it
        # would make the next attempt fail on "output already exists"
        # for a reason unrelated to what went wrong.
        print(f"Error: {exc}", file=sys.stderr)
        if not args.in_place and work_dir != project_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        return 1
    except KeyboardInterrupt:
        # A reduction runs for minutes to hours, so Ctrl-C is an ordinary
        # way to end one rather than a crash. The oracle restores the file
        # it was testing in a `finally`, so what is on disk is a state the
        # oracle accepted — partial, but usable, and worth naming.
        print(file=sys.stderr)
        print("Interrupted. The partially reduced project is at "
              f"{work_dir}", file=sys.stderr)
        return 130

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
    if result.protected_line_count:
        # The protected test and its conftest are identical before and
        # after, so they sit in both sides of the total figure and drag it
        # down for a reason that says nothing about the reduction.
        print(f"  of which protected (test/conftest): "
              f"{result.protected_line_count} lines")
        print(f"Reducible lines:   "
              f"{result.original_line_count - result.protected_line_count} -> "
              f"{result.final_line_count - result.protected_line_count} "
              f"({result.reducible_reduction_rate*100:.1f}% removed)")
    if result.undecodable_files:
        # Said out loud rather than left in the gap between the file
        # count and the reduction: these were never candidates, and a
        # run that quietly skipped part of the tree looks exactly like
        # one that examined all of it.
        print(f"Left untouched:    {len(result.undecodable_files)} file(s) "
              f"not valid UTF-8 (cannot be cut safely)")
    if result.outside_project_files:
        # Worth saying out loud rather than counting as a skip: a symlink
        # out of the tree means the project points at code this run is
        # not reducing, which usually surprises whoever set it up.
        print(f"Refused:           {len(result.outside_project_files)} "
              f"file(s) reached by a symlink out of the project")
    print(f"Unreachable dropped: {len(result.unreachable_deleted)}")
    print(f"Imported-only dropped: {len(result.imported_deleted)}")
    print(f"Inlined away:      {len(result.inlined_away)}")
    print(f"Total queries:     {result.total_queries}")
    if result.timed_out_queries:
        # Named rather than folded into the rejections, because a hang is
        # a different event from "the bug broke" and reads as one only if
        # it is counted separately.
        print(f"  of which timed out: {result.timed_out_queries} "
              "(candidate did not terminate; refused)")
    print(f"Time:              {result.time_seconds:.2f}s")
    return 0


def cmd_parse(args):
    """Execute the parse command."""
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    print(f"autoMRE: Parsing {file_path}")

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
        prog='automre',
        description='Automated Bug Reproduction Minimization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reduce a Python file
  automre reduce bug.py -o bug.min.py

  # Reduce with custom command
  automre reduce test_bug.py -c "pytest test_bug.py -v"

  # Reduce a whole project against a failing test
  automre reduce-project ./my_project -c "pytest tests/test_bug.py::test_foo -x -q"

  # Trace execution
  automre trace bug.py

  # Validate minimized code
  automre validate bug.min.py -r bug.py
        """
    )

    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {_version()}')

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
    reduce_parser.add_argument('-v', '--verbose', action='store_true',
                              help='Verbose output')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate minimized code')
    validate_parser.add_argument('file', help='Python file to validate')
    validate_parser.add_argument('-r', '--reference', help='Reference file for comparison')
    validate_parser.add_argument('-c', '--command', help='Command to execute')
    validate_parser.add_argument('-s', '--strategy', default='error_type',
                                choices=['exact', 'output_match',
                                         'error_type', 'error_message'],
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
    rp.add_argument('-s', '--strategy', default='output_match',
                    choices=['exact', 'output_match',
                             'error_type', 'error_message'],
                    help='Behavior-matching strategy. Default output_match '
                         'compares the full normalized output. error_type '
                         'only compares the exception class, which lets a '
                         'reduction swap in a DIFFERENT bug of the same type '
                         'and still be accepted — use it only when you '
                         'genuinely want that latitude.')
    rp.add_argument('--aggressive-inline', action='store_true',
                    help='Inline modules even when they have top-level '
                         'side effects; roll back if the inline breaks '
                         'the bug. Trades safety for tighter reductions.')
    rp.add_argument('--no-coverage-prune', action='store_true',
                    help='Disable Phase 4a coverage-based bulk pruning '
                         '(useful for ablation studies). Default is on.')
    rp.add_argument('--use-learned-oracle', action='store_true',
                    help='Consult the learned removability oracle: filter '
                         'Phase 4a prune candidates and skip hopeless '
                         'Phase 4b attempts. Requires the [oracle] extra '
                         'and a trained model; falls back to the heuristic '
                         'if either is missing. Note the Phase 4b skip can '
                         'leave removable code in the output if the model '
                         'is miscalibrated.')
    rp.add_argument('--python', default=None, metavar='PATH',
                    help='Interpreter to trace coverage with. Must match '
                         'the one your test command uses, or Phase 1 will '
                         'measure a different environment than the one '
                         'reductions are validated against. Defaults to '
                         'the interpreter running automre.')
    rp.add_argument('--oracle-model', default=None, metavar='PATH',
                    help='Path to the pickled oracle model. Defaults to '
                         'the one shipped in automre/src/ml/.')
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

    try:
        return commands[args.subcommand](args)
    except KeyboardInterrupt:
        # reduce-project says something more useful and handles this
        # itself; this is the backstop for the rest, so that stopping a
        # long run is an exit code rather than a stack trace.
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
