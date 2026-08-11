#!/bin/bash
# autoMRE: one command to see the tool work on this machine.
#
# Runs the test suite and reduces a bundled example project, which
# together take about half a minute. It does not reproduce the benchmark
# in README.md — that is ten tasks against three cloned repositories and
# takes hours — it prints the command for it instead.
#
# Usage: ./entrypoint.sh [test|example|all|help]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_section() { echo -e "${YELLOW}>> $1${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }

check_env() {
    print_section "Checking the environment..."
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not installed"
        exit 1
    fi
    echo "  Python $(python3 --version 2>&1 | awk '{print $2}')"

    # Deliberately does not install anything. A script that quietly
    # writes to the interpreter it happens to find is a worse trade than
    # one line of instruction.
    local missing=""
    python3 -c "import tree_sitter, tree_sitter_python, coverage" 2> /dev/null \
        || missing="autoMRE's dependencies"
    # Checked separately because it is the one that used to be missing on
    # a clean install, and failing here with the fix beats failing four
    # lines later with "No module named pytest".
    python3 -c "import pytest" 2> /dev/null \
        || missing="${missing:+$missing and }pytest"
    if [ -n "$missing" ]; then
        print_error "$missing not installed. Install the package first:"
        echo ""
        echo "    python3 -m venv .venv && source .venv/bin/activate"
        echo "    pip install -e \".[dev]\""
        echo ""
        exit 1
    fi
    print_success "Dependencies present"
}

run_tests() {
    print_section "Running the test suite..."
    cd "$SCRIPT_DIR"
    python3 -m pytest automre/tests/ -q
    print_success "Tests passed"
}

run_example() {
    print_section "Reducing a bundled example project..."
    cd "$SCRIPT_DIR"

    local example="$SCRIPT_DIR/automre/examples/multi_file/project1_cross_file_type_error"
    local out
    out="$(mktemp -d)"
    trap 'rm -rf "$out"' RETURN

    echo "  Project: automre/examples/multi_file/project1_cross_file_type_error"
    echo "  Bug reproduces with: python3 main.py"
    echo ""

    # --output into a temp directory: the example is checked in, and a
    # demonstration should not leave a second copy of it in the tree.
    # Run through the shim rather than the `automre` console script,
    # which is only on PATH if the active environment installed it.
    python3 "$SCRIPT_DIR/automre.py" reduce-project "$example" \
        -c "python3 main.py" --output "$out/reduced" --force

    echo ""
    echo "  Reduced project:"
    for f in "$out/reduced"/*.py; do
        echo "    --- $(basename "$f") ---"
        sed 's/^/    /' "$f"
    done
    print_success "Example reduced"
}

show_benchmark() {
    print_section "Reproducing the published numbers"
    cat <<'EOF'
  The results table in README.md is ten pytest tasks across requests,
  flask and tomlkit. It clones those repositories, provisions a pinned
  virtualenv, and took 3.3 hours on an M4 MacBook Air:

      python3 evaluation/gistify_runner.py

  One repository's worth (requests was 15 minutes of that):

      python3 evaluation/gistify_runner.py --only requests

  Results land in evaluation/results_gistify_*.json. Note that several
  older files in that directory are withdrawn measurements, kept for the
  record and listed as void under "Withdrawn results" in README.md — do
  not read numbers out of them.
EOF
}

show_usage() {
    echo "Usage: ./entrypoint.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all         Tests, then the example reduction (default)"
    echo "  test        Test suite only"
    echo "  example     Example reduction only"
    echo "  benchmark   Print how to reproduce the published numbers"
    echo "  help        This message"
}

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  autoMRE${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    case "${1:-all}" in
        all)
            check_env
            run_tests
            run_example
            echo ""
            show_benchmark
            ;;
        test)      check_env; run_tests ;;
        example)   check_env; run_example ;;
        benchmark) show_benchmark ;;
        help|--help|-h) show_usage ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
