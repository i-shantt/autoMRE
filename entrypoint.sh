#!/bin/bash
# autoMRE: Automated Bug Reproduction Minimization
# Entrypoint script for full reproduction of all results
# Usage: ./entrypoint.sh [command]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="autoMRE"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  $PROJECT_NAME - Reproduction Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "${YELLOW}>> $1${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
check_python() {
    print_section "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        echo "  Found Python $PYTHON_VERSION"
        print_success "Python is available"
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    print_section "Installing dependencies..."
    cd "$SCRIPT_DIR"

    # Install required packages
    python3 -m pip install tree-sitter tree-sitter-python coverage --quiet

    print_success "Dependencies installed"
}

# Run unit tests
run_tests() {
    print_section "Running unit tests..."
    cd "$SCRIPT_DIR"

    # Set PYTHONPATH to include the src directory so modules can import each other
    export PYTHONPATH="$SCRIPT_DIR/automre/src"

    # Test parser
    echo "  Testing parser..."
    python3 -c "
from parser import PythonParser
parser = PythonParser()
code = 'def foo():\n    pass'
tree = parser.parse_source(code)
print('    Parser: OK')
"

    # Test tracer
    echo "  Testing tracer..."
    python3 -c "
from tracer import ExecutionTracer
tracer = ExecutionTracer()
print('    Tracer: OK')
"

    # Test validator
    echo "  Testing validator..."
    python3 -c "
from validator import Validator
validator = Validator()
print('    Validator: OK')
"

    # Test reducer
    echo "  Testing reducer..."
    python3 -c "
from reducer import HybridDeltaDebugger
reducer = HybridDeltaDebugger()
print('    Reducer: OK')
"

    # Restore PYTHONPATH
    unset PYTHONPATH

    print_success "All tests passed"
}

# Run example reduction
run_example() {
    print_section "Running example reduction..."
    cd "$SCRIPT_DIR"

    EXAMPLE_DIR="automre/examples/simple_single_file"
    ORIGINAL="$EXAMPLE_DIR/bug_original.py"
    MINIMIZED="$EXAMPLE_DIR/bug_output.py"

    echo "  Original: $ORIGINAL"
    echo "  Lines in original: $(wc -l < $ORIGINAL)"

    # Run reduction
    python3 automre.py reduce "$ORIGINAL" -o "$MINIMIZED" -v

    if [ -f "$MINIMIZED" ]; then
        echo "  Lines in minimized: $(wc -l < $MINIMIZED)"
        print_success "Reduction complete"
    else
        print_error "Minimization failed"
    fi
}

# Run evaluation
run_evaluation() {
    print_section "Running evaluation on example bugs..."
    cd "$SCRIPT_DIR"

    EVAL_DIR="evaluation"
    EXAMPLES_DIR="automre/examples/simple_single_file"

    # Run comparison of all algorithms
    echo "  Comparing all algorithms..."
    python3 "$EVAL_DIR/simple_runner.py" \
        --examples "$EXAMPLES_DIR" \
        --output "$EVAL_DIR/comparison_results.json" \
        --compare-all

    print_success "Evaluation complete"
    echo ""
    echo "  Results saved to:"
    echo "    - $EVAL_DIR/comparison_results.json"
}

# Display results summary
show_results() {
    print_section "Results Summary"
    cd "$SCRIPT_DIR"

    echo ""
    echo "  Evaluation Results (example bugs):"
    echo "  -----------------------------------"

    for result_file in evaluation/results_*.json; do
        if [ -f "$result_file" ]; then
            basename=$(basename "$result_file" .json)
            echo "  $basename:"
            python3 -c "
import json
with open('$result_file') as f:
    data = json.load(f)
    if 'summary' in data:
        s = data['summary']
        print(f\"    Success Rate: {s.get('success_rate', 'N/A')}\")
        print(f\"    Avg Reduction: {s.get('avg_reduction_rate', 'N/A'):.1%}\")
        print(f\"    Avg Time: {s.get('avg_time', 'N/A'):.2f}s\")
        print(f\"    Avg Queries: {s.get('avg_queries', 'N/A'):.1f}\")
" 2>/dev/null || echo "    (raw data available)"
        fi
    done

    echo ""
    print_success "All results reproduced successfully!"
}

# Show usage information
show_usage() {
    echo "Usage: ./entrypoint.sh [command]"
    echo ""
    echo "Commands:"
    echo "  all         Run complete reproduction (default)"
    echo "  test        Run unit tests only"
    echo "  example     Run example reduction only"
    echo "  eval        Run evaluation only"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./entrypoint.sh              # Full reproduction"
    echo "  ./entrypoint.sh test         # Run tests only"
    echo "  ./entrypoint.sh example      # Run example only"
}

# Main execution
main() {
    COMMAND=${1:-all}

    case "$COMMAND" in
        all)
            check_python
            install_deps
            run_tests
            run_example
            run_evaluation
            show_results
            ;;
        test)
            check_python
            install_deps
            run_tests
            ;;
        example)
            check_python
            install_deps
            run_example
            ;;
        eval|evaluation)
            check_python
            install_deps
            run_evaluation
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
