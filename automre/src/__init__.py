"""
autoMRE: Automated Bug Reproduction Minimization

A Python tool for automatically minimizing bug-triggering code while
preserving reproduction capability using hybrid delta debugging.

Based on research in:
- Delta Debugging (Zeller & Hildebrandt, 2002)
- Hierarchical Delta Debugging (Misherghi & Su, 2006)
- Execution-guided Reduction (HDD-E)
"""

__version__ = "0.1.0"
__author__ = "autoMRE Research Project"

from .parser import PythonParser, CodeUnit
from .tracer import ExecutionTracer, ExecutionTrace
from .validator import Validator, ValidationResult, OriginalBehavior
from .reducer import (
    HybridDeltaDebugger,
    LineLevelDeltaDebugger,
    ReductionResult,
    ReductionStats,
    create_reducer
)

__all__ = [
    'PythonParser',
    'CodeUnit',
    'ExecutionTracer',
    'ExecutionTrace',
    'Validator',
    'ValidationResult',
    'OriginalBehavior',
    'HybridDeltaDebugger',
    'LineLevelDeltaDebugger',
    'ReductionResult',
    'ReductionStats',
    'create_reducer',
]
