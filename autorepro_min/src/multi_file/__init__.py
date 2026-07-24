"""
AutoRepro-Min: Multi-File Reduction Subpackage

Implements MF-HDD-E (Multi-File Hierarchical Delta Debugging with
Execution Guidance): extends the single-file HDD-E algorithm to whole
project directories.

Public entry point: MultiFileDebugger.reduce_project(...)
"""

from __future__ import annotations

from .dependency_analyzer import DependencyAnalyzer, FileClass, ProjectAnalysis
from .import_inliner import ImportInliner, InlineResult
from .multi_file_debugger import MultiFileDebugger, MultiFileReductionResult
from .multi_file_validator import MultiFileValidator, ProjectFileValidator

__all__ = [
    "DependencyAnalyzer",
    "FileClass",
    "ProjectAnalysis",
    "ImportInliner",
    "InlineResult",
    "MultiFileDebugger",
    "MultiFileReductionResult",
    "MultiFileValidator",
    "ProjectFileValidator",
]
