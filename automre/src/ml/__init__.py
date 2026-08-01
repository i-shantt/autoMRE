"""
autoMRE: Learned Removability Oracle

A small gradient-boosted classifier that predicts whether a given code
unit can be safely removed without breaking the target test. Used as a
prior for two decision points inside MultiFileDebugger:

  * Phase 4a — coverage-based bulk prune: filter the pruner's candidate
    list to units the oracle predicts safe with p > 0.9. The intent is
    to prevent the ~12 whole-file rollbacks/benchmark caused by fixture
    / decorator / parametrize machinery that coverage.py can't reason
    about.
  * Phase 4b — per-file HDD-E: skip removal attempts where the oracle
    predicts p(safe) < 0.1. This trades a small risk of leaving
    removable code in the output against real query-count savings on
    definitely-not-removable units.

The oracle is optional (`[oracle]` extra). Nothing in the core reducer
requires sklearn or numpy.
"""

from __future__ import annotations
