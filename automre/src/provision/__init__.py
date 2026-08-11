"""Turn a project directory into something autoMRE can be pointed at.

Three steps, and a reduction needs all three before it can start:

    provision()   an environment where the project's tests can run
    discover()    which tests it offers
    check()       whether the one you named is worth reducing against

Until now these lived in two places that did not know about each other.
`web/worker/jobs.py` grew a virtualenv-per-job provisioner and an
AST-based test finder because the web app had to accept arbitrary
uploads; `evaluation/gistify_runner.py` grew the readiness checks because
the benchmark kept measuring nothing. Neither could be used by the other,
so the benchmark's task list stayed hand-written — ten tasks over three
repositories, and two cross-validation folds for the learned oracle.

The structure follows SWE-Hub (arXiv 2603.00575), which names the same
three steps Env Agent, Test Agent and verification gate. Its code was
never released; what is borrowed is the observation that these belong
together, which is the part that was missing here.
"""

from __future__ import annotations

from .discovery import discover, node_id_exists
from .environment import (
    DEFAULT_PYTEST_PIN,
    EnvSpec,
    ProvisionError,
    Step,
    discard,
    provision,
)
from .gate import (
    Readiness,
    baseline_health,
    check,
    purge_bytecode,
    top_level_packages,
)

__all__ = [
    "DEFAULT_PYTEST_PIN",
    "EnvSpec",
    "ProvisionError",
    "Readiness",
    "Step",
    "baseline_health",
    "check",
    "discard",
    "discover",
    "node_id_exists",
    "provision",
    "purge_bytecode",
    "top_level_packages",
]
