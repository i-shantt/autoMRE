"""Speculative parallelism must change runtime and nothing else.

The accumulating pass is sequential by nature: whether candidate *i+1* is
offered at all depends on whether *i* was accepted, because an accepted
span changes both the text the next candidate is cut from and which later
spans now overlap a hole. Asking in parallel means guessing those answers
ahead of time, and a wrong guess produces a plan built on a source that
never existed.

So the claim that matters is not "it is faster" — it is that the accepted
set, the query count and the bookkeeping come out **identical** to the
sequential walk for every possible pattern of oracle answers. These tests
check that against the real `_sequential_pass`, over randomised oracles,
rather than against a hand-picked example that the implementation happens
to survive.

They deliberately use a fake oracle rather than real subprocesses: the
scheduler is what is under test, and a fake makes every answer pattern
reachable, including the adversarial ones a real project would not
produce in a hundred runs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "autorepro_min" / "src"))

from reducer import HybridDeltaDebugger, ReductionStats  # noqa: E402
from validator import ValidationResult, Validator  # noqa: E402


SOURCE = "\n".join(f"S{i} = {i}" for i in range(24)) + "\n"


class _RuleValidator(Validator):
    """Sequential oracle: answers from `rule`, counts subprocesses."""

    def __init__(self, rule):
        super().__init__()
        self.set_original_behavior("x", 0)
        self.rule = rule
        self.calls = 0

    def validate(self, source_code, command=None, cwd=None):
        self.calls += 1
        ok = self.rule(source_code)
        return ValidationResult(is_valid=ok, output="", return_code=0,
                                matches_original=ok, error_type_match=ok,
                                error_message_similarity=1.0 if ok else 0.0)


class _FakeParallelOracle:
    """Stands in for ParallelOracle with the same answers, in batches."""

    def __init__(self, rule, jobs):
        self.rule = rule
        self.jobs = jobs
        self.enabled = True
        self.calls = 0
        self.widest_batch = 0

    def relative_path(self, path):
        return Path("m.py")

    def ask(self, candidates):
        self.calls += len(candidates)
        self.widest_batch = max(self.widest_batch, len(candidates))
        return [self.rule(c.source) for c in candidates]


def _stats():
    return ReductionStats(original_size=0, final_size=0, iterations=0,
                          queries=0, time_seconds=0.0,
                          successful_removals=0, failed_removals=0)


def _run_both(rule, jobs, source=SOURCE):
    """The same pass, sequential and speculative. Returns both outcomes."""
    seq_debugger = HybridDeltaDebugger(validator=_RuleValidator(rule))
    seq_debugger._file_path = Path("m.py")
    seq_debugger._executed = set()
    candidates = seq_debugger._candidates(source, trace=None)
    seq_stats = _stats()
    seq_spans = seq_debugger._sequential_pass(
        source, candidates, set(), None, None, None, seq_stats)

    fake = _FakeParallelOracle(rule, jobs)
    par_debugger = HybridDeltaDebugger(validator=_RuleValidator(rule),
                                       speculator=fake)
    par_debugger._file_path = Path("m.py")
    par_debugger._executed = set()
    par_candidates = par_debugger._candidates(source, trace=None)
    par_stats = _stats()
    par_spans = par_debugger._speculative_pass(
        source, par_candidates, set(), None, None, None, par_stats)

    return (seq_spans, seq_stats, seq_debugger.validator.calls), \
           (par_spans, par_stats, fake)


@pytest.mark.parametrize("jobs", [2, 4, 8])
def test_identical_when_everything_is_accepted(jobs):
    """The best case for an optimistic chain: nothing is ever discarded."""
    (seq_spans, seq_stats, _), (par_spans, par_stats, fake) = _run_both(
        lambda src: True, jobs)

    assert par_spans == seq_spans
    assert par_stats.queries == seq_stats.queries
    assert par_stats.successful_removals == seq_stats.successful_removals
    assert par_stats.speculative_discarded == 0
    assert fake.widest_batch > 1, "nothing was actually batched"


@pytest.mark.parametrize("jobs", [2, 4, 8])
def test_identical_when_everything_is_rejected(jobs):
    """The worst case: every batch is discarded after its first answer."""
    (seq_spans, seq_stats, _), (par_spans, par_stats, _) = _run_both(
        lambda src: False, jobs)

    assert par_spans == seq_spans == []
    assert par_stats.queries == seq_stats.queries
    assert par_stats.failed_removals == seq_stats.failed_removals


@pytest.mark.parametrize("seed", range(25))
def test_identical_for_arbitrary_oracles(seed):
    """Randomised answers, including patterns a real project never makes.

    The oracle answers on a hash of the candidate text, so it is a pure
    function of the source — the same property a real oracle has, and the
    one the whole scheme rests on — while still being arbitrary.
    """
    rng = random.Random(seed)
    threshold = rng.random()

    def rule(src):
        return (hash((seed, src)) % 1000) / 1000.0 < threshold

    jobs = rng.choice([2, 3, 4, 8])
    (seq_spans, seq_stats, _), (par_spans, par_stats, _) = _run_both(rule, jobs)

    assert par_spans == seq_spans, f"seed={seed} jobs={jobs}"
    assert par_stats.queries == seq_stats.queries
    assert par_stats.successful_removals == seq_stats.successful_removals
    assert par_stats.failed_removals == seq_stats.failed_removals
    assert par_stats.syntax_rejected == seq_stats.syntax_rejected
    assert par_stats.oracle_skipped == seq_stats.oracle_skipped


def test_discarded_work_is_reported_not_hidden():
    """`queries` stays comparable with a sequential run; waste is separate."""
    # Reject everything, so each batch is thrown away after its first
    # answer and the discard count is the rest of the batch.
    (_, seq_stats, _), (_, par_stats, _) = _run_both(lambda s: False, jobs=4)

    assert par_stats.queries == seq_stats.queries
    assert par_stats.speculative_discarded > 0, (
        "rejecting everything must discard the tail of every batch")


def test_oracle_skips_are_counted_once():
    """A skipped unit must not be re-counted when a batch is rescanned."""
    debugger = HybridDeltaDebugger(
        validator=_RuleValidator(lambda s: False),
        speculator=_FakeParallelOracle(lambda s: False, jobs=4))
    debugger._file_path = Path("m.py")
    debugger._executed = set()
    candidates = debugger._candidates(SOURCE, trace=None)
    # Mark every other unit hopeless.
    hopeless = {id(u) for i, u in enumerate(candidates) if i % 2 == 0}

    par_stats = _stats()
    debugger._speculative_pass(SOURCE, candidates, hopeless, None, None,
                               None, par_stats)

    seq = HybridDeltaDebugger(validator=_RuleValidator(lambda s: False))
    seq._file_path = Path("m.py")
    seq._executed = set()
    seq_stats = _stats()
    seq._sequential_pass(SOURCE, candidates, hopeless, None, None, None,
                         seq_stats)

    assert par_stats.oracle_skipped == seq_stats.oracle_skipped


# ------------------------------------------------------- real subprocesses

def _tiny_project(root: Path) -> list:
    """A project whose test really does exercise the library."""
    (root / "mylib").mkdir()
    (root / "mylib" / "__init__.py").write_text("from .core import keep\n")
    (root / "mylib" / "core.py").write_text(
        '"""Doc — with an em dash, so offsets matter here too."""\n'
        "UNUSED_A = 1\n"
        "UNUSED_B = 2\n"
        "UNUSED_C = 3\n"
        "\n"
        "def dead_one():\n"
        "    return 'nobody calls me'\n"
        "\n"
        "def dead_two():\n"
        "    return 'nor me'\n"
        "\n"
        "def keep():\n"
        "    return 42\n")
    (root / "test_mylib.py").write_text(
        "from mylib import keep\n"
        "\n"
        "def test_keep():\n"
        "    assert keep() == 42\n")
    return [sys.executable, "-m", "pytest", "test_mylib.py", "-x", "-q",
            "--no-header"]


def test_parallel_run_matches_sequential_run(tmp_path):
    """End to end, with real worker copies and real subprocesses.

    The fake oracle above cannot catch the thing most likely to go wrong
    in the real pool: a worker whose `import mylib` resolves somewhere
    other than its own copy, which would make every answer describe code
    nobody is editing. Only actually running it can.
    """
    from multi_file.multi_file_debugger import MultiFileDebugger

    outcomes = {}
    for jobs in (1, 4):
        root = tmp_path / f"run{jobs}"
        root.mkdir()
        command = _tiny_project(root)
        summary = MultiFileDebugger(timeout=60, jobs=jobs).reduce_project(
            root, command)
        outcomes[jobs] = {
            "source": (root / "mylib" / "core.py").read_text(),
            "queries": summary.total_queries,
            "lines": summary.final_line_count,
            "discarded": summary.speculative_discarded,
        }

    assert outcomes[4]["source"] == outcomes[1]["source"], (
        "parallel reduction produced different code")
    assert outcomes[4]["lines"] == outcomes[1]["lines"]
    assert outcomes[4]["queries"] == outcomes[1]["queries"], (
        "query counts must stay comparable between the two modes")
    # The reduction must be real, not a no-op that trivially agrees.
    assert "dead_one" not in outcomes[1]["source"]
    assert "keep" in outcomes[1]["source"]


def test_worker_refusing_verification_falls_back_to_sequential(tmp_path):
    """A pool that cannot vouch for itself must not be used at all."""
    from multi_file.parallel_oracle import ParallelOracle

    root = tmp_path / "proj"
    root.mkdir()
    command = _tiny_project(root)

    class _NeverMatches:
        def _matches_original(self, output, code):
            return False

    pool = ParallelOracle(root, command, oracle=_NeverMatches(),
                          timeout=60, jobs=2)
    try:
        assert pool.start() is False
        assert pool.enabled is False
        assert "does not reproduce" in (pool.reason or "")
    finally:
        pool.close()


def test_the_pool_really_starts_on_that_fixture(tmp_path):
    """Guard the guard.

    `test_parallel_run_matches_sequential_run` compares jobs=4 against
    jobs=1, and would pass just as happily if the pool had quietly
    refused and run sequentially both times. Pin that it does start, and
    that its workers resolve imports to their own copies.
    """
    from multi_file.multi_file_validator import MultiFileValidator
    from multi_file.parallel_oracle import Candidate, ParallelOracle

    root = tmp_path / "proj"
    root.mkdir()
    command = _tiny_project(root)

    validator = MultiFileValidator(command, root, timeout=60)
    validator.capture_original()

    pool = ParallelOracle(root, command, oracle=validator._oracle,
                          timeout=60, jobs=3)
    try:
        assert pool.start() is True, f"pool refused: {pool.reason}"
        assert len(pool._roots) == 3
        for worker in pool._roots:
            assert (worker / "mylib" / "core.py").exists()

        # A harmless edit must still reproduce; deleting keep() must not.
        good = (root / "mylib" / "core.py").read_text().replace(
            "UNUSED_A = 1", "UNUSED_A = 99")
        bad = (root / "mylib" / "core.py").read_text().replace(
            "    return 42", "    return -1")
        verdicts = pool.ask([
            Candidate(Path("mylib/core.py"), good),
            Candidate(Path("mylib/core.py"), bad),
        ])
        assert verdicts == [True, False], verdicts

        # And the real tree must be untouched by anything the pool did.
        assert "UNUSED_A = 1" in (root / "mylib" / "core.py").read_text()
    finally:
        pool.close()
    assert not (root.parent / f".{root.name}_workers").exists()
