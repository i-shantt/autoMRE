"""The oracle must judge the code on disk, not a cached compile of it.

CPython treats a cached .pyc as fresh when the source's (mtime, size) is
unchanged, and the mtime it records has one-second resolution. The reducer
rewrites the same file many times per second, so two candidates for one
file that happen to share a byte length are indistinguishable to that
check — Python imports the earlier candidate's bytecode and the validator
reports on code that is not on disk.

Equal-length candidates are not a corner case. Delta debugging routinely
swaps one statement for another of the same width, and every deletion that
happens to remove as many bytes as a previous one produces a collision.

The symptom is a verdict that depends on the clock: the same reduction run
twice takes different paths and spends a different number of queries,
because whether two writes land in the same tick is timing, not logic. The
danger is worse than the noise — a candidate that genuinely breaks the bug
can be accepted because the stale bytecode still holds the working code.

This test pins the property directly: rewrite a module with a same-length
body that changes behavior, and require the validator to notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from multi_file.multi_file_debugger import MultiFileDebugger  # noqa: E402
from multi_file.multi_file_validator import MultiFileValidator  # noqa: E402


# Two library versions of byte-identical length whose behavior differs.
LIB_OK = 'def check():\n    return "AAA"\n'
LIB_BAD = 'def check():\n    return "BBB"\n'

MAIN = 'from lib import check\n\nprint(check())\n'


def test_validator_sees_same_length_rewrite(tmp_path: Path) -> None:
    (tmp_path / "lib.py").write_text(LIB_OK)
    (tmp_path / "main.py").write_text(MAIN)

    validator = MultiFileValidator([sys.executable, "main.py"], tmp_path)
    original = validator.capture_original()
    assert "AAA" in original.output, original.output
    assert validator.validate(), "baseline should reproduce"

    # Same byte length, different behavior, written immediately after — the
    # exact shape that lets a stale .pyc through.
    assert len(LIB_BAD) == len(LIB_OK)
    (tmp_path / "lib.py").write_text(LIB_BAD)

    assert not validator.validate(), (
        "validator accepted a changed library as unchanged — it is judging "
        "cached bytecode, not the source on disk"
    )


def test_repeated_same_length_rewrites_all_observed(tmp_path: Path) -> None:
    """Back-to-back rewrites within one clock tick must each be seen."""
    (tmp_path / "lib.py").write_text(LIB_OK)
    (tmp_path / "main.py").write_text(MAIN)

    validator = MultiFileValidator([sys.executable, "main.py"], tmp_path)
    validator.capture_original()

    # Alternate as fast as the loop allows; every flip must be detected.
    for i in range(4):
        expect_match = i % 2 == 1
        (tmp_path / "lib.py").write_text(LIB_OK if expect_match else LIB_BAD)
        assert validator.validate() is expect_match, (
            f"iteration {i}: validator did not observe the rewrite"
        )


def test_reduction_discards_inherited_bytecode(tmp_path: Path) -> None:
    """Bytecode left by whoever ran the project before us must not survive.

    Suppressing writes keeps the reduction from caching its own candidates,
    but says nothing about a `__pycache__` that already existed — a caller
    verifying the command works, a prior test run, an editor. Those entries
    are reused on any source whose (mtime, size) still matches, so the
    reducer has to clear them itself rather than assume a clean tree.

    Driven through `reduce_project` rather than the helper, because the
    property at stake is that reduction *starts* clean. Testing the helper
    alone would pass even if nothing ever called it. Since the reduction's
    own subprocesses no longer write bytecode, any `__pycache__` left at
    the end is necessarily the inherited one, still sitting there.
    """
    import subprocess

    proj = tmp_path / "inherited"
    proj.mkdir()
    (proj / "mylib.py").write_text("def add(a, b):\n    return a + b\n")
    (proj / "test_mylib.py").write_text(
        "import mylib\n\n\ndef test_add():\n    assert mylib.add(2, 3) == 5\n")

    target = "test_mylib.py::test_add"
    cmd = [sys.executable, "-m", "pytest", target, "-q", "--no-header"]

    # Stand in for anything that ran the project first: a normal run with
    # bytecode caching left on, which is what populates __pycache__.
    assert subprocess.run(cmd, cwd=proj, capture_output=True,
                          text=True, timeout=120).returncode == 0
    assert list(proj.rglob("__pycache__")), (
        "precondition failed: the setup run wrote no bytecode, so this test "
        "would pass without exercising anything"
    )

    debugger = MultiFileDebugger(verbose=False, timeout=60,
                                 python=sys.executable)
    debugger.reduce_project(proj, cmd)

    assert not list(proj.rglob("__pycache__")), (
        "reduction ran against a tree still holding bytecode it inherited "
        "from an earlier run; a same-length rewrite could be judged against "
        "that stale compile"
    )
