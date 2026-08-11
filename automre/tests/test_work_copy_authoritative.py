"""The benchmark must refuse to score a tree it is not actually reducing.

This is the same class of bug that has now bitten the project three
times: the reproduction command runs against a copy of the package other
than the one being edited, so every deletion validates, reduction looks
spectacular, and execution fidelity is 1 while nothing was measured.

Asking where imports resolve *before* reduction is not sufficient, and a
real run proved it: a benchmark venv built with `--system-site-packages`
on a hosted runtime resolved to the work copy at baseline and fell
through to the host's preinstalled copy the moment the reducer deleted
the work copy's files. It scored 6/6 fidelity having deleted flask down
to its test file in 56 queries.

The positive control inside `provision.gate` is the answer: remove the
package and require the test to stop passing. These tests pin both
halves — it must not fire on a healthy tree, and it must fire on the
shadowed one.

Slow by nature: it builds real venvs and installs a real package,
because the failure lives in packaging behavior and cannot be mocked.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "automre" / "src"))

from provision import check, top_level_packages  # noqa: E402

CMD_TAIL = ["-m", "pytest", "tests/test_demo.py::test_answer", "-x", "-q",
            "--no-header"]


def _write_project(dest: Path) -> Path:
    """A minimal package + its own test, in a src/ layout.

    The src/ layout is load-bearing, not incidental. setuptools gives a
    flat layout a strict import finder that hard-fails once the package
    is gone, so the fall-through cannot happen and the scenario proves
    nothing. requests and flask both use src/, where an editable install
    is a plain sys.path entry: delete the package and the entry stays,
    pointing at nothing, and the import continues down sys.path.
    """
    proj = dest / "demoproj"
    (proj / "src" / "demopkg").mkdir(parents=True)
    (proj / "tests").mkdir()
    (proj / "src" / "demopkg" / "__init__.py").write_text(
        "def answer():\n    return 42\n")
    (proj / "tests" / "test_demo.py").write_text(textwrap.dedent("""
        from demopkg import answer

        def test_answer():
            assert answer() == 42
    """))
    (proj / "pyproject.toml").write_text(textwrap.dedent("""
        [build-system]
        requires = ["setuptools"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "demoproj"
        version = "0.1"

        [tool.setuptools.packages.find]
        where = ["src"]
    """))
    return proj


def _build_case(root: Path, pristine: Path, shadow_copy: bool) -> tuple:
    """Set up one world; returns (work_dir, cmd, baseline_rc).

    `shadow_copy` puts a second complete copy of the package on a
    sys.path entry *after* the work copy's. The ordering is the entire
    mechanism: earlier would shadow from the start and the old guard
    would catch it; later lies dormant until the reducer starts
    deleting. A .pth file reproduces that ordering faithfully, since
    .pth entries are appended after what is already on sys.path — which
    is where a host's site-packages sits relative to an editable
    install.
    """
    venv = root / "bench_venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    py = venv / "bin" / "python"
    subprocess.run([str(py), "-m", "pip", "install", "--quiet",
                    "pytest==9.0.0"], check=True)

    work = root / "work" / "demoproj"
    work.parent.mkdir()
    shutil.copytree(pristine, work)
    subprocess.run([str(py), "-m", "pip", "install", "--quiet", "-e",
                    str(work), "--no-deps"], check=True)

    if shadow_copy:
        site = next(venv.glob("lib/python*/site-packages"))
        host_dir = root / "host_site"
        shutil.copytree(pristine / "src", host_dir)
        (site / "zz_hostcopy.pth").write_text(str(host_dir) + "\n")

    return work, [str(py)] + CMD_TAIL


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    return _write_project(tmp_path_factory.mktemp("pristine"))


def test_control_passes_when_work_copy_is_under_test(pristine, tmp_path):
    work, cmd = _build_case(tmp_path, pristine, shadow_copy=False)

    verdict = check(work, cmd, timeout=120, require_pass=True)

    assert verdict.ok, verdict.reason


def test_control_fires_when_a_host_copy_shadows_the_work_copy(pristine,
                                                              tmp_path):
    work, cmd = _build_case(tmp_path, pristine, shadow_copy=True)

    # The reason a static check is not enough: at baseline, before any
    # deletion, the package resolves to the work copy exactly as it
    # should. Nothing is visibly wrong until the reducer starts cutting.
    probe = subprocess.run(
        [cmd[0], "-c", "import demopkg, os; print(os.path.realpath("
                       "demopkg.__file__))"],
        cwd=work, capture_output=True, text=True)
    assert probe.stdout.strip().startswith(str(work.resolve()))

    verdict = check(work, cmd, timeout=120, require_pass=True)

    assert not verdict.ok, (
        "the control did not fire on a shadowed work copy — this is the "
        "exact setup that produced 6/6 fidelity on nothing")
    assert "vacuous" in verdict.reason


def test_control_restores_the_tree_it_sabotages(pristine, tmp_path):
    """A guard that damages the tree would be worse than no guard."""
    work, cmd = _build_case(tmp_path, pristine, shadow_copy=False)
    before = {p.relative_to(work): p.read_bytes()
              for p in sorted(work.rglob("*.py"))}

    check(work, cmd, timeout=120, require_pass=True)

    after = {p.relative_to(work): p.read_bytes()
             for p in sorted(work.rglob("*.py"))}
    assert before == after
    assert top_level_packages(work) == ["demopkg"]
