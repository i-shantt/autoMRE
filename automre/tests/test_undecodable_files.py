"""A source file that is not UTF-8 must not stop the reduction.

Python still honours a PEP 263 encoding declaration, so a `.py` file is
not obliged to be UTF-8, and real projects still contain ones that are
not. `pylint-dev/pylint` carries exactly one:
`tests/functional/i/implicit/implicit_str_concat_latin1.py`, one file in
2,189.

That single file crashed the whole reduction, and it did so before any
work happened at all — `_line_count` runs over every discovered file to
report the starting size, and it called `read_text()`. The run died with
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9` having spent
zero queries, which is how it presented: not as an encoding problem but
as a task that scored 0% reduction and fidelity 0.

The chosen behaviour is to leave such files alone and say so. Reducing
one would mean reading it, cutting spans out of it and writing it back
with bytes the reducer cannot decode preserved exactly; losing that
round-trip would corrupt a file while reporting success. Skipping costs
a little reduction and cannot corrupt anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "automre" / "src"))

from multi_file.dependency_analyzer import DependencyAnalyzer  # noqa: E402
from multi_file.multi_file_debugger import MultiFileDebugger  # noqa: E402

# A latin-1 file with a coding declaration, the way pylint's fixture is.
LATIN1 = b'# -*- coding: latin-1 -*-\nS = "caf\xe9 na\xefve"\n'

LIB = "def add(a, b):\n    return a + b\n"
TEST = "import mylib\n\n\ndef test_add():\n    assert mylib.add(1, 2) == 3\n"


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "tests").mkdir(parents=True)
    (proj / "mylib.py").write_text(LIB)
    (proj / "tests" / "test_mylib.py").write_text(TEST)
    (proj / "fixture_latin1.py").write_bytes(LATIN1)
    return proj


def test_discovery_separates_files_it_cannot_read(tmp_path):
    analysis = DependencyAnalyzer().analyze(
        _project(tmp_path),
        [sys.executable, "-m", "pytest", "tests/test_mylib.py", "-x", "-q"])

    names = {p.name for p in analysis.all_files}
    assert "mylib.py" in names
    assert "fixture_latin1.py" not in names
    assert [p.name for p in analysis.undecodable_files] == ["fixture_latin1.py"]


def test_a_latin1_file_does_not_crash_the_reduction(tmp_path):
    """The whole point: zero queries spent, reported as 0% and fidelity 0."""
    proj = _project(tmp_path)

    result = MultiFileDebugger().reduce_project(
        proj,
        [sys.executable, "-m", "pytest", "tests/test_mylib.py", "-x", "-q"])

    assert result.total_queries > 0


def test_the_file_it_would_not_touch_comes_back_byte_identical(tmp_path):
    proj = _project(tmp_path)

    MultiFileDebugger().reduce_project(
        proj,
        [sys.executable, "-m", "pytest", "tests/test_mylib.py", "-x", "-q"])

    assert (proj / "fixture_latin1.py").read_bytes() == LATIN1


def test_a_symlink_out_of_the_project_is_refused(tmp_path):
    """The reducer's remit is the directory it was given.

    `resolve()` follows symlinks, so a link pointing out of the tree put
    somebody else's file into the candidate set. Nothing outside was ever
    damaged, but only because `relative_to` happened to raise ValueError
    several phases later and kill the run — a crash standing in for a
    boundary. The boundary is now the boundary.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    precious = outside / "precious.py"
    precious.write_text("IMPORTANT = 'never touched'\n")
    before = precious.read_bytes()

    proj = _project(tmp_path)
    (proj / "escaped.py").symlink_to(precious)

    result = MultiFileDebugger().reduce_project(
        proj,
        [sys.executable, "-m", "pytest", "tests/test_mylib.py", "-x", "-q"])

    assert [p.name for p in result.outside_project_files] == ["precious.py"]
    assert precious.exists()
    assert precious.read_bytes() == before
    assert not any(p == precious.resolve() for p in result.unreachable_deleted)


def test_line_counting_survives_bytes_it_cannot_decode(tmp_path):
    """Counting lines is the first thing done to a tree, and it crashed.

    errors='replace' only ever substitutes non-ASCII bytes, so newlines
    are untouched and the count stays exact.
    """
    path = tmp_path / "x.py"
    path.write_bytes(LATIN1)

    assert MultiFileDebugger()._line_count(path) == 2
