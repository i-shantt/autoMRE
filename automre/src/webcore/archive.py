"""Getting an uploaded project out of a zip without trusting the zip.

Reduction executes the code in the archive, so these are the first line of
defence and not merely resource limits — see web/README.md, which does not
pretend they are sufficient on their own.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from .errors import JobError

# Upload guards.
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
MAX_FILES = 5000

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv"}


def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack a zip, refusing anything that escapes `dest` or is huge.

    `zipfile.extractall` will happily write to `../../etc/whatever` if the
    archive says so. Every member is therefore resolved against the
    destination and rejected if it lands outside, and the uncompressed
    total is capped so a zip bomb cannot fill the disk.
    """
    total = 0
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if len(members) > MAX_FILES:
            raise JobError(f"archive holds {len(members)} entries; "
                           f"the limit is {MAX_FILES}")
        for info in members:
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise JobError("archive expands past "
                               f"{MAX_UNCOMPRESSED_BYTES // (1024*1024)} MB")
            target = (dest / info.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise JobError(f"archive entry escapes the extraction "
                               f"directory: {info.filename}")
        zf.extractall(dest)


def project_root(extracted: Path) -> Path:
    """The directory the project actually starts at.

    Zipping a folder usually produces a single top-level directory, and
    running the reproduction command one level above it would fail for a
    reason that has nothing to do with the user's project.
    """
    entries = [p for p in extracted.iterdir() if not p.name.startswith("__")]
    visible = [p for p in entries if not p.name.startswith(".")]
    if len(visible) == 1 and visible[0].is_dir():
        return visible[0]
    return extracted


def count_python(root: Path):
    files = [p for p in root.rglob("*.py")
             if not any(part in _SKIP_DIRS for part in p.parts)]
    lines = 0
    for f in files:
        try:
            lines += len(f.read_text().splitlines())
        except (OSError, UnicodeDecodeError):
            continue
    return len(files), lines
