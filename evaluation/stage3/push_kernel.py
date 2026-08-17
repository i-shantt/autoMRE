#!/usr/bin/env python3
"""Turn kaggle_cells.md into a notebook and push it to Kaggle.

The cells document is the source of truth — it is reviewable in the
repository and diffs like code, which a .ipynb does not. This script is
the boring half: read the fenced python blocks out of it, wrap them in
notebook JSON, write the metadata Kaggle wants, and hand the directory
to `kaggle kernels push`.

It exists because the MCP server that used to do this was launched from
a directory that no longer exists, so the notebook had no way to reach
Kaggle at all. A dependency on one machine's local checkout of another
project is not one this experiment should have.

    python3 evaluation/stage3/push_kernel.py --slug automre-stage3 \
        --include "Cell 0,Cell 2,Cell 5"     # the capacity check
    python3 evaluation/stage3/push_kernel.py --slug automre-stage3

`--include` selects cells by heading substring; without it every cell
runs, always in document order, because later cells use names the
earlier ones bind.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent

_HEADING = re.compile(r"^##\s+(.*)$")
_FENCE = re.compile(r"^```(\w*)\s*$")


def read_cells(path: Path) -> List[Tuple[str, str]]:
    """(heading, code) for every fenced python block, in document order.

    A heading with several code blocks under it yields several cells,
    all carrying that heading — which is what `--include` filters on.
    """
    cells: List[Tuple[str, str]] = []
    heading = "(preamble)"
    code: List[str] = []
    in_python = False

    for line in path.read_text().splitlines():
        fence = _FENCE.match(line)
        if fence:
            if in_python:
                cells.append((heading, "\n".join(code)))
                code, in_python = [], False
            elif fence.group(1) == "python":
                in_python = True
            continue
        if in_python:
            code.append(line)
        else:
            found = _HEADING.match(line)
            if found:
                heading = found.group(1).strip()

    if in_python:
        raise SystemExit(f"{path}: unclosed ``` fence under {heading!r}")
    return cells


def slugify(title: str) -> str:
    """Kaggle's own rule for turning a title into a URL slug."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-",
                                     title.lower())).strip("-")


def notebook(cells: List[Tuple[str, str]]) -> dict:
    return {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": (f"# {heading}\n{code}").splitlines(True)}
            for heading, code in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True,
                    help="kernel slug; qualified with your username")
    ap.add_argument("--cells", default=str(_HERE / "kaggle_cells.md"))
    ap.add_argument("--include", default="",
                    help="comma-separated heading substrings; empty = all")
    ap.add_argument("--title", default="")
    ap.add_argument("--build-dir", default="")
    ap.add_argument("--kaggle", default="kaggle",
                    help="path to the kaggle CLI")
    ap.add_argument("--no-gpu", action="store_true",
                    help="CPU only — for cells that do not need a GPU, so "
                         "they do not spend the weekly GPU quota")
    ap.add_argument("--accelerator", default="nvidiaTeslaT4",
                    help="Kaggle machine shape. The default is deliberate: "
                         "left unset, a session can be given a Tesla P100, "
                         "which is compute capability sm_60, and Kaggle's "
                         "own PyTorch is built for sm_70 and up — so every "
                         "kernel launch fails with "
                         "cudaErrorNoKernelImageForDevice after the model "
                         "has finished loading. A P100's 16 GB also has to "
                         "offload part of a 7B fp16 model to the CPU. Pass "
                         "an empty string to take whatever is offered.")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the directory and stop, without pushing")
    args = ap.parse_args()

    cells = read_cells(Path(args.cells))
    if args.include:
        wanted = [w.strip() for w in args.include.split(",") if w.strip()]
        cells = [(h, c) for h, c in cells if any(w in h for w in wanted)]
        if not cells:
            raise SystemExit(f"--include {args.include!r} matched no cell")

    build = Path(args.build_dir) if args.build_dir else _HERE / "_kernel"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    name = args.slug.split("/")[-1]

    # Kaggle derives the kernel's URL from the *title*, not the id, and
    # only warns when the two disagree. Pushing "autoMRE stage 3
    # capacity check" against the id `automre-stage3-capacity` created
    # `automre-stage-3-capacity-check` instead, so the id could no
    # longer address the kernel — and a second push would have made a
    # second kernel and spent the GPU quota twice. Refuse instead.
    if args.title and slugify(args.title) != name:
        raise SystemExit(
            f"title {args.title!r} resolves to {slugify(args.title)!r}, "
            f"not {name!r}. Kaggle would push to that slug instead and "
            f"the id could not address it. Pick a title that slugifies "
            f"to the slug, or pass no title.")
    (build / f"{name}.ipynb").write_text(json.dumps(notebook(cells), indent=1))
    (build / "kernel-metadata.json").write_text(json.dumps({
        "id": args.slug if "/" in args.slug else f"ishantchintapatla/{args.slug}",
        "title": args.title or name,
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        # Private by default: a public kernel publishes the code and
        # every result under the account, and this one is an experiment
        # in progress, not a finding.
        "is_private": True,
        "enable_gpu": not args.no_gpu,
        "enable_internet": True,
        **({} if args.no_gpu or not args.accelerator
           else {"machine_shape": args.accelerator}),
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=1))

    for heading, _ in cells:
        print(f"  {heading}")
    print(f"{len(cells)} cell(s) -> {build}")

    if args.dry_run:
        return 0

    done = subprocess.run([args.kaggle, "kernels", "push", "-p", str(build)],
                          capture_output=True, text=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
