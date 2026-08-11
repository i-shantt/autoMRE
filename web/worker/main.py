"""HTTP surface for the autoMRE web service.

Deliberately small: upload an archive, inspect it, start a reduction,
poll it, download the result. Progress is polled rather than streamed —
a reduction emits an event per query and there is nothing a user can do
with sixty updates a second, so a snapshot every second or two carries
the same information with none of the streaming-proxy edge cases.

This process executes the code that is uploaded to it. That is not a
side effect of the design, it is what reduction *is*: the tool learns
whether a candidate still reproduces the bug by running it. Deploy it
accordingly — see web/README.md.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from jobs import (
    MAX_ARCHIVE_BYTES,
    JobError,
    JobRegistry,
    count_python,
    discover,
    project_root,
    safe_extract,
)

app = FastAPI(title="autoMRE", version="0.3.0")

# The frontend is served from a different origin (Vercel), so it needs
# permission to call this. Set ALLOWED_ORIGINS to your deployment's URL
# in production; the default is permissive so local development works
# without configuration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

registry = JobRegistry(max_concurrent=int(os.environ.get("MAX_CONCURRENT", "1")))


async def _save_upload(upload: UploadFile) -> Path:
    """Stream an upload to disk, refusing one that is too large.

    Streamed in chunks and checked as it goes, so an oversized upload is
    rejected part-way through rather than after it has all been held in
    memory.
    """
    if not (upload.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "upload a .zip of your project")
    fd, tmp_name = tempfile.mkstemp(suffix=".zip")
    tmp = Path(tmp_name)
    size = 0
    with os.fdopen(fd, "wb") as out:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                out.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"archive is larger than "
                         f"{MAX_ARCHIVE_BYTES // (1024 * 1024)} MB")
            out.write(chunk)
    return tmp


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """What is in this archive, and which tests could anchor a reduction.

    Called on upload so the user picks a reproduction command from what
    is actually there. autoMRE needs that command to be exact — it is the
    definition of the behavior being preserved — and asking someone to
    type a pytest node id from memory is how they get it wrong.
    """
    archive = await _save_upload(file)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            try:
                safe_extract(archive, dest)
            except JobError as exc:
                raise HTTPException(400, str(exc)) from exc
            root = project_root(dest)
            files, lines = count_python(root)
            targets = discover(root)
            return {
                "files": files,
                "lines": lines,
                "installable": ((root / "pyproject.toml").exists()
                                or (root / "setup.py").exists()),
                "suggested_commands": [f"pytest {t}" for t in targets],
            }
    finally:
        archive.unlink(missing_ok=True)


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), command: str = Form(...)):
    parts = command.split()
    if not parts:
        raise HTTPException(400, "give a reproduction command")
    archive = await _save_upload(file)
    job = registry.submit(archive, parts)
    return JSONResponse({"job_id": job.job_id}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = registry.get(job_id)
    if job is None:
        # Jobs live in this process, so an unknown id after a restart is
        # a lost job rather than a wrong id. Say so.
        raise HTTPException(404, "no such job — the worker may have "
                                 "restarted since it was started")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not registry.cancel(job_id):
        raise HTTPException(409, "job is not running")
    return {"cancelled": True}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.expired:
        raise HTTPException(410, "this result has expired and its files have "
                                 "been deleted; start the reduction again")
    if job.result_zip is None or not job.result_zip.exists():
        raise HTTPException(409, f"no result yet (status: {job.status})")
    return FileResponse(job.result_zip, media_type="application/zip",
                        filename=f"automre-{job.job_id}.zip")
