# The image one reduction runs in, inside a Vercel Sandbox microVM.
#
# NOT YET BUILT OR PUSHED. It is a direct derivation of
# web/worker/Dockerfile, which is built and working, but nothing here has
# been through `docker build` — no daemon was available when it was
# written. Build it before trusting it.
#
# Same premise as the worker's image: this runs code that users upload,
# which is not incidental — running the candidate is how the oracle
# works. The difference is where the isolation comes from. The worker
# relies on a non-root user inside a shared container; here the sandbox
# is the boundary, so the job gets a microVM of its own and this image
# only has to carry the tools.
#
# Build from the repository root, not from this directory — it copies
# automre/ as well:
#
#     docker build -f web/sandbox/Dockerfile -t automre-sandbox .
#
# Then push to Vercel Container Registry and name it in Sandbox.create's
# `image`. The point of a custom image is that per-job setup drops to
# zero: installing build-essential and autoMRE's own stack costs 30-60
# seconds of real CPU, and on a 5-CPU-hour monthly budget paying that on
# every job spends up to 17% of the month installing the same four
# packages.

FROM python:3.12-slim

# git because a project's own build may want it; build-essential because
# a `pip install -e .` on a project with a C extension fails without a
# compiler, and that failure reads as an autoMRE bug rather than a
# missing toolchain. Both belong in the image rather than in the job:
# they are the same for every project and apt is slow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# autoMRE's own dependencies — tree-sitter, tree-sitter-python, coverage.
# The uploaded project's dependencies are a different thing entirely and
# are installed per job into a throwaway venv by `provision`, which is
# also why they cannot be baked in here.
COPY web/worker/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY automre /srv/automre
COPY web/runner /srv/web/runner

# No USER, no EXPOSE, no CMD, deliberately — the three things the
# worker's image needs and this one does not:
#
#   * the sandbox supplies its own user and the microVM is the boundary,
#     so a hardcoded uid 10001 is inert at best and misowns files at worst
#   * nothing listens here; the runner is invoked through runCommand, and
#     an exposed port would also make its traffic billable
#   * the command is supplied per job by the control plane, with the job
#     id, archive, reproduction command and budget
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# PYTHONDONTWRITEBYTECODE is not a tidiness setting. The reducer rewrites
# the same file several times a second, and any two candidates that
# happen to share a byte length are indistinguishable to Python's
# bytecode staleness check — so it imports the previous candidate's
# bytecode and the validator reports on code that is not on disk. The
# oracle is unsound without this.

WORKDIR /srv
