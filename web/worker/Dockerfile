# The reduction worker. This container runs code that users upload to it,
# which is not incidental — running the candidate is how the oracle
# works. Everything below assumes that and tries to make it survivable.

FROM python:3.12-slim

# git is here because a project's own build may want it; build-essential
# because a `pip install -e .` on a project with a C extension will fail
# without a compiler, and that failure would look like an autoMRE bug.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY web/worker/requirements.txt /srv/web/worker/requirements.txt
RUN pip install --no-cache-dir -r /srv/web/worker/requirements.txt

COPY automre /srv/automre
COPY web/worker /srv/web/worker

# Never as root. A job creates its own venv and installs the uploaded
# project into it, so this user needs a writable home and nothing else.
RUN useradd --create-home --uid 10001 runner \
    && chown -R runner:runner /srv
USER runner
ENV HOME=/home/runner \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv/web/worker
EXPOSE 8000

# One worker process. Jobs are held in memory and a reduction saturates a
# core, so a second process would both lose jobs (a poll could land on
# the process that does not have it) and halve the speed of the first.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
