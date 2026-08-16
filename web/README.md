# autoMRE on the web

Drop in a zipped Python project, name the test that has to keep working,
get back the smallest project that still behaves the same way.

Two pieces, because they have incompatible requirements:

```
web/frontend/   Next.js UI            -> Vercel
web/worker/     FastAPI + autoMRE     -> Fly.io / Render / any container host
```

## Why the worker is not on Vercel

It cannot be, and this is worth stating plainly rather than discovering
during a deploy.

A reduction works by running your test command over and over — once per
candidate deletion — and keeping the deletions that leave the behavior
unchanged. That is thousands of subprocesses over minutes to hours: 3.6
minutes for a 11k-line project, 10 minutes for a 17k-line one, hours for
a parser-shaped one. It also needs a real Python environment with your
project's dependencies installed, and a working directory that survives
between requests.

Vercel functions cap at 60s (Hobby), 300s (Pro), 800s (Fluid maximum),
have an ephemeral filesystem, and cannot run background work past the
response. Every one of those is a hard blocker, so the UI ships to
Vercel and the engine ships to a machine that can hold a long job.

## Run it locally

Two terminals.

```bash
# 1. the worker
cd web/worker
pip install -r requirements.txt
uvicorn main:app --port 8000

# 2. the UI
cd web/frontend
npm install
cp .env.example .env.local          # WORKER_URL=http://127.0.0.1:8000
npm run dev
```

Then open http://localhost:3000.

## Deploy

**Worker** — Fly.io, from the repo root (the Dockerfile copies `automre/`
as well as `web/worker/`, so it must build from the root):

```bash
fly launch --no-deploy --name automre-worker
fly deploy --dockerfile web/worker/Dockerfile
```

Any container host works the same way; `web/worker/fly.toml` documents
the settings that matter. Two of them are load-bearing:

- **`auto_stop_machines = false`.** A job lives in the worker's memory
  and on its disk. A machine that suspends between polls loses it.
- **One process, one job at a time.** Jobs are held in memory, so a
  second process would answer polls for jobs it does not have; and a
  reduction saturates a core, so running two halves the speed of both
  and makes every ETA wrong. Raise `MAX_CONCURRENT` only alongside more
  CPUs.

**Frontend** — Vercel, with the root directory set to `web/frontend`:

```bash
cd web/frontend
vercel --prod
```

Set `WORKER_URL` in the Vercel project's environment variables to the
worker's public URL. It stays server-side: the browser only ever calls
this app's own `/api/*` routes, which forward.

Set `ALLOWED_ORIGINS` on the worker to your Vercel URL once you know it.
The default is `*`, which is right for local development and wrong for
anything public.

## Security

**The worker executes the code that is uploaded to it.** This is not an
oversight to be hardened away — running the candidate is *how the oracle
works*. There is no version of this tool that decides whether a deletion
is safe without running your code.

What is in place:

- the container runs as a non-root user with no privileges;
- uploads are capped at 25 MB, 250 MB uncompressed, 5000 entries;
- archive entries that resolve outside the extraction directory are
  refused, so a zip cannot write to `../../anywhere`;
- each job gets a fresh temp directory and its own virtualenv;
- a job is killed at one hour;
- a finished job's directory — virtualenv, project, result zip, on the
  order of a hundred megabytes — is deleted an hour after it finishes,
  and a download after that is told the result expired rather than being
  handed a stale one.

What is **not** in place, and what you should add before pointing this
at the public internet:

- no network egress restrictions — an uploaded project can `pip install`
  and can make outbound requests;
- no per-user isolation, rate limiting, or authentication;
- no CPU or memory cgroup limits beyond the machine's own.

For a public deployment, run each job in a disposable microVM or
gVisor-style sandbox with egress denied, and put auth in front. For a
team-internal deployment on trusted code, what is here is reasonable.

If you would rather not upload anything at all, the same reduction runs
locally and always has:

```bash
automre reduce-project ./my_project -c "pytest tests/test_bug.py::test_foo"
```

## Why a job can be refused before it starts

Between installing the project and the first deletion, the worker runs
`provision.gate.check`: the named test has to exist, the command has to
run, do the same thing twice, and stop working when the package is
renamed aside. A job that fails any of those is reported as failed with
the reason rather than reduced.

This is not caution for its own sake. Without it, a job naming a test
that does not exist ran to completion and reported success: pytest exited
4, the oracle adopted *exiting 4* as the behavior to preserve, and the
reducer deleted everything not needed to keep exiting 4. The download
contained the library reduced to blank lines.

A reduction cannot notice that from the inside — every candidate really
does preserve the behavior it was handed. The only place to catch it is
before the first query.

## How progress is reported

A reduction cannot say how long it will take, and the interesting part
of this UI is that it does not pretend to.

The unit of work is the **query**: one run of your test command. How
many a project needs depends on how much of it turns out to be
removable, which is only learned by asking — measured across the
benchmark repos, the cost varies from 0.11 to 0.48 queries per line, a
fourfold spread, and seven repositories outside the benchmark — 54k to
687k lines — all came in *under* the bottom of that range, between 0.052
and 0.097. So the query count is displayed as a count of work done,
never as a fraction of a total.

The progress bar instead tracks **lines examined out of lines to
examine**. The reducer walks the surviving files once in Phase 4, which
is ~97% of a run's queries, and it knows how many lines that walk
covers before it starts. That is a real fraction.

The ETA extrapolates from the rate of that walk, times out at the
prelude (unpacking, install, coverage trace) rather than including it,
and is shown as a range rather than a number. It stays hidden until 3%
of the work is done, because the reducer sorts files largest-first and
an estimate drawn from the first file alone is systematically
pessimistic.

## API

The worker is small enough to drive directly:

| | |
|---|---|
| `POST /api/detect` | multipart `file` → file/line counts and suggested pytest node ids |
| `POST /api/jobs` | multipart `file`, `command` → `{job_id}` |
| `GET /api/jobs/{id}` | status, phase, progress, ETA range |
| `POST /api/jobs/{id}/cancel` | stop early, keeping what has been reduced |
| `GET /api/jobs/{id}/result` | the reduced project as a zip |

Polling, not streaming: a reduction emits an event per query — hundreds
a minute — and nobody can use that. A snapshot every second and a half
carries the same information and survives any proxy in between.
