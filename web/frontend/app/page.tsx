'use client';

/**
 * The whole UI, as one page of three steps: give us the project, tell us
 * which test defines the behavior to keep, watch it shrink.
 *
 * The interesting design constraint is that a reduction takes minutes to
 * hours and nobody can be asked to stare at a spinner for that long
 * without being told what is happening. So the progress panel reports
 * the unit of work the tool actually spends — one run of your test
 * command, which it calls a question — and an ETA presented as a range,
 * because the number of questions a project needs depends on how much of
 * it turns out to be removable, which cannot be known before asking.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import Result, { type Manifest } from './components/Result';
import {
  type Collected,
  ProjectError,
  collectEntries,
  collectFileList,
  commonPrefix,
  countPython,
  entriesFromDrop,
  zipProject,
} from './lib/project';

type Detected = {
  files: number;
  lines: number;
  installable: boolean;
  suggested_commands: string[];
};

type Status = {
  job_id: string;
  status:
    | 'queued'
    | 'preparing'
    | 'running'
    | 'done'
    | 'failed'
    | 'cancelled'
    // Out of budget rather than the user's choice. Still a real result:
    // every deletion in it was verified.
    | 'timed_out';
  phase: string;
  message: string;
  queries: number;
  estimated_queries: number;
  timed_out_queries: number;
  work_total: number;
  work_done: number;
  fraction: number | null;
  current_file: string | null;
  elapsed_seconds: number;
  eta_low_seconds: number | null;
  eta_high_seconds: number | null;
  error: string | null;
  original_files: number;
  original_lines: number;
  final_files: number;
  final_lines: number;
  protected_lines: number;
  has_result: boolean;
  has_manifest: boolean;
  expired: boolean;
  baseline_seconds: number | null;
  // A ceiling, not a prediction — it was 6.7x high on one measured
  // project and 4x low on another. Never render it as time remaining;
  // eta_low_seconds / eta_high_seconds are the honest pair.
  projected_ceiling_seconds: number | null;
};

const PHASES: Record<string, string> = {
  queued: 'Waiting to start',
  analyze: 'Tracing your test',
  analyzed: 'Tracing your test',
  'delete-unreachable': 'Deleting untouched files',
  'probe-imported': 'Probing imported-only files',
  inline: 'Trying to inline modules',
  'reduce-files': 'Shrinking files',
  'final-sweep': 'Final sweep',
  done: 'Finished',
};

function duration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export default function Page() {
  // The archive we will upload, and a human label for whatever produced
  // it — a dropped folder has no single filename.
  const [archive, setArchive] = useState<Blob | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [localCount, setLocalCount] = useState<number | null>(null);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [detected, setDetected] = useState<Detected | null>(null);
  const [command, setCommand] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dirInputRef = useRef<HTMLInputElement>(null);

  /** Send an archive to /api/detect so step 2 can offer real test names. */
  const inspect = useCallback(async (blob: Blob, name: string) => {
    setArchive(blob);
    setLabel(name);
    setDetected(null);
    setError(null);
    setJobId(null);
    setStatus(null);
    setManifest(null);
    setBusy(true);
    try {
      const body = new FormData();
      // The worker checks the filename ends in .zip before reading it.
      body.append('file', blob, 'project.zip');
      const res = await fetch('/api/detect', { method: 'POST', body });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'could not read the archive');
      setDetected(json);
      if (json.suggested_commands.length > 0) {
        setCommand(json.suggested_commands[0]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  /** A dropped or chosen folder: walk it, zip it, then inspect it. */
  const acceptCollected = useCallback(
    async (collected: Collected, fallbackName: string) => {
      setBusy(true);
      setError(null);
      setSkipped(collected.skipped);
      try {
        const pythons = countPython(collected);
        if (pythons === 0) {
          throw new ProjectError(
            'no .py files in there. autoMRE reduces Python projects — ' +
              'drop the folder that holds your package and its tests.',
          );
        }
        setLocalCount(pythons);
        const blob = await zipProject(collected);
        const name =
          commonPrefix(collected.files.keys()) ?? fallbackName;
        await inspect(blob, name);
      } catch (e) {
        setArchive(null);
        setLabel(null);
        setError(e instanceof Error ? e.message : String(e));
        setBusy(false);
      }
    },
    [inspect],
  );

  /** An already-zipped project: no walking needed. */
  const acceptZip = useCallback(
    async (picked: File) => {
      setSkipped([]);
      setLocalCount(null);
      await inspect(picked, picked.name);
    },
    [inspect],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const transfer = event.dataTransfer;
      // Read the entries synchronously: the DataTransfer is neutered as
      // soon as this handler yields, so anything awaited first finds it
      // empty. Collecting the files themselves can happen afterwards.
      const entries = entriesFromDrop(transfer);
      const files = Array.from(transfer.files ?? []);

      const singleZip =
        files.length === 1 && files[0].name.toLowerCase().endsWith('.zip');
      if (singleZip) {
        void acceptZip(files[0]);
        return;
      }
      if (entries.length > 0) {
        void collectEntries(entries).then((collected) =>
          acceptCollected(collected, 'dropped folder'),
        );
        return;
      }
      if (files.length > 0) {
        void acceptCollected(
          collectFileList(transfer.files!),
          'dropped files',
        );
      }
    },
    [acceptCollected, acceptZip],
  );

  const start = useCallback(async () => {
    if (!archive || !command.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const body = new FormData();
      body.append('file', archive, 'project.zip');
      body.append('command', command.trim());
      const res = await fetch('/api/jobs', { method: 'POST', body });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'could not start the job');
      setJobId(json.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [archive, command]);

  // Poll rather than stream. A reduction emits an event per question —
  // hundreds a minute — and no one can use that; a snapshot every second
  // and a half says the same thing and survives any proxy in between.
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    // A single bad response used to end polling for the rest of the
    // session, so a restarting worker or one dropped request left the
    // progress bar frozen at whatever it last said while the reduction
    // carried on finishing. Retried with a backoff instead, and only
    // given up on after enough consecutive failures to mean something.
    let failures = 0;
    const GIVE_UP_AFTER = 8;

    const tick = async () => {
      let delay = 1500;
      try {
        const res = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
        const json = await res.json();
        if (!live) return;
        if (res.status === 404) {
          // The one unrecoverable answer: jobs live in the worker's
          // memory, so an unknown id means it restarted and the job is
          // gone. Retrying cannot help.
          setError(json.detail ?? 'the job is no longer known to the worker');
          return;
        }
        if (!res.ok) throw new Error(json.detail ?? 'poll failed');
        failures = 0;
        setStatus(json);
        if (['done', 'failed', 'cancelled', 'timed_out'].includes(json.status)) {
          return;
        }
      } catch (e) {
        failures += 1;
        if (failures >= GIVE_UP_AFTER) {
          setError(
            e instanceof Error
              ? `lost track of the job: ${e.message}`
              : 'lost track of the job',
          );
          return;
        }
        delay = Math.min(10000, 1500 * 2 ** (failures - 1));
      }
      if (live) setTimeout(tick, delay);
    };
    tick();
    return () => {
      live = false;
    };
  }, [jobId]);

  // The per-file account, fetched once the job stops. Separate from the
  // status poll because it is answered once and the status is answered
  // every second and a half.
  useEffect(() => {
    if (!jobId || !status) return;
    if (!['done', 'cancelled', 'timed_out'].includes(status.status)) return;
    if (manifest) return;
    let live = true;
    void (async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}/manifest`, {
          cache: 'no-store',
        });
        if (!res.ok) return;
        const json = await res.json();
        if (live) setManifest(json);
      } catch {
        /* the numbers still render; the file list is the extra */
      }
    })();
    return () => {
      live = false;
    };
  }, [jobId, status, manifest]);

  const running =
    status !== null && ['queued', 'preparing', 'running'].includes(status.status);
  // timed_out belongs here, not with failures: a job that ran out of
  // budget still returns a verified, partially reduced project.
  const finished =
    status != null &&
    ['done', 'cancelled', 'timed_out'].includes(status.status);
  // Progress is lines-examined, not queries. The reducer walks the
  // surviving files once and knows how many lines that is; how many
  // questions it will take to get through them it cannot know, because
  // that depends on how much turns out to be removable.
  const pct = status?.fraction != null ? Math.min(99, status.fraction * 100) : 0;
  const indeterminate = status?.fraction == null;

  return (
    <main>
      <h1>autoMRE</h1>
      <p className="lede">
        Give it a Python project and one test. It deletes everything the
        test does not need — whole files, then definitions, then
        statements — checking after every deletion that your test still
        behaves exactly as it did. What comes back is the smallest
        project that still does the same thing.
      </p>

      {/* ---------------------------------------------------- step 1 */}
      <section className="step" data-state={archive ? 'ready' : 'active'}>
        <div className="step-head">
          <span className="step-num">1</span>
          <h2>Drop your project</h2>
        </div>
        <div
          className="drop"
          data-over={dragging}
          onClick={() => dirInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          {/* Two inputs because a browser file picker cannot offer both a
              folder and a file in one dialog. The folder picker is the
              default; the zip one is the fallback link below. */}
          <input
            ref={dirInputRef}
            type="file"
            /* Non-standard, supported everywhere, and not in React's
               typings — hence the spread. */
            {...({ webkitdirectory: '', directory: '' } as any)}
            multiple
            onChange={(e) => {
              if (e.target.files?.length) {
                void acceptCollected(
                  collectFileList(e.target.files),
                  'chosen folder',
                );
              }
            }}
          />
          <input
            ref={inputRef}
            type="file"
            accept=".zip"
            onChange={(e) => {
              const picked = e.target.files?.[0];
              if (picked) void acceptZip(picked);
            }}
          />
          {archive ? (
            <>
              <strong className="mono">{label}</strong>
              <p className="small muted" style={{ margin: '6px 0 0' }}>
                {detected
                  ? `${detected.files} Python files, ${detected.lines.toLocaleString()} lines` +
                    (detected.installable
                      ? ' · installable (pyproject/setup.py found)'
                      : ' · no pyproject.toml or setup.py')
                  : busy
                    ? localCount
                      ? `zipping ${localCount} Python files…`
                      : 'reading…'
                    : 'reading…'}
              </p>
            </>
          ) : (
            <>
              <strong>Drop the folder with the bug in it</strong>
              <p className="small muted" style={{ margin: '6px 0 0' }}>
                or click to choose a folder · up to 25&nbsp;MB
              </p>
              <p className="small muted" style={{ margin: '2px 0 0' }}>
                Caches, virtualenvs and <code>.git</code> are left behind
                automatically.
              </p>
            </>
          )}
        </div>
        {!archive && (
          <p className="small muted" style={{ marginTop: 8 }}>
            Already have a zip?{' '}
            <button className="link" onClick={() => inputRef.current?.click()}>
              Upload that instead
            </button>
            .
          </p>
        )}
        {skipped.length > 0 && (
          <p className="small muted" style={{ marginTop: 8 }}>
            Left out {skipped.length} path{skipped.length === 1 ? '' : 's'}:{' '}
            <span className="mono">
              {skipped.slice(0, 3).join(', ')}
              {skipped.length > 3 ? ', …' : ''}
            </span>
          </p>
        )}
      </section>

      {/* ---------------------------------------------------- step 2 */}
      <section className="step" data-state={detected ? 'active' : 'waiting'}>
        <div className="step-head">
          <span className="step-num">2</span>
          <h2>Name the test that must keep working</h2>
        </div>
        <p className="small muted">
          This is the definition of what gets preserved. Everything the
          command still needs survives; everything else goes. The test
          file itself is never touched — it is the question, not the
          answer.
        </p>

        {detected && detected.suggested_commands.length > 0 && (
          <div className="chips">
            {detected.suggested_commands.slice(0, 12).map((c) => (
              <button
                key={c}
                className="chip"
                data-active={command === c}
                onClick={() => setCommand(c)}
                title={c}
              >
                {c.replace(/^pytest /, '')}
              </button>
            ))}
          </div>
        )}

        <input
          type="text"
          value={command}
          placeholder="pytest tests/test_bug.py::test_the_thing"
          onChange={(e) => setCommand(e.target.value)}
          disabled={!detected}
        />

        {detected && detected.suggested_commands.length === 0 && (
          <div className="note">
            No <code>test_*.py</code> files found. You can still use any
            command that reproduces the behavior, such as{' '}
            <code>python main.py</code> — a script that crashes works too,
            and the traceback becomes the thing being preserved.
          </div>
        )}

        <button
          className="primary"
          disabled={!archive || !command.trim() || busy || running}
          onClick={start}
        >
          {busy ? 'Working…' : 'Shrink it'}
        </button>
      </section>

      {/* ---------------------------------------------------- step 3 */}
      {(running || finished || status?.status === 'failed') && (
        <section className="step">
          <div className="step-head">
            <span className="step-num">3</span>
            <h2>{PHASES[status?.phase ?? 'queued'] ?? 'Working'}</h2>
          </div>

          <p className="small muted" style={{ marginBottom: 0 }}>
            {status?.message}
          </p>

          {running && (
            <>
              <div className="bar" data-indeterminate={indeterminate}>
                <span style={{ width: `${pct}%` }} />
              </div>
              <p className="small muted">
                {indeterminate ? (
                  <>Working through the setup phases · </>
                ) : (
                  <>
                    {status!.work_done.toLocaleString()} of{' '}
                    {status!.work_total.toLocaleString()} lines examined ·{' '}
                  </>
                )}
                {status!.queries.toLocaleString()} questions asked ·{' '}
                {duration(status!.elapsed_seconds)} elapsed
                {status!.eta_low_seconds !== null && (
                  <>
                    {' '}
                    · about {duration(status!.eta_low_seconds)}–
                    {duration(status!.eta_high_seconds!)} to go
                  </>
                )}
              </p>
              {status!.eta_low_seconds !== null && (
                <p className="small muted">
                  The range is wide on purpose. Time here is dominated by
                  running your test over and over, and how many runs a file
                  needs depends on how much of it turns out to be
                  removable — which is only learned by asking.
                </p>
              )}
              {status!.timed_out_queries > 0 && (
                <div className="note">
                  {status!.timed_out_queries} question
                  {status!.timed_out_queries === 1 ? '' : 's'} timed out —
                  a deletion made your code stop terminating, so it was
                  refused and the code kept. Nothing is wrong.
                </div>
              )}
              <p>
                <button
                  className="link"
                  onClick={() => fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' })}
                >
                  Stop and keep what has been reduced so far
                </button>
              </p>
            </>
          )}

          {finished && (
            <>
              <div className="stats">
                <div className="stat">
                  <b>
                    {(
                      ((status!.original_lines - status!.final_lines) /
                        Math.max(1, status!.original_lines)) *
                      100
                    ).toFixed(1)}
                    %
                  </b>
                  <span>of lines removed</span>
                </div>
                <div className="stat">
                  <b>
                    {status!.original_lines.toLocaleString()} →{' '}
                    {status!.final_lines.toLocaleString()}
                  </b>
                  <span>lines</span>
                </div>
                <div className="stat">
                  <b>
                    {status!.original_files} → {status!.final_files}
                  </b>
                  <span>files</span>
                </div>
                <div className="stat">
                  <b>{status!.queries.toLocaleString()}</b>
                  <span>questions asked</span>
                </div>
                <div className="stat">
                  <b>{duration(status!.elapsed_seconds)}</b>
                  <span>taken</span>
                </div>
              </div>

              {/* The actual claim, said once in a sentence. The tiles
                  above report how much went away; this is what makes it
                  a result rather than a statistic. */}
              <p className="claim">
                {status!.original_files} file
                {status!.original_files === 1 ? '' : 's'} →{' '}
                {status!.final_files}, {status!.original_lines.toLocaleString()}{' '}
                → {status!.final_lines.toLocaleString()} lines, and{' '}
                <code>{command.trim()}</code> still does exactly what it did
                before — checked {status!.queries.toLocaleString()} times.
              </p>

              {status!.status === 'cancelled' && (
                <div className="note">
                  Stopped early, so this is not fully minimized — but every
                  deletion in it was checked against your test, so it does
                  still behave the same way.
                </div>
              )}

              {status!.status === 'timed_out' && (
                <div className="note">
                  This hit the time limit rather than finishing, so there
                  may be more to remove. Everything in it was still checked
                  against your test. Running it locally has no time limit:{' '}
                  <code>
                    automre reduce-project . -c &quot;{command.trim()}&quot;
                  </code>
                </div>
              )}

              <p style={{ marginTop: 18 }}>
                <a className="primary" href={`/api/jobs/${jobId}/result`}
                   style={{ textDecoration: 'none', padding: '10px 18px',
                            background: 'var(--accent)', color: '#fff',
                            borderRadius: 8 }}>
                  Download the reduced project
                </a>
              </p>
              <p className="small muted">
                Unzip it and run your command again — it should print
                exactly what it printed before.
              </p>

              {manifest && <Result jobId={jobId!} manifest={manifest} />}
            </>
          )}

          {status?.status === 'failed' && (
            <div className="note error">{status.error}</div>
          )}
        </section>
      )}

      {error && <div className="note error">{error}</div>}

      <footer>
        <p>
          Everything you upload is executed on the server — that is how
          the tool works, since it learns whether a deletion is safe by
          running your test. Do not upload anything you would not run on a
          machine you do not control.
        </p>
        <p>
          The same reduction runs locally with{' '}
          <code>automre reduce-project . -c &quot;pytest ...&quot;</code>,
          with no upload at all.
        </p>
      </footer>
    </main>
  );
}
