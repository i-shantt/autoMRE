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

type Detected = {
  files: number;
  lines: number;
  installable: boolean;
  suggested_commands: string[];
};

type Status = {
  job_id: string;
  status: 'queued' | 'preparing' | 'running' | 'done' | 'failed' | 'cancelled';
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
  has_result: boolean;
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
  const [file, setFile] = useState<File | null>(null);
  const [detected, setDetected] = useState<Detected | null>(null);
  const [command, setCommand] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback(async (picked: File) => {
    setFile(picked);
    setDetected(null);
    setError(null);
    setJobId(null);
    setStatus(null);
    setBusy(true);
    try {
      const body = new FormData();
      body.append('file', picked);
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

  const start = useCallback(async () => {
    if (!file || !command.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const body = new FormData();
      body.append('file', file);
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
  }, [file, command]);

  // Poll rather than stream. A reduction emits an event per question —
  // hundreds a minute — and no one can use that; a snapshot every second
  // and a half says the same thing and survives any proxy in between.
  useEffect(() => {
    if (!jobId) return;
    let live = true;
    const tick = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
        const json = await res.json();
        if (!live) return;
        if (!res.ok) {
          setError(json.detail ?? 'lost track of the job');
          return;
        }
        setStatus(json);
        if (['done', 'failed', 'cancelled'].includes(json.status)) return;
      } catch {
        /* a dropped poll is not worth surfacing; the next one will tell */
      }
      if (live) setTimeout(tick, 1500);
    };
    tick();
    return () => {
      live = false;
    };
  }, [jobId]);

  const running =
    status !== null && ['queued', 'preparing', 'running'].includes(status.status);
  const finished = status?.status === 'done' || status?.status === 'cancelled';
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
      <section className="step" data-state={file ? 'ready' : 'active'}>
        <div className="step-head">
          <span className="step-num">1</span>
          <h2>Upload your project</h2>
        </div>
        <div
          className="drop"
          data-over={dragging}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const dropped = e.dataTransfer.files?.[0];
            if (dropped) accept(dropped);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".zip"
            onChange={(e) => {
              const picked = e.target.files?.[0];
              if (picked) accept(picked);
            }}
          />
          {file ? (
            <>
              <strong className="mono">{file.name}</strong>
              <p className="small muted" style={{ margin: '6px 0 0' }}>
                {detected
                  ? `${detected.files} Python files, ${detected.lines.toLocaleString()} lines` +
                    (detected.installable
                      ? ' · installable (pyproject/setup.py found)'
                      : ' · no pyproject.toml or setup.py')
                  : 'reading…'}
              </p>
            </>
          ) : (
            <>
              <strong>Drop a .zip of your project</strong>
              <p className="small muted" style={{ margin: '6px 0 0' }}>
                or click to choose · up to 25&nbsp;MB
              </p>
            </>
          )}
        </div>
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
          disabled={!file || !command.trim() || busy || running}
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

              {status!.status === 'cancelled' && (
                <div className="note">
                  Stopped early, so this is not fully minimized — but every
                  deletion in it was checked against your test, so it does
                  still behave the same way.
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
