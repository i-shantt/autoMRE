'use client';

/**
 * What the reduction did, file by file.
 *
 * The five numeric tiles that used to be the whole result screen answer
 * "how much went away" and not "what went away", which is the question
 * somebody who is about to paste this into a bug report actually has. So
 * the left pane is every file with its verdict, and the right is the
 * lines that left.
 *
 * The diff is computed in the browser. The server has both sides on disk
 * and could produce it, but a reduction's cost is CPU and a unified diff
 * of two small files is free here.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { diffLines } from 'diff';

export type ManifestFile = {
  path: string;
  status:
    | 'deleted'
    | 'inlined'
    | 'reduced'
    | 'unchanged'
    | 'protected'
    | 'untouchable'
    | 'outside-project'
    | 'errored';
  before_lines: number | null;
  after_lines: number | null;
  reason?: string;
  error?: string;
  /** Changed without getting shorter — see the note in the diff pane. */
  same_line_count?: boolean;
};

export type Manifest = {
  job_id: string;
  command: string[];
  status: string;
  root_prefix: string;
  files: ManifestFile[];
  truncated: boolean;
};

/** Ordered so the interesting files come first. */
const GROUP_ORDER: ManifestFile['status'][] = [
  'reduced',
  'unchanged',
  'protected',
  'inlined',
  'untouchable',
  'errored',
  'outside-project',
  'deleted',
];

const GROUP_LABEL: Record<string, string> = {
  reduced: 'Shrunk',
  unchanged: 'Kept whole',
  protected: 'Never touched',
  inlined: 'Folded into another file',
  untouchable: 'Left alone (not valid UTF-8)',
  errored: 'Skipped after an error',
  'outside-project': 'Outside the project',
  deleted: 'Deleted',
};

const GROUP_NOTE: Record<string, string> = {
  protected:
    'The test being preserved, and any conftest.py above it. These define ' +
    'the behaviour being kept, so they are the question rather than the ' +
    'answer — identical on purpose.',
  inlined:
    'The contents moved into the file that imported them, and then the ' +
    'file itself was no longer needed.',
  untouchable:
    'Not valid UTF-8, so the reducer would not risk rewriting the bytes.',
  errored:
    'Reducing these raised. The run continued without them rather than ' +
    'ending over one file, so they are larger than they could be.',
  deleted: 'Nothing the command runs reached these.',
};

function pct(file: ManifestFile): string | null {
  if (file.before_lines == null || file.after_lines == null) return null;
  if (file.before_lines === 0) return null;
  const removed = file.before_lines - file.after_lines;
  if (removed <= 0) return null;
  return `−${Math.round((removed / file.before_lines) * 100)}%`;
}

type Side = { exists: boolean; text: string | null; binary?: boolean };

export default function Result({
  jobId,
  manifest,
}: {
  jobId: string;
  manifest: Manifest;
}) {
  const groups = useMemo(() => {
    const by = new Map<string, ManifestFile[]>();
    for (const file of manifest.files) {
      if (!by.has(file.status)) by.set(file.status, []);
      by.get(file.status)!.push(file);
    }
    return GROUP_ORDER.filter((s) => by.has(s)).map((s) => ({
      status: s,
      files: by.get(s)!,
    }));
  }, [manifest]);

  // Open the groups worth reading, collapse the long boring one. On a
  // real reduction most files are deleted and that list says little; the
  // handful that survived is the result.
  const [open, setOpen] = useState<Set<string>>(
    () => new Set(groups.filter((g) => g.status !== 'deleted').map((g) => g.status)),
  );
  const [selected, setSelected] = useState<string | null>(
    () => groups.find((g) => g.status === 'reduced')?.files[0]?.path ?? null,
  );
  const [before, setBefore] = useState<Side | null>(null);
  const [after, setAfter] = useState<Side | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(
    async (path: string) => {
      setLoading(true);
      setLoadError(null);
      setBefore(null);
      setAfter(null);
      try {
        const fetchSide = async (side: 'before' | 'after') => {
          const res = await fetch(
            `/api/jobs/${jobId}/file?path=${encodeURIComponent(path)}&side=${side}`,
            { cache: 'no-store' },
          );
          const json = await res.json();
          if (!res.ok) throw new Error(json.detail ?? `could not read ${side}`);
          return json as Side;
        };
        const [b, a] = await Promise.all([fetchSide('before'), fetchSide('after')]);
        setBefore(b);
        setAfter(a);
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [jobId],
  );

  useEffect(() => {
    if (selected) load(selected);
  }, [selected, load]);

  const selectedFile = manifest.files.find((f) => f.path === selected) ?? null;

  const hunks = useMemo(() => {
    if (!before?.text && !after?.text) return null;
    return diffLines(before?.text ?? '', after?.text ?? '');
  }, [before, after]);

  return (
    <div className="result">
      <aside className="tree">
        {manifest.truncated && (
          <p className="small muted" style={{ padding: '0 4px 8px' }}>
            The list is capped; every surviving file is here, and some of
            the deleted ones are not.
          </p>
        )}
        {groups.map((group) => (
          <div key={group.status} className="tree-group">
            <button
              className="tree-head"
              aria-expanded={open.has(group.status)}
              onClick={() =>
                setOpen((prev) => {
                  const next = new Set(prev);
                  if (next.has(group.status)) next.delete(group.status);
                  else next.add(group.status);
                  return next;
                })
              }
            >
              <span className="caret">{open.has(group.status) ? '▾' : '▸'}</span>
              {GROUP_LABEL[group.status] ?? group.status}
              <span className="count">{group.files.length}</span>
            </button>
            {open.has(group.status) && (
              <>
                {GROUP_NOTE[group.status] && (
                  <p className="tree-note small muted">
                    {GROUP_NOTE[group.status]}
                  </p>
                )}
                <ul>
                  {group.files.map((file) => (
                    <li key={file.path}>
                      <button
                        className="tree-file"
                        data-active={file.path === selected}
                        data-status={file.status}
                        onClick={() => setSelected(file.path)}
                        title={file.path}
                      >
                        <span className="mono path">{file.path}</span>
                        {pct(file) && <span className="delta">{pct(file)}</span>}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        ))}
      </aside>

      <section className="diff">
        {!selected && (
          <p className="small muted">Pick a file to see what changed.</p>
        )}
        {selected && (
          <>
            <header>
              <strong className="mono">{selected}</strong>
              {selectedFile && (
                <span className="small muted">
                  {selectedFile.status}
                  {selectedFile.before_lines != null &&
                    selectedFile.after_lines != null && (
                      <>
                        {' · '}
                        {selectedFile.before_lines} → {selectedFile.after_lines}{' '}
                        lines
                      </>
                    )}
                </span>
              )}
            </header>

            {loading && <p className="small muted">Reading…</p>}
            {loadError && <div className="note error">{loadError}</div>}

            {!loading && !loadError && selectedFile?.status === 'protected' && (
              <div className="note">
                Identical on purpose. This file is the behaviour being
                preserved, so the reducer was never allowed to touch it.
              </div>
            )}

            {!loading && !loadError && selectedFile?.same_line_count && (
              <div className="note">
                Same number of lines, different code. Removing a function
                body leaves a <code>pass</code> behind, so a definition
                that is now a placeholder costs about as many lines as it
                did — the logic is gone even though the count did not move.
              </div>
            )}

            {!loading && !loadError && after && !after.exists && (
              <div className="note">
                This file is gone from the result — nothing the command
                runs needed it.
              </div>
            )}

            {!loading && !loadError && (before?.binary || after?.binary) && (
              <div className="note">
                Not valid UTF-8, so there is nothing to show as text.
              </div>
            )}

            {!loading && !loadError && hunks && (
              <pre className="hunks">
                {hunks.map((part, i) => {
                  const cls = part.added ? 'add' : part.removed ? 'del' : 'same';
                  const sign = part.added ? '+' : part.removed ? '−' : ' ';
                  // Trailing newline would render one blank row per hunk.
                  const lines = part.value.replace(/\n$/, '').split('\n');
                  return (
                    <span key={i} className={cls}>
                      {lines.map((line, j) => (
                        <span key={j} className="row">
                          <span className="sign">{sign}</span>
                          {line || ' '}
                        </span>
                      ))}
                    </span>
                  );
                })}
              </pre>
            )}
          </>
        )}
      </section>
    </div>
  );
}
