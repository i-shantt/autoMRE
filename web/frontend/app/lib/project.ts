/**
 * Turning whatever someone dropped into the one thing the worker takes.
 *
 * The API takes a zip, and asking a user to make one first is the kind of
 * friction this interface exists to remove — the project is already a
 * folder on their disk. So a dropped folder is walked here and zipped in
 * the browser.
 *
 * Folder drops are not `DataTransfer.files`. That list contains a useless
 * zero-byte entry for a directory; the real tree only comes from the
 * non-standard but universally supported `webkitGetAsEntry`. The entries
 * must also be read out of `dataTransfer.items` *synchronously* during
 * the drop event, because the DataTransfer is neutered the moment the
 * handler yields — hence the two-pass shape below, collecting entries
 * first and awaiting them afterwards.
 */

import { zip } from 'fflate';

/** Caps mirrored from webcore/archive.py, checked before we upload. */
export const MAX_ARCHIVE_BYTES = 25 * 1024 * 1024;
export const MAX_FILES = 5000;

/** Directories never worth uploading: caches, environments, and vendored deps. */
const SKIP_DIRS = new Set([
  '.git', '__pycache__', '.venv', 'venv', 'env', '.env',
  'node_modules', '.tox', '.nox', '.mypy_cache', '.ruff_cache',
  '.pytest_cache', '.idea', '.vscode', 'dist', 'build', '.eggs',
  '.DS_Store', '.next', 'site-packages',
]);

export type Collected = {
  /** Relative path -> file. Paths use forward slashes, as zips do. */
  files: Map<string, File>;
  bytes: number;
  /** Names of things left out, so the UI can say so rather than silently drop them. */
  skipped: string[];
};

function shouldSkip(path: string): boolean {
  return path.split('/').some((part) => SKIP_DIRS.has(part));
}

function readEntries(reader: any): Promise<any[]> {
  // readEntries yields at most 100 at a time and signals the end with an
  // empty batch, so a single call silently truncates any directory with
  // more than 100 children.
  return new Promise((resolve, reject) => {
    const all: any[] = [];
    const pump = () =>
      reader.readEntries((batch: any[]) => {
        if (batch.length === 0) {
          resolve(all);
          return;
        }
        all.push(...batch);
        pump();
      }, reject);
    pump();
  });
}

function fileOf(entry: any): Promise<File> {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

async function walk(entry: any, prefix: string, out: Collected): Promise<void> {
  const path = prefix ? `${prefix}/${entry.name}` : entry.name;
  if (shouldSkip(path)) {
    out.skipped.push(path);
    return;
  }
  if (entry.isFile) {
    const file = await fileOf(entry);
    out.files.set(path, file);
    out.bytes += file.size;
    return;
  }
  if (entry.isDirectory) {
    const children = await readEntries(entry.createReader());
    for (const child of children) await walk(child, path, out);
  }
}

/**
 * Collect a drop. Call `items` reading synchronously — see the note above.
 */
export function entriesFromDrop(transfer: DataTransfer): any[] {
  const entries: any[] = [];
  for (const item of Array.from(transfer.items ?? [])) {
    const entry = (item as any).webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }
  return entries;
}

export async function collectEntries(entries: any[]): Promise<Collected> {
  const out: Collected = { files: new Map(), bytes: 0, skipped: [] };
  for (const entry of entries) await walk(entry, '', out);
  return out;
}

/** Collect a plain FileList, from `<input webkitdirectory>` or a multi-select. */
export function collectFileList(list: FileList): Collected {
  const out: Collected = { files: new Map(), bytes: 0, skipped: [] };
  for (const file of Array.from(list)) {
    // webkitRelativePath is set when the input had `webkitdirectory`;
    // otherwise the file was picked individually and has no path.
    const path = (file as any).webkitRelativePath || file.name;
    if (shouldSkip(path)) {
      out.skipped.push(path);
      continue;
    }
    out.files.set(path, file);
    out.bytes += file.size;
  }
  return out;
}

/**
 * The single top-level directory every collected path shares, if there is
 * one.
 *
 * Worth knowing because the worker's `project_root` strips exactly one
 * such wrapper, so this is what its paths will be relative to — and it is
 * the obvious thing to show the user as "the project you dropped".
 */
export function commonPrefix(paths: Iterable<string>): string | null {
  let prefix: string | null = null;
  for (const path of paths) {
    const head = path.includes('/') ? path.slice(0, path.indexOf('/')) : null;
    if (head === null) return null;
    if (prefix === null) prefix = head;
    else if (prefix !== head) return null;
  }
  return prefix;
}

export class ProjectError extends Error {}

/**
 * Zip a collection, preserving relative paths.
 *
 * Stored, not deflated. Python source compresses well, but the whole
 * archive is unpacked again a second later on the other side, and
 * spending a user's main thread on it buys nothing: the limit that
 * matters is the 25 MB the worker enforces on the *uncompressed* bytes
 * anyway. Compression is left to fflate's default level for anything
 * large enough to matter over the wire.
 */
export async function zipProject(collected: Collected): Promise<Blob> {
  if (collected.files.size === 0) {
    throw new ProjectError(
      'nothing to upload — the drop contained no files, or all of them ' +
        'were caches and virtualenvs',
    );
  }
  if (collected.files.size > MAX_FILES) {
    throw new ProjectError(
      `${collected.files.size.toLocaleString()} files; the limit is ` +
        `${MAX_FILES.toLocaleString()}. Drop just the package and its tests.`,
    );
  }
  if (collected.bytes > MAX_ARCHIVE_BYTES) {
    throw new ProjectError(
      `${(collected.bytes / 1024 / 1024).toFixed(1)} MB; the limit is ` +
        `${MAX_ARCHIVE_BYTES / 1024 / 1024} MB.`,
    );
  }

  const payload: Record<string, Uint8Array> = {};
  for (const [path, file] of collected.files) {
    payload[path] = new Uint8Array(await file.arrayBuffer());
  }

  return new Promise((resolve, reject) => {
    zip(payload, { level: 6 }, (err, data) => {
      if (err) reject(new ProjectError(`could not build the archive: ${err}`));
      // Copy into a fresh buffer: fflate hands back a view that may be
      // backed by a larger pooled ArrayBuffer, and Blob would otherwise
      // capture the whole thing.
      else resolve(new Blob([new Uint8Array(data)], { type: 'application/zip' }));
    });
  });
}

/** How many Python files are in here, for the "is this the right folder" check. */
export function countPython(collected: Collected): number {
  let n = 0;
  for (const path of collected.files.keys()) if (path.endsWith('.py')) n += 1;
  return n;
}
