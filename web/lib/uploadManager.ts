// The upload queue as a module-level store, so it survives the page that started it.
//
// The queue lived in the "new annotation" page's React state, which means it lived exactly as long as that
// component was mounted. Clicking a row's "open" link, a breadcrumb, or any menu item unmounted the page and
// abandoned every upload still in flight. A 186-file, 41GB batch is over an hour of transfer; asking somebody
// not to touch the rest of the app for an hour is not a workable design, and losing the batch because they
// did is worse.
//
// So the state and the run loop are module-scoped, in the same shape and for the same reason as toast.ts:
// a plain module can own something the whole app observes, and React subscribes to it rather than holding
// it. Navigating within the app now re-renders a view over a queue that never stopped. The fetches were
// never the problem, they continue regardless of React; the problem was that nothing was left holding their
// results.
//
// What this does not survive is the tab closing, because nothing in a browser can. That is what the
// beforeunload guard is for: it is registered only while work is in flight, so it never nags somebody who
// has nothing running.

import { toast } from "./toast";
import {
  type QueueItem,
  buildQueue,
  importedSessions,
  nextPending,
  patchItem,
  summarize,
} from "./uploadQueue";

export type Phase = "idle" | "importing" | "autolabeling";

export type UploadState = {
  items: QueueItem[];
  running: boolean;
  /** Which pass is in flight, so a global indicator can say what it is doing rather than only that it is. */
  phase: Phase;
  /** The settings the run was started with, so a view can show what a queue is importing as. */
  target: { format: string; vehicle: string; city: string } | null;
};

export type UploadDeps = {
  upload: (file: File, onProgress: (frac: number) => void) => Promise<string>;
  startImport: (body: { format: string; source_uri: string; target_vehicle: string; city: string })
    => Promise<{ job_id: string }>;
  watchImport: (jobId: string, onProgress: (j: { progress?: number | null }) => void) => Promise<string>;
  firstFrame: (sessionId: string) => Promise<{ frame_id: string }>;
  humanizeError: (e: unknown) => string;
  startAutolabel?: (sessionId: string) => Promise<{ job_id: string }>;
  watchAutolabel?: (jobId: string, onProgress: (j: { progress?: number | null }) => void)
    => Promise<{ objects?: number }>;
};

type Listener = (s: UploadState) => void;

const listeners = new Set<Listener>();
const blobs = new Map<string, File>();
let state: UploadState = { items: [], running: false, phase: "idle", target: null };

function emit() {
  // A fresh object each time, so a subscriber comparing by reference sees the change.
  state = { ...state };
  listeners.forEach((fn) => fn(state));
}

export function subscribeUploads(fn: Listener): () => void {
  listeners.add(fn);
  fn(state);
  return () => {
    listeners.delete(fn);
  };
}

export function getUploadState(): UploadState {
  return state;
}

/** Replace the queue with a new selection. Refused mid-run so a batch cannot be swapped underneath itself. */
export function enqueue(files: File[]): { skipped: string[]; duplicates: number; accepted: number } {
  if (state.running) return { skipped: [], duplicates: 0, accepted: 0 };
  const built = buildQueue(files.map((f) => ({ name: f.name, size: f.size, type: f.type })));
  blobs.clear();
  for (const f of files) blobs.set(`${f.name}:${f.size}`, f);
  state = { ...state, items: built.items };
  emit();
  return { skipped: built.skipped, duplicates: built.duplicates, accepted: built.items.length };
}

export function clearUploads(): void {
  if (state.running) return;
  blobs.clear();
  state = { items: [], running: false, phase: "idle", target: null };
  emit();
}

function patch(id: string, p: Partial<QueueItem>) {
  state = { ...state, items: patchItem(state.items, id, p) };
  emit();
}

// Registered only while a run is in flight. A permanent handler would nag on every navigation away from a
// page that has nothing to lose.
let guardArmed = false;
function beforeUnload(e: BeforeUnloadEvent) {
  e.preventDefault();
  e.returnValue = "";
}
function armGuard(on: boolean) {
  if (typeof window === "undefined" || on === guardArmed) return;
  if (on) window.addEventListener("beforeunload", beforeUnload);
  else window.removeEventListener("beforeunload", beforeUnload);
  guardArmed = on;
}

/**
 * Ask the browser for permission to raise a desktop notification.
 *
 * Requested when a run starts rather than on page load, because a permission prompt that appears before the
 * user has done anything is the one people deny reflexively, and a denial is permanent.
 */
export function requestNotifyPermission(): void {
  try {
    if (typeof Notification !== "undefined" && Notification.permission === "default") {
      void Notification.requestPermission();
    }
  } catch {
    /* a browser without the API is not an error, it just gets the in-app toast */
  }
}

function notifyDone(done: number, failed: number) {
  const msg = failed
    ? `${done} imported, ${failed} failed`
    : `${done} session${done === 1 ? "" : "s"} imported`;
  toast(`Upload finished: ${msg}`, failed ? "warn" : "success");
  try {
    // Only when the tab is hidden. A desktop notification for something the user is already looking at is
    // noise, and the toast has it covered.
    if (typeof document !== "undefined" && document.visibilityState === "hidden"
        && typeof Notification !== "undefined" && Notification.permission === "granted") {
      new Notification("LabeloxAV", { body: `Upload finished: ${msg}` });
    }
  } catch {
    /* notification failures must not affect the queue */
  }
}

async function runOne(item: QueueItem, deps: UploadDeps, target: NonNullable<UploadState["target"]>)
  : Promise<Partial<QueueItem>> {
  const file = blobs.get(item.id);
  if (!file) return { status: "error", detail: "file handle lost, re-select it" };

  let uri: string;
  try {
    patch(item.id, { status: "uploading", progress: 0 });
    uri = await deps.upload(file, (frac) => patch(item.id, { progress: frac }));
  } catch (e) {
    return { status: "error", detail: "upload failed: " + deps.humanizeError(e) };
  }

  let jobId: string;
  try {
    patch(item.id, { status: "importing", progress: 0 });
    const res = await deps.startImport({
      format: target.format === "auto" ? item.format : target.format,
      source_uri: uri, target_vehicle: target.vehicle, city: target.city,
    });
    jobId = res.job_id;
  } catch (e) {
    return { status: "error", detail: "import did not start: " + deps.humanizeError(e) };
  }

  let sessionId: string;
  try {
    sessionId = await deps.watchImport(jobId, (j) => patch(item.id, { progress: j.progress || 0 }));
  } catch (e) {
    return { status: "error", detail: "import failed: " + deps.humanizeError(e) };
  }

  // A session that imported with no frames is a real outcome rather than a failure, so it is recorded as
  // done with no frame to open instead of sending somebody to a 404.
  try {
    const { frame_id } = await deps.firstFrame(sessionId);
    return { status: "done", progress: 1, sessionId, frameId: frame_id };
  } catch {
    return { status: "done", progress: 1, sessionId, detail: "imported, no frames to open" };
  }
}

/**
 * Work the queue to completion. Idempotent: calling it while a run is in flight is a no-op, so a remount
 * cannot start a second loop over the same items.
 */
export async function startUploads(target: NonNullable<UploadState["target"]>, deps: UploadDeps)
  : Promise<void> {
  if (state.running || !state.items.length) return;
  state = { ...state, running: true, phase: "importing", target };
  emit();
  armGuard(true);
  requestNotifyPermission();

  try {
    for (;;) {
      const next = nextPending(state.items);
      if (!next) break;
      const result = await runOne(next, deps, target);
      patch(next.id, result);
    }
  } finally {
    state = { ...state, running: false, phase: "idle" };
    emit();
    armGuard(false);
  }

  const s = summarize(state.items);
  notifyDone(s.done, s.failed);
}

/**
 * Label every session the import produced, one after another.
 *
 * A second pass rather than a step inside the first, because they answer different questions. The import
 * has to finish before there is anything to label, and somebody may well want the clips in and nothing
 * else. Running it here rather than per session in another tab is what keeps "in order" true: autolabel is
 * GPU work on a machine with one slot, so firing 186 of them at once would queue them server-side anyway,
 * with no way to see where it had got to.
 *
 * Refused while the import is still running. Labelling a session while its sibling is still decoding puts
 * two GPU jobs in flight for no gain, and the ordering the caller asked for is the point.
 */
export async function startAutolabel(deps: UploadDeps): Promise<void> {
  if (state.running || !deps.startAutolabel || !deps.watchAutolabel) return;
  const targets = importedSessions(state.items);
  if (!targets.length) return;

  state = { ...state, running: true, phase: "autolabeling" };
  emit();
  armGuard(true);

  let labeled = 0;
  let failed = 0;
  try {
    for (const t of targets) {
      patch(t.id, { status: "autolabeling", progress: 0 });
      try {
        const { job_id } = await deps.startAutolabel(t.sessionId);
        const res = await deps.watchAutolabel(job_id, (j) => patch(t.id, { progress: j.progress || 0 }));
        patch(t.id, { status: "labeled", progress: 1, labeled: res?.objects ?? 0 });
        labeled++;
      } catch (e) {
        // The clip is still imported: the labelling failed, not the ingest. It goes back to `done` rather
        // than to `error`, because calling it an error would lose the session that plainly exists.
        patch(t.id, { status: "done", progress: 1, detail: "autolabel failed: " + deps.humanizeError(e) });
        failed++;
      }
    }
  } finally {
    state = { ...state, running: false, phase: "idle" };
    emit();
    armGuard(false);
  }

  toast(failed ? `Autolabel finished: ${labeled} labelled, ${failed} failed`
               : `Autolabel finished: ${labeled} session${labeled === 1 ? "" : "s"} labelled`,
        failed ? "warn" : "success");
}
