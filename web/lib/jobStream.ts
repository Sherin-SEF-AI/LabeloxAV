// One connection to the jobs stream, shared by everything that watches work happen.
//
// `useEventStream` opens a new EventSource per hook call, and six pages call `useJobStream()`. That was
// tolerable while the callers were pages, since only one is mounted at a time. It stops being tolerable the
// moment the top bar wants the same data, because the top bar is on every page: every one of those six would
// hold two connections to the same endpoint carrying the same frames.
//
// So the stream moves to module scope with refcounted subscribers, which is the shape `toast.ts` and
// `uploadManager.ts` already use and for the same reason: a plain module can own something the whole app
// observes, and React subscribes rather than holds. One EventSource exists while at least one subscriber is
// attached and closes when the last leaves.
//
// The snapshot is kept so a late subscriber renders immediately instead of showing nothing until the server
// next pushes. The server pushes on change, so "next push" can be minutes on a quiet system, and a chip that
// is blank for the first two minutes after every navigation is indistinguishable from a broken one.

import { getUser } from "./user";
import type { JobStream } from "./useEventStream";

// Statuses that mean work is actually moving. `pending` and `queued-cloud` are not progress: this deployment
// carries 67 autolabel jobs pending since late June and 11 parked for a cloud A100, and a chip reporting 78
// running forever would be worse than no chip at all.
const ACTIVE = new Set(["running"]);
const WAITING = new Set(["pending", "queued", "queued-cloud"]);

type Listener = (s: JobStream | null) => void;

const listeners = new Set<Listener>();
let source: EventSource | null = null;
let snapshot: JobStream | null = null;
let connected = false;

/** The last frame the server sent, or null before the first one. */
export function getJobs(): JobStream | null {
  return snapshot;
}

export function jobsConnected(): boolean {
  return connected;
}

function open() {
  if (source || typeof window === "undefined") return;
  const token = getUser()?.token;
  // Without a token the stream 401s and EventSource would retry forever. Staying closed is the honest state:
  // a subscriber gets the null snapshot and renders nothing, which is correct on a signed-out page.
  if (!token) return;

  const es = new EventSource(`/api/events/jobs?token=${encodeURIComponent(token)}`);
  source = es;
  es.addEventListener("open", () => { connected = true; notify(); });
  es.addEventListener("jobs", (ev) => {
    try {
      snapshot = JSON.parse((ev as MessageEvent).data) as JobStream;
      connected = true;
      notify();
    } catch {
      // A malformed frame must not tear down a working stream; the next one is usually fine.
    }
  });
  es.addEventListener("error", () => {
    // EventSource reconnects on its own with backoff. Reopening here would stack duplicate connections,
    // which is the failure this module exists to prevent.
    connected = false;
    // Subscribers are told even though the data did not change, because losing the connection IS the change
    // for anything showing a live indicator. Without this a dropped stream keeps claiming to be live, since
    // the only thing that used to wake a subscriber was a frame and a dead stream sends none.
    notify();
  });
}

function notify() {
  listeners.forEach((fn) => fn(snapshot));
}

function closeIfIdle() {
  if (listeners.size > 0 || !source) return;
  source.close();
  source = null;
  connected = false;
  // The snapshot is deliberately kept. A page that unmounts and remounts during a navigation should not
  // flash empty on the way back, and a stale frame is corrected within one server push.
}

/**
 * Watch the jobs stream. Returns an unsubscribe.
 *
 * The listener fires immediately with the current snapshot, so a subscriber that attaches between pushes has
 * something to render.
 */
export function subscribeJobs(fn: Listener): () => void {
  listeners.add(fn);
  open();
  fn(snapshot);
  return () => {
    listeners.delete(fn);
    closeIfIdle();
  };
}

/** For tests, and for a sign-out that has to drop a connection authenticated as somebody else. */
export function resetJobStream(): void {
  source?.close();
  source = null;
  snapshot = null;
  connected = false;
  listeners.clear();
  summaryListeners.clear();
  summaryOff = null;
  summary = { running: [], waiting: 0, connected: false };
}

export type JobsSummary = {
  /** Jobs actually progressing, with their ids so a caller can avoid double-counting its own. */
  running: { job_id: string; kind: string; progress: number }[];
  /** Jobs the system holds but is not working, which is the answer to "why is nothing happening". */
  waiting: number;
  /**
   * Whether the event stream is actually up.
   *
   * Part of the summary rather than a separate subscription, because a chip showing "idle" while the
   * stream is down is not reporting the system, it is reporting its own silence. The comment on the
   * error handler above already says losing the connection IS the change for anything showing a live
   * indicator; this is what carries that to the indicator.
   */
  connected: boolean;
};

// The summary is a fan-out of the same stream rather than a second source. `subscribeJobs` replays its
// snapshot to each new subscriber, and before the first frame that snapshot is null, which means "no answer
// yet" and not "nothing is running": summarising it would blank the chip every time another subscriber
// attached.
let summary: JobsSummary = { running: [], waiting: 0, connected: false };
const summaryListeners = new Set<(s: JobsSummary) => void>();
// One jobs listener for the whole group, not one per summary subscriber. A listener each would summarise the
// same frame N times and deliver it N-squared times, which two subscribers already make visible.
let summaryOff: (() => void) | null = null;

/** Watch the running/waiting summary. Fires immediately with what is known. */
export function subscribeJobSummary(fn: (s: JobsSummary) => void): () => void {
  summaryListeners.add(fn);
  summaryOff ??= subscribeJobs((frame) => {
    // A null frame means no answer yet, so the counts are left alone - but the connection state is still
    // news, and dropping the whole callback is why a disconnect never reached this summary at all.
    summary = frame
      ? { ...summarizeJobs(frame), connected: jobsConnected() }
      : { ...summary, connected: jobsConnected() };
    summaryListeners.forEach((l) => l(summary));
  });
  fn(summary);
  return () => {
    summaryListeners.delete(fn);
    if (summaryListeners.size === 0) {
      summaryOff?.();
      summaryOff = null;
    }
  };
}

/** Flatten a stream frame into what a status chip needs.
 *
 * `connected` is read from the module rather than from the frame, because it is not a property of a frame:
 * the last frame received says nothing about whether the stream carrying it is still up. */
export function summarizeJobs(s: JobStream | null): JobsSummary {
  const running: JobsSummary["running"] = [];
  let windowWaiting = 0;
  if (!s) return { running, waiting: 0, connected: jobsConnected() };
  // `ingest` rides the same stream but is a progress summary rather than a list of jobs, so it has no job_id
  // to key on and is not a job for this purpose.
  for (const kind of ["import", "autolabel", "training", "export"] as const) {
    for (const r of s[kind] ?? []) {
      if (ACTIVE.has(r.status)) {
        running.push({ job_id: r.job_id, kind, progress: Math.max(0, Math.min(1, r.progress ?? 0)) });
      } else if (WAITING.has(r.status)) {
        windowWaiting++;
      }
    }
  }
  // The server's total when it sends one, since the lists above are a recent tail and counting them
  // under-reports. Counting the window is the fallback for a server that predates the field, and it is a
  // floor rather than an answer.
  const total = s.waiting
    ? Object.values(s.waiting).reduce((a: number, b) => a + (b ?? 0), 0)
    : windowWaiting;
  return { running, waiting: total, connected: jobsConnected() };
}
