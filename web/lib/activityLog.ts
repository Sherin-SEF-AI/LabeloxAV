/**
 * A console transcript: what was run, what it returned, and what is still running.
 *
 * The agent console drives roughly fourteen operations from one page and reported all of them through a
 * single `msg` string, written by 64 call sites. Several of those operations are genuinely asynchronous on
 * the server - the error sweep, the overnight audit, drift investigation, fleet planning - and land minutes
 * after the click. So the second thing you start erases the first thing's result, and a background result
 * arriving late erases whatever you were reading.
 *
 * A toast is the wrong shape for the same reason: it auto-dismisses, and these results are the output of the
 * work, not an acknowledgement of it. A console wants a transcript.
 *
 * Kept as a pure reducer over an entry list rather than component state so ordering, the running/settled
 * transition and the cap are testable without mounting anything.
 */

export type ActivityStatus = "running" | "ok" | "failed";

export type ActivityEntry = {
  id: number;
  /** The operation, stable across its running -> settled transition. */
  action: string;
  status: ActivityStatus;
  message: string;
  /** Shown alongside a failure when the status justifies it. */
  hint?: string;
  startedAt: number;
  settledAt?: number;
};

/** Enough history to see a night's background work without growing without bound. */
export const MAX_ENTRIES = 50;

export type ActivityLog = { entries: ActivityEntry[]; seq: number };

export const emptyLog: ActivityLog = { entries: [], seq: 0 };

/**
 * Begin an operation. Returns the new log and the id to settle it with.
 *
 * Starting the same action twice is allowed and produces two entries. Collapsing them would hide a
 * double-click that fired two sweeps, which is exactly the thing an operator needs to see.
 */
export function begin(log: ActivityLog, action: string, at: number): { log: ActivityLog; id: number } {
  const id = log.seq + 1;
  const entry: ActivityEntry = { id, action, status: "running", message: "running", startedAt: at };
  return { log: { entries: cap([entry, ...log.entries]), seq: id }, id };
}

/**
 * Settle a running operation.
 *
 * An unknown id is ignored rather than appended. A late result whose entry has already been evicted by the
 * cap must not reappear at the top of the list as though it had just started.
 */
export function settle(log: ActivityLog, id: number, status: Exclude<ActivityStatus, "running">,
                       message: string, at: number, hint?: string): ActivityLog {
  let found = false;
  const entries = log.entries.map((e) => {
    if (e.id !== id) return e;
    found = true;
    return { ...e, status, message, hint, settledAt: at };
  });
  return found ? { ...log, entries } : log;
}

/** A result with no preceding `begin`, for something that completed without a visible start. */
export function record(log: ActivityLog, action: string, status: Exclude<ActivityStatus, "running">,
                       message: string, at: number, hint?: string): ActivityLog {
  const id = log.seq + 1;
  const entry: ActivityEntry = { id, action, status, message, hint, startedAt: at, settledAt: at };
  return { entries: cap([entry, ...log.entries]), seq: id };
}

export function clear(log: ActivityLog): ActivityLog {
  // The sequence is deliberately preserved: an in-flight operation settling after a clear must not match an
  // id minted later, or its result would land on an unrelated entry.
  return { entries: [], seq: log.seq };
}

export const isRunning = (log: ActivityLog, action: string): boolean =>
  log.entries.some((e) => e.action === action && e.status === "running");

export const runningCount = (log: ActivityLog): number =>
  log.entries.filter((e) => e.status === "running").length;

function cap(entries: ActivityEntry[]): ActivityEntry[] {
  if (entries.length <= MAX_ENTRIES) return entries;
  // Evict settled entries before running ones: a long-running job must not be dropped from the list while
  // it is still the thing the operator is waiting on.
  const kept: ActivityEntry[] = [];
  const settled: ActivityEntry[] = [];
  for (const e of entries) (e.status === "running" ? kept : settled).push(e);
  const room = Math.max(0, MAX_ENTRIES - kept.length);
  const keptSet = new Set([...kept, ...settled.slice(0, room)].map((e) => e.id));
  return entries.filter((e) => keptSet.has(e.id));
}
