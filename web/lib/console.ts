// What the console shows, decided as data.
//
// The app could not answer "is anything happening" from inside itself. Job state lived on one page, the
// machine lived in a terminal, and background work lived in a log file. So the honest answer to a slow
// system was to open three things, two of which most people using this do not have.
//
// The rules here are the ones worth testing without a browser: what counts as work in flight, when a
// resource reading deserves attention, and how to say an uptime or a byte count in a way somebody can read
// at a glance.

import type { JobStream, SystemStream } from "./useEventStream";

export type WorkItem = {
  id: string;
  kind: string;
  status: string;
  /** 0..1, or null when the job reports no progress rather than zero progress. */
  progress: number | null;
  label: string;
};

const RUNNING = "running";
const WAITING = new Set(["pending", "queued", "queued-cloud"]);

/**
 * Everything the engine is actually doing, newest kind first.
 *
 * `ingest` rides the same stream but is a progress summary with no job id, so it is not a job for this
 * purpose and folding it in would key a list on undefined.
 */
export function workInFlight(s: JobStream | null): {
  running: WorkItem[]; waiting: WorkItem[]; waitingTotal: number;
} {
  const running: WorkItem[] = [];
  const waiting: WorkItem[] = [];
  if (!s) return { running, waiting, waitingTotal: 0 };
  for (const kind of ["training", "autolabel", "import", "export"] as const) {
    for (const r of s[kind] ?? []) {
      const item: WorkItem = {
        id: r.job_id, kind, status: r.status,
        // Null, not zero: a job that reports nothing has not reported zero, and a bar pinned at 0% reads as
        // stalled work rather than as unmeasured work.
        progress: typeof r.progress === "number" ? Math.max(0, Math.min(1, r.progress)) : null,
        label: kind,
      };
      if (r.status === RUNNING) running.push(item);
      else if (WAITING.has(r.status)) waiting.push(item);
    }
  }
  // The lists are a recent tail per kind, so counting them under-reports: 67 jobs parked for a GPU pod are
  // all older than the ten most recent. The server sends the true totals for exactly this reason, and the
  // top-bar chip already uses them. A console disagreeing with the chip about how much work is queued is
  // worse than either number alone.
  const waitingTotal = s.waiting
    ? Object.values(s.waiting).reduce((a: number, b) => a + (b ?? 0), 0)
    : waiting.length;
  return { running, waiting, waitingTotal };
}

export type Level = "ok" | "warn" | "critical";

/**
 * How worried to be about a fraction.
 *
 * Two thresholds rather than a gradient, because the only question a colour answers here is whether to look
 * now, look later, or not at all. The disk on this machine sits at 91%, which is exactly the reading that
 * should have been visible before an export failed rather than after.
 */
export function level(frac: number | null | undefined, warn = 0.8, critical = 0.92): Level {
  if (frac == null) return "ok";
  if (frac >= critical) return "critical";
  if (frac >= warn) return "warn";
  return "ok";
}

/** Bytes as a person reads them. Megabytes in, because that is what every source here reports. */
export function mb(v: number | null | undefined): string {
  if (v == null) return "-";
  if (v >= 1000) return `${(v / 1000).toFixed(1)} GB`;
  return `${Math.round(v)} MB`;
}

/** A duration a person reads at a glance, not a count of seconds. */
export function uptime(seconds: number | null | undefined): string {
  if (seconds == null) return "-";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

/**
 * The one-line answer to "is anything happening".
 *
 * A GPU held by somebody else is the single most common reason work will not start here, and it is
 * invisible from every other surface: the card can be full while this application is doing nothing at all.
 */
export function headline(jobs: JobStream | null, sys: SystemStream | null): string {
  const { running, waitingTotal } = workInFlight(jobs);
  if (running.length) {
    const kinds = [...new Set(running.map((r) => r.kind))].join(", ");
    return `${running.length} running (${kinds})`;
  }
  const gpu = sys?.gpus?.[0];
  const foreign = (gpu?.processes ?? []).filter((p) => !(p.name || "").includes("python"));
  if (foreign.length && (gpu?.memory_used_frac ?? 0) > 0.25) {
    return `idle, but ${foreign[0].name.split("/").pop()} holds ${mb(foreign[0].used_mb)} of the GPU`;
  }
  if (waitingTotal) return `nothing running, ${waitingTotal} queued`;
  return "idle";
}
