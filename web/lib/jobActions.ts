// What the jobs dashboard can do to a job, decided as data rather than in JSX.
//
// Two bugs live in this file's subject. The filter row was a hardcoded list of four kinds while the server
// emits seven, so export, map_fusion and relabel rows could be seen under "all" and never isolated. Deriving
// the list from the rows that actually arrived fixes that at the root: a job kind added to the API can no
// longer be invisible here, because nobody has to remember to add a string.
//
// And until now the page could only open a job, never stop one. There was exactly one cancel route in the
// whole API and it was training's, so clearing anything else meant an UPDATE against the database. For
// auto-label that was not an inconvenience: the start route refuses while any row reads `running`, so one
// unwanted job blocked every user of the deployment.

import type { JobRow } from "./types";

// Terminal, in the sense that nothing more will happen. Cancelling one would rewrite history rather than
// stop work.
const FINISHED = new Set(["done", "error", "canceled", "cancelled"]);

// Training is not in the unified cancel route: it is drained by a separate worker holding a GPU lease, so it
// cancels through its own endpoint, which asks the worker rather than writing the row.
const TRAINING = "training";

/**
 * The filter buttons to offer, derived from the rows in hand.
 *
 * Ordered by how many jobs of that kind exist, so the kinds somebody is actually running come first and a
 * deployment that never exports does not carry a permanent dead button.
 */
export function jobKinds(jobs: readonly JobRow[]): string[] {
  const counts = new Map<string, number>();
  for (const j of jobs) counts.set(j.kind, (counts.get(j.kind) ?? 0) + 1);
  const kinds = [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([k]) => k);
  return ["all", ...kinds];
}

/** Whether this job can still be stopped. */
export function canCancel(job: JobRow): boolean {
  return !FINISHED.has(job.status);
}

/** Where the cancel for this job lives. Training has its own, for the reason above. */
export function cancelPath(job: JobRow): string {
  return job.kind === TRAINING
    ? `/api/training/${job.job_id}/cancel`
    : `/api/jobs/${job.kind}/${job.job_id}/cancel`;
}

export type JobCounts = { running: number; queued: number; parked: number };

/**
 * What the header should say, keeping the three kinds of not-finished apart.
 *
 * They used to be one number called "active", which counted `pending`. This deployment holds 67 auto-label
 * jobs parked for a GPU pod that is not provisioned; reporting 68 active on an idle machine is how a status
 * line stops being read.
 */
export function jobCounts(jobs: readonly JobRow[]): JobCounts {
  let running = 0, queued = 0, parked = 0;
  for (const j of jobs) {
    if (j.status === "running") running++;
    else if (j.status === "queued-cloud") parked++;
    else if (j.status === "pending" || j.status === "queued") queued++;
  }
  return { running, queued, parked };
}
