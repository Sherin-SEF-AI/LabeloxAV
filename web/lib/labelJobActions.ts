// What an annotator can do to a labeling job, and why the board could not do any of it.
//
// The backend has the whole lifecycle: `set_state` moves a job within its stage, `submit_job` scores the
// honeypots and advances annotation to validation to acceptance, and `/api/labelops/my-jobs` answers what
// is assigned to you. All three have typed clients in `web/lib/api.ts` and, until now, no callers. So
// `LabelJob.state` sat at its default `new` forever, `stage` never advanced, and `honeypot_accuracy`, which
// only `submit_job` writes, was never computed. The board's four state columns and the scorecard's accuracy
// column could only ever render a dash: work could be created and assigned, and never finished.
//
// The rules live here rather than in the page because they decide whether somebody's work is submittable,
// which is worth testing without a browser.

export type LabelJobLike = {
  job_id: string;
  stage: string;
  state: string;
  assignee_id: string | null;
  version?: number;
};

export type JobAction =
  | { kind: "start"; label: string; hint: string }
  | { kind: "submit"; label: string; hint: string }
  | null;

// annotation -> validation -> acceptance, as `services/labelops/jobs.py` advances it on submit.
const NEXT_STAGE: Record<string, string> = {
  annotation: "validation",
  validation: "acceptance",
};

/**
 * The one action this viewer can take on this job, or null.
 *
 * Only the assignee acts. A job nobody holds is not startable, because starting it would record work
 * against no one and the honeypot score would have no annotator to attach to.
 */
export function nextAction(job: LabelJobLike, viewerId: string | null): JobAction {
  if (!viewerId || job.assignee_id !== viewerId) return null;
  switch (job.state) {
    case "new":
      return { kind: "start", label: "start",
               hint: "take this job and start the clock on it" };
    case "rejected":
      // Submitted, failed its own honeypot floor, and came back in the same stage to its author. Starting
      // again is the only way forward, and calling it "start" would hide that this is a second attempt.
      return { kind: "start", label: "retry",
               hint: "this came back below the honeypot floor; fix it and submit again" };
    case "in_progress":
      return { kind: "submit", label: "submit",
               hint: submitHint(job.stage) };
    default:
      return null;
  }
}

function submitHint(stage: string): string {
  const next = NEXT_STAGE[stage];
  return next
    ? `score the honeypots and move this to ${next}`
    : "score the honeypots and complete this job";
}

/** Where a submit sends the job, for a confirm that says what will happen. */
export function stageAfterSubmit(stage: string): string {
  return NEXT_STAGE[stage] ?? "acceptance";
}

/**
 * Whether a job's honeypot accuracy cleared the project floor.
 *
 * Null is not a failure, it is an unmeasured job. Rendering an unmeasured job as failing is the mistake
 * that made "0.00" look like a real score everywhere the number had simply never been computed.
 */
export function honeypotVerdict(accuracy: number | null | undefined, floor: number | null | undefined):
  "unmeasured" | "pass" | "fail" {
  if (accuracy == null) return "unmeasured";
  return accuracy >= (floor ?? 0) ? "pass" : "fail";
}
