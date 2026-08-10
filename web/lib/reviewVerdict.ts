// What a rapid-review keystroke sends to the server, and what taking it back sends.
//
// This lived inline in the page, where it could not be tested without rendering a component, and two bugs
// hid in it for exactly that reason.
//
// The first is that undo never worked. It re-posted `state: "needs_review"`, which is not an object state
// at all: it is a model-promotion verdict from the VERDYX gate. The database says so itself, through the
// `ck_object_state` check constraint, which admits only review, auto_accept, accepted, rejected, annotate
// and submitted. So every undo raised a CheckViolationError, the transaction rolled back, and the verdict
// stood. The schema defending itself is the only reason the corpus was not corrupted.
//
// What made it invisible is the page: the "undid X" success toast fires synchronously while the failure
// arrives later on a rejected promise, and the cursor moves back regardless. The annotator is told the
// reversal happened, sees an error toast they have no reason to connect to it, and moves on. In a corpus
// where 252 of 570,379 objects have ever been reviewed, a correction path that silently does nothing is
// expensive.
//
// The second is quieter. Undo restored the state and not the class, so taking back a reclassify would have
// left the object holding the new label with none of the intent behind it.

import type { TriageRow } from "@/lib/types";

// The page's own vocabulary for a keystroke. It differs from the API's action verb (`reclass` versus
// `reclassify`), and translating between the two is one of this module's jobs.
export type Verdict = "accept" | "reject" | "reclass" | "skip";

/** The state an object returns to when its verdict is taken back. */
export const UNREVIEWED_STATE = "review";

export interface ReviewPayload {
  reviewer: string;
  action: string;
  state?: string;
  class_name?: string;
  time_spent_ms: number;
}

/** What a verdict sends. `acceptState` differs by role, so it is passed in rather than resolved here. */
export function verdictPayload(
  verdict: Exclude<Verdict, "skip">,
  opts: { reviewer: string; acceptState: string; className?: string; timeSpentMs: number },
): ReviewPayload {
  return {
    reviewer: opts.reviewer,
    action: verdict === "accept" ? "confirm" : verdict === "reject" ? "reject" : "reclassify",
    state: verdict === "reject" ? "rejected" : opts.acceptState,
    class_name: opts.className,
    // Real elapsed time, so the productivity numbers this feeds are measurements rather than a constant.
    time_spent_ms: Math.max(0, opts.timeSpentMs),
  };
}

/** What taking a verdict back sends, so the object returns to the queue it came from. */
export function undoPayload(row: TriageRow, opts: { reviewer: string }): ReviewPayload {
  return {
    reviewer: opts.reviewer,
    // `revert` rather than `confirm`: the audit trail should say a verdict was withdrawn, not record a
    // second confirmation of something nobody confirmed. The server derives state from the action verb
    // only when no explicit state is given, and one is given here.
    action: "revert",
    state: UNREVIEWED_STATE,
    // Restore the class the object had before the verdict, so undoing a reclassify is a real reversal.
    // TriageRow carries the class as it was when the queue was built, which is that value.
    class_name: row.class_name,
    // Not a measurement of anybody's judgement, so it must not enter the throughput statistics as one.
    time_spent_ms: 0,
  };
}
