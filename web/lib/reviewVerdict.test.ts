import { describe, expect, it } from "vitest";
import type { TriageRow } from "@/lib/types";
import { UNREVIEWED_STATE, undoPayload, verdictPayload } from "./reviewVerdict";

// The states the database will actually accept, from the `ck_object_state` check constraint on `object`.
// Duplicated deliberately: the point of these tests is to catch a client sending something outside this
// set, so reading the set from the client would defeat them. `submitted` is admitted by the constraint
// (the VLM QA path writes it) though it is not in core/schemas.py GateState.
const GATE_STATES = ["review", "auto_accept", "accepted", "rejected", "annotate", "submitted"];

// What GET /api/triage asks for (services/api/routers/triage.py, states default).
const TRIAGE_STATES = ["review", "annotate"];

const row = (over: Partial<TriageRow> = {}): TriageRow => ({
  object_id: "obj-1", frame_id: "frm-1", class_name: "rider", conf: 0.42,
  state: "review", why: "low conf", priority: 1, flags: [], session_id: "ses-1",
  ...over,
} as TriageRow);

describe("verdictPayload", () => {
  it("accepts into whatever state the role permits", () => {
    const p = verdictPayload("accept", { reviewer: "jo", acceptState: "accepted", timeSpentMs: 4200 });
    expect(p).toMatchObject({ action: "confirm", state: "accepted", time_spent_ms: 4200 });
  });

  it("rejects into rejected regardless of role", () => {
    const p = verdictPayload("reject", { reviewer: "jo", acceptState: "accepted", timeSpentMs: 10 });
    expect(p.state).toBe("rejected");
  });

  it("carries the new class on a reclassify", () => {
    const p = verdictPayload("reclass", {
      reviewer: "jo", acceptState: "accepted", className: "motorcycle", timeSpentMs: 900,
    });
    expect(p).toMatchObject({ action: "reclassify", class_name: "motorcycle" });
  });

  it("never sends a negative elapsed time", () => {
    // A clock adjustment mid-review would otherwise write a negative number into the throughput stats.
    expect(verdictPayload("accept", {
      reviewer: "jo", acceptState: "accepted", timeSpentMs: -5,
    }).time_spent_ms).toBe(0);
  });

  it("only ever emits a real object state", () => {
    for (const v of ["accept", "reject", "reclass"] as const) {
      const p = verdictPayload(v, { reviewer: "jo", acceptState: "accepted", timeSpentMs: 1 });
      expect(GATE_STATES).toContain(p.state);
    }
  });
});

describe("undoPayload", () => {
  it("returns the object to a state the queue actually asks for", () => {
    // The whole bug: an undone object has to come back, and it comes back only if its state is one of the
    // states triage queries.
    const p = undoPayload(row(), { reviewer: "jo" });
    expect(p.state).toBe(UNREVIEWED_STATE);
    expect(TRIAGE_STATES).toContain(p.state);
    expect(GATE_STATES).toContain(p.state);
  });

  it("restores the class, so undoing a reclassify is a real reversal", () => {
    expect(undoPayload(row({ class_name: "rider" }), { reviewer: "jo" }).class_name).toBe("rider");
  });

  it("records a withdrawal rather than a second confirmation", () => {
    expect(undoPayload(row(), { reviewer: "jo" }).action).toBe("revert");
  });

  it("contributes no time to the throughput statistics", () => {
    expect(undoPayload(row(), { reviewer: "jo" }).time_spent_ms).toBe(0);
  });
});

describe("the behaviour this replaced", () => {
  // What the page sent before, kept executable so the regression stays demonstrable rather than becoming
  // a line in a commit message.
  function oldUndoPayload(reviewer: string) {
    return { reviewer, action: "confirm", state: "needs_review", time_spent_ms: 0 };
  }

  it("sent a state the database refuses, so every undo failed", () => {
    // "needs_review" is a VERDYX model-promotion verdict, not an object state. The `ck_object_state` check
    // constraint rejects it, the transaction rolls back, and the verdict the annotator meant to withdraw
    // stands. Verified live against the running API: the old payload returns 500, the new one 200.
    const old = oldUndoPayload("jo");
    expect(GATE_STATES).not.toContain(old.state);
    expect(GATE_STATES).toContain(undoPayload(row(), { reviewer: "jo" }).state);
  });

  it("would not have returned the object to the queue even had it been accepted", () => {
    expect(TRIAGE_STATES).not.toContain(oldUndoPayload("jo").state);
    expect(TRIAGE_STATES).toContain(undoPayload(row(), { reviewer: "jo" }).state);
  });

  it("left a reclassified object holding the class it was reclassified to", () => {
    expect(oldUndoPayload("jo")).not.toHaveProperty("class_name");
    expect(undoPayload(row({ class_name: "cattle" }), { reviewer: "jo" }).class_name).toBe("cattle");
  });
});
