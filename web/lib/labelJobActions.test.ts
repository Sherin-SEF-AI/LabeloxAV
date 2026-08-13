// A labeling job could be created and assigned, and never finished.
//
// `set_state`, `submit_job` and `/api/labelops/my-jobs` all exist, all have typed clients, and none had a
// caller. So `LabelJob.state` stayed at `new` forever, `stage` never advanced through annotation to
// validation to acceptance, and `honeypot_accuracy` (written only by `submit_job`) was never computed. The
// board's state columns and the scorecard's accuracy column could only render a dash.

import { describe, expect, it } from "vitest";

import { honeypotVerdict, nextAction, stageAfterSubmit } from "./labelJobActions";

const job = (over: Partial<Parameters<typeof nextAction>[0]> = {}) => ({
  job_id: "j1", stage: "annotation", state: "new", assignee_id: "me", ...over,
});

describe("who can act", () => {
  it("the assignee can", () => {
    expect(nextAction(job(), "me")?.kind).toBe("start");
  });

  it("somebody else cannot, even on an open job", () => {
    expect(nextAction(job({ assignee_id: "someone" }), "me")).toBeNull();
  });

  it("an unassigned job has no action", () => {
    // Starting it would record work against nobody, and the honeypot score would have no annotator to
    // attach to.
    expect(nextAction(job({ assignee_id: null }), "me")).toBeNull();
  });

  it("a signed-out viewer has no action", () => {
    expect(nextAction(job(), null)).toBeNull();
  });
});

describe("what the action is", () => {
  it("a new job starts", () => {
    expect(nextAction(job({ state: "new" }), "me")?.kind).toBe("start");
  });

  it("a job in progress submits", () => {
    expect(nextAction(job({ state: "in_progress" }), "me")?.kind).toBe("submit");
  });

  it("a rejected job is a retry, not a fresh start", () => {
    // It was submitted, scored below the project's honeypot floor, and came back in the same stage to its
    // author. Calling that "start" would hide that it is a second attempt.
    const a = nextAction(job({ state: "rejected" }), "me");
    expect(a?.kind).toBe("start");
    expect(a?.label).toBe("retry");
    expect(a?.hint).toContain("honeypot floor");
  });

  it("a completed job has nothing left", () => {
    expect(nextAction(job({ state: "completed" }), "me")).toBeNull();
  });

  it("says where a submit will send the job", () => {
    expect(nextAction(job({ state: "in_progress", stage: "annotation" }), "me")?.hint)
      .toContain("validation");
    expect(nextAction(job({ state: "in_progress", stage: "validation" }), "me")?.hint)
      .toContain("acceptance");
  });

  it("the last stage completes rather than naming a stage that does not exist", () => {
    expect(nextAction(job({ state: "in_progress", stage: "acceptance" }), "me")?.hint)
      .toContain("complete");
  });
});

describe("stageAfterSubmit", () => {
  it("walks annotation to validation to acceptance", () => {
    expect(stageAfterSubmit("annotation")).toBe("validation");
    expect(stageAfterSubmit("validation")).toBe("acceptance");
  });

  it("stops at acceptance rather than falling off the end", () => {
    expect(stageAfterSubmit("acceptance")).toBe("acceptance");
    expect(stageAfterSubmit("something-else")).toBe("acceptance");
  });
});

describe("honeypotVerdict", () => {
  it("an unmeasured job is not a failing job", () => {
    // Every job in this deployment is unmeasured, because nothing ever called submit. Rendering those as
    // failures would be a red board that means nothing.
    expect(honeypotVerdict(null, 0.9)).toBe("unmeasured");
    expect(honeypotVerdict(undefined, 0.9)).toBe("unmeasured");
  });

  it("clears the floor at exactly the floor", () => {
    expect(honeypotVerdict(0.9, 0.9)).toBe("pass");
  });

  it("fails below it", () => {
    expect(honeypotVerdict(0.89, 0.9)).toBe("fail");
  });

  it("a project with no floor passes anything measured", () => {
    expect(honeypotVerdict(0, null)).toBe("pass");
  });
});
