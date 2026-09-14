import { describe, expect, it } from "vitest";

import { addPoint, describe as describePrompt, undoPoint } from "./samPrompt";

// The behaviour that was missing, and the failure it produced.
//
// Every layer below supported multi-point prompts: SegmentIn carries points and labels, the endpoint
// refuses a request whose lengths differ, and the canvas has sent label 0 on shift-click from the start.
// The list itself did not exist. runSam committed the pending candidate before each click, so a
// shift-click sent one negative point with no positive anchor - a prompt saying only where the object is
// NOT. That returns nothing or an arbitrary region, which looks exactly like negative clicks being
// unsupported.
//
// The refusal is the part worth testing. A silently ignored click is invisible: the mask does not change
// and the annotator cannot tell whether they missed the object or the tool dropped the input.

describe("SAM prompt accumulation", () => {
  it("keeps every click, so the second refines the first rather than replacing it", () => {
    const a = addPoint(null, [10, 10], 1);
    expect(a.ok).toBe(true);
    if (!a.ok) return;
    const b = addPoint(a.prompt, [20, 20], 1);
    expect(b.ok).toBe(true);
    if (!b.ok) return;
    expect(b.prompt.points).toEqual([[10, 10], [20, 20]]);
    expect(b.prompt.labels).toEqual([1, 1]);
  });

  it("refuses a subtraction with nothing to subtract from, and says why", () => {
    const r = addPoint(null, [10, 10], 0);
    expect(r.ok).toBe(false);
    if (r.ok) return;
    expect(r.reason).toContain("click the object first");
  });

  it("accepts a negative click once there is a positive one to subtract from", () => {
    const a = addPoint(null, [10, 10], 1);
    if (!a.ok) return;
    const b = addPoint(a.prompt, [12, 12], 0);
    expect(b.ok).toBe(true);
    if (!b.ok) return;
    expect(b.prompt.labels).toEqual([1, 0]);
    expect(b.prompt.points).toHaveLength(2);
  });

  it("still refuses a negative when every click so far has been negative", () => {
    // Not reachable through the UI today, but the guard is on the prompt's contents rather than on the
    // click count, so it holds whatever order they arrive in.
    const only = { points: [[1, 1]], labels: [0] };
    expect(addPoint(only, [2, 2], 0).ok).toBe(false);
  });

  it("undo takes back the last click and stops at the first", () => {
    const a = addPoint(null, [10, 10], 1);
    if (!a.ok) return;
    const b = addPoint(a.prompt, [20, 20], 0);
    if (!b.ok) return;
    const back = undoPoint(b.prompt);
    expect(back?.points).toEqual([[10, 10]]);
    expect(back?.labels).toEqual([1]);
    // Undoing the only remaining click clears the prompt rather than leaving an empty one, because an
    // empty prompt and no prompt are the same thing and two spellings of it invite a bug.
    expect(undoPoint(back)).toBeNull();
    expect(undoPoint(null)).toBeNull();
  });

  it("describes the prompt the way it reads on screen", () => {
    expect(describePrompt(null)).toBe("");
    expect(describePrompt({ points: [[1, 1]], labels: [1] })).toBe("1 include");
    expect(describePrompt({ points: [[1, 1], [2, 2]], labels: [1, 0] })).toBe("1 include, 1 exclude");
  });
});
