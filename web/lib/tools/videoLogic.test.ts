import { describe, expect, it } from "vitest";

import type { FanoutPreview } from "./videoApi";
import {
  defaultTicks, fanoutCount, fanoutMessage, nextSweepTarget, sweepQueue, toggleTick, trackRequestForClick,
} from "./videoLogic";

const obj = (id: string, x: number, state = "review", extra: Record<string, unknown> = {}) =>
  ({ id, state, bbox: [x, 10, x + 20, 30], ...extra });

describe("sweepQueue", () => {
  it("lists undecided machine labels left to right", () => {
    const q = sweepQueue([obj("c", 300), obj("a", 10), obj("h", 50, "accepted"), obj("b", 100, "annotate"),
                          obj("n", 5, "review", { isNew: true }), obj("s", 60, "submitted")]);
    expect(q).toEqual(["a", "b", "c"]);
  });

  it("drops what this sweep already decided and what is hidden", () => {
    const q = sweepQueue([obj("a", 10), obj("b", 20, "auto_accept"), obj("c", 30, "review", { visible: false })],
                         new Set(["a"]));
    expect(q).toEqual(["b"]);
  });
});

describe("nextSweepTarget", () => {
  it("starts at the head and steps forward, wrapping round to what was skipped", () => {
    expect(nextSweepTarget(["a", "b", "c"], null)).toBe("a");
    expect(nextSweepTarget(["a", "b", "c"], "b")).toBe("c");
    expect(nextSweepTarget(["a", "b", "c"], "c")).toBe("a");
  });

  it("goes to the head when the current one was decided and left the queue", () => {
    expect(nextSweepTarget(["b", "c"], "a")).toBe("b");
    expect(nextSweepTarget([], "a")).toBeNull();
  });
});

describe("trackRequestForClick", () => {
  const base = { frameId: "f2", pt: [10.4, 20], label: 1, className: "bus", before: 10, after: 10 };

  it("starts a new track of the class in hand", () => {
    expect(trackRequestForClick({ ...base, last: null })).toEqual({
      frame_id: "f2", class_name: "bus", before: 10, after: 10,
      prompts: [{ frame_id: "f2", points: [[10.4, 20]], labels: [1] }],
    });
  });

  it("re-prompts the last track when the click is inside its window", () => {
    const r = trackRequestForClick({ ...base, label: 0, last: { track_id: "t1", window_frame_ids: ["f1", "f2"] } });
    expect(r).toMatchObject({ track_id: "t1", prompts: [{ labels: [0] }] });
    expect(r).not.toHaveProperty("class_name");
  });

  it("refuses to start from a negative click or without a class", () => {
    expect(trackRequestForClick({ ...base, label: 0, last: null })).toBeNull();
    expect(trackRequestForClick({ ...base, className: null, last: { track_id: "t", window_frame_ids: ["x"] } }))
      .toBeNull();
  });
});

describe("fan-out ticks", () => {
  const row = (object_id: string, selectable: boolean) => ({
    object_id, selectable, frame_id: "f", class_name: "x", state: "review", source: "fused", conf: 0.5,
    track_id: null, score: null, crop_url: "", human: !selectable,
  });
  const p: FanoutPreview = {
    object_id: "o", from_class: "autorickshaw", to_class: "truck",
    track: { track_id: "t", objects: [row("t1", true), row("t2", false)] },
    lookalikes: { objects: [row("l1", true)], reason: null, threshold: 0.85 },
    confusion: { objects: [row("c1", true)], rule: "" },
  };

  it("ticks every selectable candidate and never a person's label", () => {
    const t = defaultTicks(p);
    expect(t).toEqual({ track: ["t1"], lookalikes: ["l1"], confusion: ["c1"] });
    expect(fanoutCount(t)).toBe(3);
    expect(toggleTick(t, "track", "t1").track).toEqual([]);
    expect(toggleTick(t, "confusion", "c9").confusion).toEqual(["c1", "c9"]);
  });

  it("says what was left alone", () => {
    expect(fanoutMessage({ track: 2, others: 1, skipped_human: ["h"], class_name: "truck" }))
      .toBe("3 relabelled truck; 2 on the track; 1 look-alike or same-confusion; 1 left alone (a person's label, "
            + "refused by the ontology, or changed meanwhile)");
  });
});
