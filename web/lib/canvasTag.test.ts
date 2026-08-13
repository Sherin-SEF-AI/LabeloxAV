// "Which of these forty boxes is the model unsure about" could only be answered by clicking forty boxes.

import { describe, expect, it } from "vitest";

import { LOW_CONF, MIN_TAG_BOX_PX, objectTag } from "./canvasTag";

const box = (over: Partial<Parameters<typeof objectTag>[0]> = {}) =>
  ({ class_name: "car", conf: 0.87, state: "review", ...over });

describe("what a box says on the canvas", () => {
  it("shows the class and the model's confidence as a whole number", () => {
    expect(objectTag(box(), 200)).toMatchObject({ show: true, text: "car 87" });
  });

  it("marks a detection below the review threshold, which is the one worth opening first", () => {
    expect(objectTag(box({ conf: LOW_CONF - 0.01 }), 200).low).toBe(true);
    expect(objectTag(box({ conf: LOW_CONF + 0.01 }), 200).low).toBe(false);
  });

  it("drops the number once a human has settled the object", () => {
    // The model's guess is history at that point, and leaving it up invites arguing with a decision that
    // has already been made.
    expect(objectTag(box({ state: "accepted", conf: 0.31 }), 200)).toMatchObject({
      text: "car", low: false,
    });
    expect(objectTag(box({ state: "rejected", conf: 0.31 }), 200).low).toBe(false);
  });

  it("says nothing on a box too small to hold the words", () => {
    // A tag wider than its box covers the object it is describing, which is the opposite of the point.
    expect(objectTag(box(), MIN_TAG_BOX_PX - 1).show).toBe(false);
    expect(objectTag(box(), MIN_TAG_BOX_PX).show).toBe(true);
  });

  it("says nothing on a box still being drawn", () => {
    expect(objectTag(box({ isNew: true }), 300).show).toBe(false);
  });

  it("survives a missing or absurd confidence rather than printing NaN on the canvas", () => {
    expect(objectTag(box({ conf: Number.NaN }), 200).text).toBe("car 0");
    expect(objectTag(box({ conf: 4 }), 200).text).toBe("car 100");
    expect(objectTag(box(), Number.NaN).show).toBe(false);
  });

  it("takes a threshold, so a workspace with a different cut can pass its own", () => {
    expect(objectTag(box({ conf: 0.7 }), 200, 0.8).low).toBe(true);
  });
});
