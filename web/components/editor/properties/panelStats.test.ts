import { describe, expect, it } from "vitest";

import type { EdObject } from "../useEditor";
import {
  LOW_CONF, lowConfCount, qualityTone, reviewCounts, reviewWidths,
} from "./panelStats";

// The panel header prints a review ratio next to a three-segment bar, and the page footer prints its own
// "N confirmed" from a separate expression. The tests that matter here are the ones that keep those two
// from drifting, and the ones that pin the bar to states the database actually admits.
//
// The full legal set, from the ck_object_state check constraint quoted in lib/reviewVerdict.ts:
// review, auto_accept, accepted, rejected, annotate, submitted. Every one of them is exercised below,
// because the bucketing is a two-branch if/else and the third bucket is derived: a state nobody thought
// about lands in `open`, which is the safe place for it, and this suite is what proves it.

const STATES = ["review", "auto_accept", "accepted", "rejected", "annotate", "submitted"] as const;

function obj(over: Partial<EdObject> = {}): EdObject {
  return {
    id: "a", class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, ...over,
  };
}

describe("reviewCounts", () => {
  it("counts accepted as confirmed and auto_accept separately", () => {
    // The distinction the whole bar exists for: both are green in StateBadge, but only one had a person
    // rule on it.
    const c = reviewCounts([
      obj({ state: "accepted" }), obj({ state: "accepted" }),
      obj({ state: "auto_accept" }),
      obj({ state: "review" }),
    ]);
    expect(c).toEqual({ total: 4, confirmed: 2, auto: 1, open: 1 });
  });

  it("puts every non-accepted state in open, including ones nobody planned for", () => {
    for (const state of STATES) {
      const c = reviewCounts([obj({ state })]);
      const bucket = state === "accepted" ? "confirmed" : state === "auto_accept" ? "auto" : "open";
      expect({ state, n: c[bucket] }).toEqual({ state, n: 1 });
    }
    // A state the server could add tomorrow must not vanish from the total.
    expect(reviewCounts([obj({ state: "needs_review" })])).toEqual(
      { total: 1, confirmed: 0, auto: 0, open: 1 });
  });

  it("always sums to the total", () => {
    const objects = STATES.map((state) => obj({ state }));
    const c = reviewCounts(objects);
    expect(c.confirmed + c.auto + c.open).toBe(c.total);
    expect(c.total).toBe(STATES.length);
  });

  it("returns zeros for an empty frame rather than dividing by it", () => {
    expect(reviewCounts([])).toEqual({ total: 0, confirmed: 0, auto: 0, open: 0 });
  });
});

describe("reviewWidths", () => {
  it("gives an empty frame zero-width segments instead of NaN", () => {
    // NaN in a style attribute does not throw; the browser keeps whatever width it last had, so the bar
    // renders full and the panel claims a frame with no objects is fully confirmed.
    const w = reviewWidths(reviewCounts([]));
    expect(w).toEqual({ confirmed: 0, auto: 0, open: 0 });
    expect(Number.isNaN(w.confirmed)).toBe(false);
  });

  it("spans exactly 100% across the three segments", () => {
    const w = reviewWidths(reviewCounts([
      obj({ state: "accepted" }), obj({ state: "auto_accept" }),
      obj({ state: "review" }), obj({ state: "rejected" }),
    ]));
    expect(w.confirmed + w.auto + w.open).toBeCloseTo(100, 10);
    expect(w).toEqual({ confirmed: 25, auto: 25, open: 50 });
  });
});

describe("lowConfCount", () => {
  it("is strictly below the threshold, so an object at exactly 0.5 is not low", () => {
    // The selection chip labelled "conf < 0.5" passes 0.5 as its value and the reducer compares with <.
    // An inclusive compare here makes the header badge and the chip disagree by however many objects sit
    // on the boundary, which on a calibrated model is not a rare tie.
    expect(LOW_CONF).toBe(0.5);
    expect(lowConfCount([obj({ conf: 0.5 })])).toBe(0);
    expect(lowConfCount([obj({ conf: 0.49999 })])).toBe(1);
  });

  it("counts only the objects under the threshold", () => {
    expect(lowConfCount([obj({ conf: 0.1 }), obj({ conf: 0.9 }), obj({ conf: 0.2 })])).toBe(2);
  });

  it("treats a missing confidence as certain, matching the reducer", () => {
    expect(lowConfCount([obj({ conf: undefined as unknown as number })])).toBe(0);
  });
});

describe("qualityTone", () => {
  it("bands on the same thresholds the badge and the row dot both used", () => {
    expect(qualityTone(0.4)).toBe("good");
    expect(qualityTone(0.39999)).toBe("weak");
    expect(qualityTone(0.25)).toBe("weak");
    expect(qualityTone(0.24999)).toBe("bad");
    expect(qualityTone(0)).toBe("bad");
    expect(qualityTone(1)).toBe("good");
  });
});
