// A reviewer stepping through a session could not tell a finished frame from an untouched one.

import { describe, expect, it } from "vitest";

import { tileProgress } from "./filmstrip";

describe("what a filmstrip tile says about its frame", () => {
  it("calls a frame nobody has confirmed untouched", () => {
    expect(tileProgress(6, 0)).toMatchObject({ state: "untouched", frac: 0 });
  });

  it("calls a fully confirmed frame done", () => {
    expect(tileProgress(6, 6)).toMatchObject({ state: "done", frac: 1 });
  });

  it("shows a half-finished frame as partial, which is where somebody was interrupted", () => {
    // The state a boolean hides, and the one worth stepping back to.
    expect(tileProgress(6, 3)).toMatchObject({ state: "partial", frac: 0.5 });
  });

  it("distinguishes a frame with nothing in it from one nobody has touched", () => {
    // An empty frame is finished by having nothing to do, so marking it untouched would send a reviewer
    // back to a frame that has no work in it.
    expect(tileProgress(0, 0).state).toBe("empty");
  });

  it("never draws a bar past the end of the tile", () => {
    expect(tileProgress(3, 9).frac).toBe(1);
  });

  it("survives counts that make no sense rather than rendering NaN", () => {
    expect(tileProgress(-2, -5)).toMatchObject({ state: "empty", frac: 0 });
    expect(tileProgress(4, 1.7).frac).toBeCloseTo(0.25);
  });

  it("says how far along it is in words, since the bar alone is not readable", () => {
    expect(tileProgress(8, 2).label).toBe("2 of 8 confirmed");
    expect(tileProgress(8, 8).label).toBe("8 objects, all confirmed");
  });
});
