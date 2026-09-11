// Seen in a recording of the console, on every drift and lift row:
//
//   drift_investigator  committed  findings [object Object],[object Object] · hypothesis ...
//   pseudo_lift         committed  sessions [object Object],[object Object] · child_runs ...
//
// `AgentRunRow.counts` is `Record<string, unknown>`, written as free-form JSON by whichever agent
// produced the run, and several agents put a list of objects there. The panel interpolated the value
// straight into a template string, which is defined to call `String()` on it, and `String()` of an
// object is `[object Object]`. Nothing threw and nothing logged; the console simply displayed a
// placeholder where a number belonged, for as long as the panel has existed.
//
// The fix reports what is true about each kind of value rather than trusting it to be a number.

import { describe, expect, it } from "vitest";

import { countText } from "./Panels";

describe("countText", () => {
  it("reports a list by its length, which is what the row was missing", () => {
    expect(countText("findings", [{ a: 1 }, { b: 2 }])).toBe("findings 2");
    expect(countText("sessions", [])).toBe("sessions 0");
  });

  it("reports a nested object by how many keys it has", () => {
    expect(countText("by_state", { review: 12, accepted: 3 })).toBe("by_state 2");
  });

  it("leaves ordinary values exactly as they were", () => {
    expect(countText("clouds", 64)).toBe("clouds 64");
    expect(countText("date", "2026-09-11")).toBe("date 2026-09-11");
    expect(countText("failed", 0)).toBe("failed 0");
  });

  it("never renders the placeholder, for any shape a count can take", () => {
    const shapes: unknown[] = [[{ a: 1 }], { a: 1 }, [[1]], null, undefined, 0, "", false];
    for (const v of shapes) {
      expect(countText("k", v)).not.toContain("[object Object]");
    }
  });
});
