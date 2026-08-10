// Measured on the running app: /analytics shows "-" in all six KPI cards, "no objects yet" in class
// distribution and label-source mix, "no scenarios mined", and "no reviews yet" for roughly ten seconds
// after load. Then /api/analytics/overview returns 36,905 frames and 570,505 objects and every one of those
// statements is replaced. /review/queue gets this right with skeletons; analytics asserts instead.
//
// The tests below are mostly about the two directions this can fail. Asserting empty while loading is the
// bug being fixed. Blanking a panel that already has data because some other endpoint on the page is still
// in flight would be the overcorrection, and on this page it would blank almost everything, since it loads
// eleven endpoints with very different latencies.

import { describe, expect, it } from "vitest";

import { PENDING, placeholder, statValue, titleCount } from "./emptyState";

describe("placeholder", () => {
  it("does not claim emptiness while the request is open", () => {
    const p = placeholder(true, 0, "no objects yet");
    expect(p?.kind).toBe("loading");
    expect(p?.text).not.toContain("no objects");
  });

  it("says empty once the answer is known to be nothing", () => {
    expect(placeholder(false, 0, "no objects yet")).toEqual({ kind: "empty", text: "no objects yet" });
  });

  it("draws the data when there is data", () => {
    expect(placeholder(false, 12, "no objects yet")).toBeNull();
  });

  it("keeps drawing data that has already arrived while the page finishes loading", () => {
    // Analytics fires eleven requests. Class distribution returns long before the cluster map, and blanking
    // it until the slowest one lands would make the page emptier than it is.
    expect(placeholder(true, 12, "no objects yet")).toBeNull();
  });

  it("lets a caller word the wait", () => {
    expect(placeholder(true, 0, "no reviews yet", "counting reviews")?.text).toBe("counting reviews");
  });
});

describe("statValue", () => {
  it("shows the value once it is known", () => {
    expect(statValue(false, 36905)).toBe("36905");
    expect(statValue(true, 36905)).toBe("36905");
  });

  it("does not print a dash for a number it has not fetched", () => {
    // "-" in a KPI card reads as measured-and-zero, which is exactly the wrong reading on a 570k corpus.
    expect(statValue(true, null)).toBe(PENDING);
    expect(statValue(true, undefined)).toBe(PENDING);
  });

  it("prints the absent marker once loading is done and there is still nothing", () => {
    expect(statValue(false, null)).toBe("-");
    expect(statValue(false, null, "none")).toBe("none");
  });

  it("treats a real zero as a value, not as absence", () => {
    // scenarios is genuinely 0 on this corpus. It must render as 0, never as "-" or "...".
    expect(statValue(false, 0)).toBe("0");
    expect(statValue(true, 0)).toBe("0");
  });
});

describe("titleCount", () => {
  it("does not put a zero in a title before the count is known", () => {
    expect(titleCount(true, 0, false)).toBe(PENDING);
  });

  it("states the count once it is known", () => {
    expect(titleCount(false, 152, true)).toBe("152");
    expect(titleCount(true, 152, true)).toBe("152");
  });

  it("states a known zero", () => {
    expect(titleCount(false, 0, true)).toBe("0");
  });
});
