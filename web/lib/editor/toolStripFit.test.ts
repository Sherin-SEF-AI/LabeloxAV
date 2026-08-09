// The strip promised one row with no clip and delivered one row that clips.
//
// Measured on the running editor: at 1600px the seven Objects groups need 928px and get 947px, so it fits.
// At 1280px they get 862px and `measure` scrolls out of sight; at 1100px they get 682px and both `cuboid`
// and `measure` go. There is no scrollbar and no chevron, so the strip silently claims those tools do not
// exist. The hotkeys still fire, which is worse rather than better: pressing R highlights a button nobody
// can see.
//
// These tests pin the arithmetic that decides what collapses, including the two cases where being clever
// would be wrong: everything fitting must not reserve room for a button that is never drawn, and an
// unmeasured strip must show everything rather than collapse to nothing.

import { describe, expect, it } from "vitest";

import { GAP_PX, OVERFLOW_BTN_PX, fitCount, splitGroups } from "./toolStripFit";

// The seven Objects-mode group widths as measured in the running editor at 1600px: select, Draw, AI assist,
// Mask edit, adverse, cuboid, measure. They need 837px including gaps. The strip offers 858px at 1600 (after
// the 83px mode prefix and its gap) and 773px at 1280, which is exactly why `measure` disappeared there.
const OBJECTS = [100, 110, 127, 136, 110, 103, 115];

describe("fitCount", () => {
  it("keeps every group when the row has room", () => {
    expect(fitCount(OBJECTS, 858)).toBe(OBJECTS.length);
  });

  it("does not reserve overflow space when nothing overflows", () => {
    // Exactly wide enough for all seven and not one pixel more. Reserving 34px for an overflow button that
    // will never be drawn would evict a group for no reason, which is the bug in reverse.
    const exact = OBJECTS.reduce((a, b) => a + b, 0) + GAP_PX * (OBJECTS.length - 1);
    expect(fitCount(OBJECTS, exact)).toBe(OBJECTS.length);
  });

  it("collapses the tail when the row is short", () => {
    const n = fitCount(OBJECTS, 593); // what the 1100px viewport leaves for groups
    expect(n).toBeLessThan(OBJECTS.length);
    expect(n).toBeGreaterThan(0);
  });

  it("drops more groups the narrower it gets", () => {
    expect(fitCount(OBJECTS, 400)).toBeLessThanOrEqual(fitCount(OBJECTS, 773));
    expect(fitCount(OBJECTS, 773)).toBeLessThanOrEqual(fitCount(OBJECTS, 858));
  });

  it("leaves room for the overflow button itself", () => {
    const avail = 700;
    const n = fitCount(OBJECTS, avail);
    const used = OBJECTS.slice(0, n).reduce((a, b) => a + b, 0) + GAP_PX * Math.max(0, n - 1);
    expect(used + OVERFLOW_BTN_PX).toBeLessThanOrEqual(avail);
  });

  it("shows everything before the container has been measured", () => {
    // First paint reports 0. Collapsing then would flash an empty strip on every mode switch.
    expect(fitCount(OBJECTS, 0)).toBe(OBJECTS.length);
    expect(fitCount(OBJECTS, -1)).toBe(OBJECTS.length);
  });

  it("shows everything when a group has not reported a width yet", () => {
    expect(fitCount([96, 0, 104], 200)).toBe(3);
  });

  it("can collapse every group when the row is impossibly narrow", () => {
    // Better an overflow button alone than a strip that overflows its container.
    expect(fitCount(OBJECTS, 40)).toBe(0);
  });

  it("handles an empty mode", () => {
    expect(fitCount([], 800)).toBe(0);
  });
});

describe("splitGroups", () => {
  it("preserves order across the split and loses nothing", () => {
    const keys = ["select", "draw", "ai", "mask", "adverse", "cuboid", "measure"];
    const { visible, hidden } = splitGroups(keys, OBJECTS, 593);
    expect([...visible, ...hidden]).toEqual(keys);
    expect(hidden.length).toBeGreaterThan(0);
  });

  it("hides nothing when everything fits", () => {
    const keys = ["select", "draw", "ai", "mask", "adverse", "cuboid", "measure"];
    expect(splitGroups(keys, OBJECTS, 858).hidden).toEqual([]);
  });
});
