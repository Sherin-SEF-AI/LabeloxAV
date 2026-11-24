import { describe, expect, it } from "vitest";

import { brushOps, traceStamps } from "./brush";

// The preview went from one Konva node per stamp to one shape. What must not have changed is what gets
// committed, and what the preview claims will be.

describe("what a stroke commits", () => {
  const stroke = [[10, 10], [14, 11], [30, 30]];

  it("is one stamp per sampled point, at the brush radius", () => {
    expect(brushOps(stroke, "brush", 14)).toEqual([
      { op: "add", center: [10, 10], radius: 14 },
      { op: "add", center: [14, 11], radius: 14 },
      { op: "add", center: [30, 30], radius: 14 },
    ]);
  });

  it("erases with the eraser and adds with anything else", () => {
    expect(brushOps(stroke, "eraser", 6).every((o) => o.op === "erase")).toBe(true);
    expect(brushOps(stroke, "brush", 6).every((o) => o.op === "add")).toBe(true);
  });

  it("commits a single click as a single stamp", () => {
    expect(brushOps([[5, 5]], "brush", 12)).toEqual([{ op: "add", center: [5, 5], radius: 12 }]);
  });

  it("commits nothing for an empty stroke", () => {
    expect(brushOps([], "brush", 12)).toEqual([]);
  });
});

describe("what the preview draws", () => {
  function recorder() {
    const calls: (string | number)[][] = [];
    return {
      calls,
      moveTo: (x: number, y: number) => { calls.push(["moveTo", x, y]); },
      arc: (x: number, y: number, r: number, a0: number, a1: number) => { calls.push(["arc", x, y, r, a0, a1]); },
    };
  }

  it("is the same circles the commit stamps: same centres, same radius", () => {
    const ctx = recorder();
    const stroke = [[10, 10], [30, 30]];
    traceStamps(ctx, stroke, 14);
    const arcs = ctx.calls.filter((c) => c[0] === "arc");
    expect(arcs.map((c) => [c[1], c[2]])).toEqual(brushOps(stroke, "brush", 14).map((o) => o.center));
    expect(arcs.every((c) => c[3] === 14 && c[4] === 0 && c[5] === Math.PI * 2)).toBe(true);
  });

  it("starts each circle as its own subpath, so nothing is filled between two stamps", () => {
    const ctx = recorder();
    traceStamps(ctx, [[10, 10], [100, 100]], 5);
    expect(ctx.calls.map((c) => c[0])).toEqual(["moveTo", "arc", "moveTo", "arc"]);
    // The moveTo lands on the circle's own start point, the angle-zero point on its right.
    expect(ctx.calls[2]).toEqual(["moveTo", 105, 100]);
  });

  it("draws nothing for an empty stroke", () => {
    const ctx = recorder();
    traceStamps(ctx, [], 5);
    expect(ctx.calls).toEqual([]);
  });
});
