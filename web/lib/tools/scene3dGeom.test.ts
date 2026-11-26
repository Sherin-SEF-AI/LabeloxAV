import { describe, expect, it } from "vitest";

import {
  LabelHistory, applyClass, brushInto, classRgb, combineMask, countByClass, countMask, groundNear, mergeSuggestion,
  pointInPolygon, selectPolygon, selectRect, voxelBox, voxelColumn, voxelOf,
} from "./scene3dGeom";

const screen = new Float32Array([10, 10, 50, 50, 90, 90, NaN, NaN, 50, 10]);

describe("selection", () => {
  it("tests polygons with the even-odd rule", () => {
    const sq: [number, number][] = [[0, 0], [100, 0], [100, 100], [0, 100]];
    expect(pointInPolygon(50, 50, sq)).toBe(true);
    expect(pointInPolygon(150, 50, sq)).toBe(false);
    const tri: [number, number][] = [[0, 0], [100, 0], [0, 100]];
    expect(Array.from(selectPolygon(screen, tri))).toEqual([1, 0, 0, 0, 1]);
    expect(countMask(selectPolygon(screen, [[0, 0], [1, 1]]))).toBe(0);
  });
  it("selects rectangles in any drag direction and skips off-screen points", () => {
    expect(Array.from(selectRect(screen, [60, 60], [0, 0]))).toEqual([1, 1, 0, 0, 1]);
  });
  it("brushes cumulatively", () => {
    const m = new Uint8Array(5);
    expect(brushInto(m, screen, 10, 10, 5)).toBe(1);
    expect(brushInto(m, screen, 50, 30, 25)).toBe(2);
    expect(brushInto(m, screen, 50, 30, 25)).toBe(0);
    expect(Array.from(m)).toEqual([1, 1, 0, 0, 1]);
  });
  it("adds and subtracts selections", () => {
    const a = new Uint8Array([1, 1, 0, 0]), b = new Uint8Array([0, 1, 1, 0]);
    expect(Array.from(combineMask(a, b, "add"))).toEqual([1, 1, 1, 0]);
    expect(Array.from(combineMask(a, b, "subtract"))).toEqual([1, 0, 0, 0]);
    expect(combineMask(a, b, "replace")).toBe(b);
    expect(combineMask(null, b, "add")).toBe(b);
  });
});

describe("labels", () => {
  it("applies a class, respecting existing work when asked", () => {
    const l = new Int16Array([-1, 5, -1, 7]);
    const m = new Uint8Array([1, 1, 0, 1]);
    expect(applyClass(l, m, 9, true)).toBe(1);
    expect(Array.from(l)).toEqual([9, 5, -1, 7]);
    expect(applyClass(l, m, -1)).toBe(3);
    expect(Array.from(l)).toEqual([-1, -1, -1, -1]);
  });
  it("merges suggestions without erasing", () => {
    const l = new Int16Array([3, -1, -1]);
    expect(mergeSuggestion(l, new Int16Array([8, 8, -1]), true)).toBe(1);
    expect(Array.from(l)).toEqual([3, 8, -1]);
    expect(mergeSuggestion(l, new Int16Array([8, -1, -1]), false)).toBe(1);
    expect(countByClass(l).get(8)).toBe(2);
  });
  it("undoes to earlier snapshots and stays bounded", () => {
    const h = new LabelHistory(2);
    const l = new Int16Array([1]);
    h.push(l); l[0] = 2; h.push(l); l[0] = 3; h.push(l);
    expect(h.size).toBe(2);
    expect(h.undo()?.[0]).toBe(3);
    expect(h.undo()?.[0]).toBe(2);
    expect(h.undo()).toBeNull();
  });
  it("colours classes stably and leaves unlabelled to the caller", () => {
    expect(classRgb(-1)).toBeNull();
    expect(classRgb(88)).toEqual(classRgb(88));
    expect(classRgb(88)).not.toEqual(classRgb(44));
    for (const c of classRgb(12)!) expect(c).toBeGreaterThanOrEqual(0);
  });
});

describe("voxels", () => {
  const origin = [-20, -30, -2], dims = [160, 120, 12];
  it("finds the voxel of a point and refuses the outside", () => {
    expect(voxelOf([0, 0, 0], origin, 0.5, dims)).toEqual([40, 60, 4]);
    expect(voxelOf([-21, 0, 0], origin, 0.5, dims)).toBeNull();
  });
  it("builds clipped boxes and columns", () => {
    expect(voxelBox([1, 1], [-1, -1], 5, 2, origin, 0.5, dims)).toEqual({ lo: [38, 58, 2], hi: [42, 62, 5] });
    expect(voxelBox([-100, -100], [100, 100], 0, 99, origin, 0.5, dims)).toEqual({ lo: [0, 0, 0], hi: [159, 119, 11] });
    expect(voxelColumn(3, 4, 2, 0)).toEqual([[3, 4, 0], [3, 4, 1], [3, 4, 2]]);
  });
  it("estimates the ground near a point", () => {
    const pts = new Float32Array([0, 0, -1.7, 0, 0.5, 0, -1.72, 0, 1, 0, -1.69, 0, 0, 1, -1.71, 0, 0, 0, 0.5, 0, 0.2, 0.2, 1.0, 0]);
    expect(groundNear(pts, 6, 0, 0)).toBeCloseTo(-1.72, 2);
    expect(groundNear(pts, 6, 50, 50)).toBeNull();
  });
});
