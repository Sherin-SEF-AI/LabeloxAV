// A SAM-segmented car arrives as several hundred vertices and the editor drew a draggable handle on every
// one, so selecting it buried the object under a solid band of circles.

import { describe, expect, it } from "vitest";

import { handleIndices, simplifyMask, simplifyPolygon } from "./simplify";

/** A straight edge sampled at pixel resolution, which is what a traced mask looks like. */
const denseLine = (n: number) => Array.from({ length: n }, (_, i) => [i, 0]).flat();

describe("simplifying a traced polygon", () => {
  it("collapses a straight run of vertices to its endpoints", () => {
    // 200 points on one line describe exactly what 2 points describe.
    expect(simplifyPolygon([...denseLine(200), 199, 50, 0, 50], 1).length).toBeLessThan(20);
  });

  it("keeps corners, which are the shape", () => {
    const square = [0, 0, 100, 0, 100, 100, 0, 100];
    expect(simplifyPolygon(square, 1)).toEqual(square);
  });

  it("keeps a deviation larger than the tolerance and drops one smaller", () => {
    const withBump = [0, 0, 50, 3, 100, 0, 100, 100, 0, 100];
    expect(simplifyPolygon(withBump, 1)).toContain(3);
    expect(simplifyPolygon(withBump, 5)).not.toContain(3);
  });

  it("never returns less than a triangle", () => {
    // A tolerance large enough to collapse the whole shape must give the shape back, not a line.
    const out = simplifyPolygon([0, 0, 100, 0, 100, 100, 0, 100], 10_000);
    expect(out.length).toBeGreaterThanOrEqual(6);
  });

  it("leaves a polygon that is already minimal alone", () => {
    expect(simplifyPolygon([0, 0, 10, 0, 5, 9], 1)).toEqual([0, 0, 10, 0, 5, 9]);
  });

  it("treats the closing seam as a corner rather than flattening it", () => {
    // Opening the ring at vertex 0 and simplifying the open path would let the first corner be dropped,
    // which puts a visible notch in the object exactly where the polygon starts.
    const square = [0, 0, 100, 0, 100, 100, 0, 100];
    expect(simplifyPolygon(square, 2)).toContain(0);
    expect(simplifyPolygon(square, 2).length).toBe(8);
  });

  it("does nothing when the tolerance is zero, so a caller can opt out", () => {
    const dense = denseLine(50);
    expect(simplifyPolygon(dense, 0)).toEqual(dense);
  });

  it("gives back a degenerate ring untouched rather than reducing it below a shape", () => {
    // A traced ring that is a straight line has no area to preserve, and simplifying it to two points
    // would hand the caller something that is no longer a polygon.
    expect(simplifyPolygon(denseLine(100), 1)).toEqual(denseLine(100));
  });

  it("simplifies every ring of a multi-part mask", () => {
    const mask = [[...denseLine(100), 99, 40, 0, 40], [0, 0, 5, 0, 5, 5, 0, 5]];
    const out = simplifyMask(mask, 1);
    expect(out).toHaveLength(2);
    expect(out[0].length).toBeLessThan(mask[0].length);
    expect(out[1]).toEqual(mask[1]);
  });
});

describe("which vertices get a handle", () => {
  it("drops handles that would overlap at this zoom", () => {
    // 100 vertices one image pixel apart, viewed at 1:1: at 9px minimum spacing that is 12 handles, not 100.
    const idx = handleIndices(denseLine(100), 1);
    expect(idx.length).toBeLessThan(20);
  });

  it("brings them back as the reviewer zooms in", () => {
    const dense = denseLine(100);
    expect(handleIndices(dense, 10).length).toBeGreaterThan(handleIndices(dense, 1).length);
  });

  it("returns real indices into the flattened polygon, since a handle drags its own vertex", () => {
    // Renumbering would move the wrong vertex, which corrupts somebody's mask with no error.
    const idx = handleIndices([0, 0, 100, 0, 200, 0], 1);
    expect(idx).toEqual([0, 2, 4]);
    expect(idx.every((i) => i % 2 === 0)).toBe(true);
  });

  it("always leaves something to grab", () => {
    expect(handleIndices([5, 5, 5, 5, 5, 5], 0.01)).toEqual([0]);
    expect(handleIndices([], 1)).toEqual([]);
  });

  it("survives a scale that is zero or not a number rather than hiding every handle", () => {
    expect(handleIndices([0, 0, 100, 0], 0).length).toBe(2);
    expect(handleIndices([0, 0, 100, 0], Number.NaN).length).toBe(2);
  });

  it("drops the redundant vertex RDP leaves at the ring seam", () => {
    // RDP always keeps its two endpoints, and on a closed ring both are vertex 0, so the vertex just
    // before the seam survives whether or not it says anything. On a square outlined with points along
    // each edge that leaves a fifth vertex sitting in the middle of one.
    //
    // It matters because core/polygons.py simplifies again when the mask is written. A client stopping
    // one vertex short would draw an outline that differs from the one being stored.
    const square: number[] = [];
    const n = 20;
    for (let i = 0; i < n; i++) square.push((200 * i) / n, 0);
    for (let i = 0; i < n; i++) square.push(200, (200 * i) / n);
    for (let i = 0; i < n; i++) square.push(200 - (200 * i) / n, 200);
    for (let i = 0; i < n; i++) square.push(0, 200 - (200 * i) / n);

    const out = simplifyPolygon(square, 1);
    expect(out.length / 2).toBe(4);
  });
});
