import { describe, expect, it } from "vitest";

import {
  bevToMetric, catmullRom, edgeScore, metricToBev, mutualInformation, nearestSegment, pairsError, pathBounds,
  plotTransform, project, rotation,
} from "./drivingGeom";

// Reference values computed with the Python the server runs (services/hdmap/bev_lanes.py, services/lidar/calib/
// align.py, services/calibration/resolve.py) on the same inputs.

describe("lane splines", () => {
  it("matches the server's Catmull-Rom", () => {
    const out = catmullRom([[0, 0], [10, 5], [20, 0], [30, 8]], 4);
    const ref = [[0, 0], [2.5, 1.4844], [5, 3.125], [7.5, 4.4531], [10, 5], [12.5, 4.1484], [15, 2.3125],
      [17.5, 0.5703], [20, 0], [22.5, 1.0859], [25, 3.1875], [27.5, 5.6953], [30, 8]];
    expect(out).toHaveLength(ref.length);
    out.forEach((p, i) => { expect(p[0]).toBeCloseTo(ref[i][0], 3); expect(p[1]).toBeCloseTo(ref[i][1], 3); });
    expect(catmullRom([[0, 0], [5, 5]])).toEqual([[0, 0], [5, 5]]);
  });
  it("round-trips BEV pixels and metres", () => {
    const v = { near_m: 2, far_m: 60, half_width_m: 12, px_per_m: 20, width: 480, height: 1160 };
    const [f, l] = bevToMetric(v, 275, 700);
    expect(f).toBeCloseTo(25);
    expect(l).toBeCloseTo(1.75);
    expect(metricToBev(v, f, l)).toEqual([275, 700]);
  });
  it("finds the segment to insert a point into", () => {
    expect(nearestSegment([[0, 0], [10, 0], [10, 10]], 9, 6)).toEqual({ index: 1, dist: 1 });
  });
});

describe("camera-LiDAR projection", () => {
  const k = { fx: 721.5, fy: 721.5, cx: 609.6, cy: 172.9 };
  const ex = { rpy_deg: [0.3, 0.6, -0.4], xyz_m: [0.27, -0.06, -0.08] };
  it("builds the resolver's rotation", () => {
    const r = rotation(ex.rpy_deg);
    const ref = [[0.00692634, 0.01050794, 0.99992079], [-0.99996227, -0.00516273, 0.00698088],
      [0.00523568, -0.99993145, 0.01047178]];
    r.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(ref[i][j], 6)));
  });
  it("projects like the server", () => {
    const pts = new Float32Array([10, 2, 0.5, 9, 9, 25, -3, -1.2, 0, 0, -5, 0, 0, 0, 0]);
    const { uv, inside, depth } = project(pts, 5, k, ex, 1242, 375);
    expect(uv[0]).toBeCloseTo(462.3728, 1);
    expect(uv[1]).toBeCloseTo(136.7607, 1);
    expect(uv[2]).toBeCloseTo(700.3233, 1);
    expect(uv[3]).toBeCloseTo(213.6546, 1);
    expect(depth[1]).toBeCloseTo(24.69579, 3);
    expect(Array.from(inside)).toEqual([1, 1, 0]);
    expect(Number.isNaN(uv[4])).toBe(true);
  });
  it("scores edges and pairs", () => {
    const field = new Float32Array(4 * 3).fill(0.25);
    field[1 * 4 + 2] = 1;
    const uv = new Float32Array([2.2, 1.7, 0.5, 0.5, 3.1, 2.1]);
    expect(edgeScore(uv, new Uint8Array([1, 1, 0]), [2, 2, 5], field, 4, 3)).toBeCloseTo((2 * 1 + 2 * 0.25) / 4);
    expect(edgeScore(uv, new Uint8Array([0, 0, 0]), [1, 1, 1], field, 4, 3)).toBeNull();
    const p3 = [[10, 2, 0.5]];
    expect(pairsError(p3, [[462.3728, 139.7607]], k, ex, 1242, 375).rms).toBeCloseTo(3, 1);
  });
  it("measures mutual information like the server", () => {
    const a: number[] = [];
    const b: number[] = [];
    const c: number[] = [];
    for (let i = 0; i < 300; i++) { a.push((i % 7) / 7); b.push(((i % 7) * 3 + (i % 3)) / 30); c.push((i % 11) / 11); }
    expect(mutualInformation(a, b)).toBeCloseTo(1.0, 6);
    expect(mutualInformation(a, c)).toBeCloseTo(0.0016119303, 6);
    expect(mutualInformation([1, 2], [3, 4])).toBeNull();
  });
});

describe("path plot", () => {
  it("keeps metres square and north up", () => {
    const b = pathBounds([[[0, 0, 0], [1, 10, 0]], [[0, 0, 20]]], 0);
    expect(b).toEqual({ minX: 0, maxX: 10, minY: 0, maxY: 20 });
    const t = plotTransform(b, 200, 200);
    const [x0, y0] = t.toPx(0, 0);
    const [x1, y1] = t.toPx(10, 20);
    expect(x1 - x0).toBeCloseTo((y0 - y1) / 2);
    expect(t.toM(...t.toPx(3, 7))[1]).toBeCloseTo(7);
  });
});
