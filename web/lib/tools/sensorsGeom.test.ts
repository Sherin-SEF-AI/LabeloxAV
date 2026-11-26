import { describe, expect, it } from "vitest";

import {
  CUBOID_EDGES, bevPixelToWorld, bevUpYaw, cuboidCorners, dopplerColour, invertRigid, nearestByTs, projectToView,
  type ViewCamera,
} from "./sensorsGeom";

// A camera at (0, 0, 1.5) looking along world +x: optical z = world x, optical x = world -y, optical y = -z.
const front: ViewCamera = {
  K: [[200, 0, 160], [0, 200, 120], [0, 0, 1]],
  T_world_cam: [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]],
  width: 320, height: 240, ortho: false,
};

// A top-down view with the drive going +y up the screen: the server's bev_view with heading (0, 1).
// right = (1, 0, 0), image down = -heading = (0, -1, 0), looking down -z; origin at (-5, 45, 30), 10 px/m.
const bev: ViewCamera = {
  K: [[10, 0, 0], [0, 10, 0], [0, 0, 1]],
  T_world_cam: [[1, 0, 0, -5], [0, -1, 0, 45], [0, 0, -1, 30], [0, 0, 0, 1]],
  width: 200, height: 500, ortho: true,
};

describe("projectToView", () => {
  it("puts a point on the optical axis at the principal point", () => {
    const p = projectToView(front, [10, 0, 1.5])!;
    expect(p.u).toBeCloseTo(160);
    expect(p.v).toBeCloseTo(120);
    expect(p.depth).toBeCloseTo(10);
  });
  it("puts world +y (left) on the image left and refuses points behind", () => {
    expect(projectToView(front, [10, 2, 1.5])!.u).toBeCloseTo(160 - 40);
    expect(projectToView(front, [-3, 0, 1.5])).toBeNull();
  });
  it("is orthographic for a top-down view and inverts back from a click", () => {
    const p = projectToView(bev, [0, 0, 0])!;
    expect(p.u).toBeCloseTo(50);
    expect(p.v).toBeCloseTo(450);
    const w = bevPixelToWorld(bev, p.u, p.v);
    expect(w.x).toBeCloseTo(0);
    expect(w.y).toBeCloseTo(0);
    expect(projectToView(bev, [0, 40, 0])!.v).toBeLessThan(p.v); // further along the drive is higher
  });
  it("finds the road's heading as the view's up", () => {
    expect(bevUpYaw(bev)).toBeCloseTo(Math.PI / 2);
  });
});

describe("cuboids", () => {
  it("has 8 corners and 12 edges, rotated by yaw", () => {
    const c = cuboidCorners([0, 0, 1], [4, 2, 2], Math.PI / 2);
    expect(c).toHaveLength(8);
    expect(CUBOID_EDGES).toHaveLength(12);
    // length now lies along y
    expect(c[0][0]).toBeCloseTo(-1);
    expect(c[0][1]).toBeCloseTo(2);
    expect(c[0][2]).toBeCloseTo(0);
    expect(c[4][2]).toBeCloseTo(2);
  });
  it("invertRigid undoes the pose", () => {
    const inv = invertRigid(front.T_world_cam);
    const p = [3, -2, 5];
    const cam = [0, 1, 2].map((i) => inv[i][0] * p[0] + inv[i][1] * p[1] + inv[i][2] * p[2] + inv[i][3]);
    const back = [0, 1, 2].map((i) => front.T_world_cam[i][0] * cam[0] + front.T_world_cam[i][1] * cam[1]
      + front.T_world_cam[i][2] * cam[2] + front.T_world_cam[i][3]);
    back.forEach((v, i) => expect(v).toBeCloseTo(p[i]));
  });
});

describe("radar helpers", () => {
  it("colours approaching blue, receding red, still grey", () => {
    expect(dopplerColour(0)).toBe("rgb(150, 156, 166)");
    expect(dopplerColour(-10)).toBe("rgb(70, 150, 255)");
    expect(dopplerColour(10)).toBe("rgb(255, 96, 70)");
  });
  it("finds the nearest timestamp", () => {
    const rows = [{ ts_ns: 0 }, { ts_ns: 100 }, { ts_ns: 200 }];
    expect(nearestByTs(rows, 140)).toBe(1);
    expect(nearestByTs(rows, 160)).toBe(2);
    expect(nearestByTs(rows, -5)).toBe(0);
    expect(nearestByTs([], 5)).toBe(-1);
  });
});

describe("groundAt", () => {
  it("takes a low percentile of the heights near the click, in world coordinates", async () => {
    const { groundAt } = await import("./sensorsGeom");
    const pts: number[] = [];
    for (let i = 0; i < 40; i++) pts.push((i % 7) * 0.1, (i % 5) * 0.1, i < 30 ? 0 : 1.5, 0, 0, 0);  // road + a car
    pts.push(10, 10, 9, 0, 0, 0);                                                                // far away
    const arr = new Float32Array(pts);
    expect(groundAt(arr, arr.length / 6, [100, 200, -1.7], 100.2, 200.2)).toBeCloseTo(-1.7);
    expect(groundAt(arr, arr.length / 6, [100, 200, -1.7], 150, 250)).toBeNull();
  });
});
