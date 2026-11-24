import { describe, expect, it } from "vitest";

import type { EdObject } from "@/components/editor/useEditor";

import { canNudge, translateObject } from "./geometry";

// The old paste was the only code that moved an object's geometry as a whole, and it forgot three of the
// kinds the editor stores. Each kind gets its own sentence here so the next one added cannot be forgotten
// quietly: a field that is a position and does not move is an object that comes apart.

function obj(over: Partial<EdObject> = {}): EdObject {
  return {
    id: "a", class_id: 1, class_name: "sedan", bbox: [10, 20, 30, 40], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, ...over,
  };
}

describe("translateObject", () => {
  it("moves the box", () => {
    expect(translateObject(obj(), 1, -2).bbox).toEqual([11, 18, 31, 38]);
  });

  it("leaves rotation alone: it is about the centre, and the centre moved with the box", () => {
    expect(translateObject(obj({ rot: 30 }), 5, 5).rot).toBe(30);
  });

  it("moves every ring of a mask, x by dx and y by dy", () => {
    const o = obj({ mask: [[0, 0, 4, 0, 4, 4], [1, 1, 2, 1, 2, 2]] });
    expect(translateObject(o, 10, 100).mask).toEqual([[10, 100, 14, 100, 14, 104], [11, 101, 12, 101, 12, 102]]);
  });

  it("moves a polyline", () => {
    expect(translateObject(obj({ polyline: [[0, 0], [5, 5]] }), 1, 2).polyline).toEqual([[1, 2], [6, 7]]);
  });

  it("moves the whole-extent box with the visible one", () => {
    expect(translateObject(obj({ bbox_amodal: [0, 0, 50, 50] }), 3, 4).bbox_amodal).toEqual([3, 4, 53, 54]);
  });

  it("does not invent a whole-extent box nobody drew", () => {
    // Undefined and null both mean nobody has judged the hidden part. A translation must not turn that
    // into a claim.
    expect(translateObject(obj(), 3, 4).bbox_amodal).toBeUndefined();
    expect(translateObject(obj({ bbox_amodal: null }), 3, 4).bbox_amodal).toBeNull();
  });

  it("moves placed keypoints and keeps their visibility", () => {
    const o = obj({ keypoints: { skeleton: "person_17", points: [[1, 1, 2], [5, 5, 1]] } });
    expect(translateObject(o, 10, 20).keypoints).toEqual({ skeleton: "person_17", points: [[11, 21, 2], [15, 25, 1]] });
  });

  it("leaves an unplaced keypoint at the origin", () => {
    // v = 0 is "not labelled" and sits at 0,0 by convention. Moving it would make it a point.
    const o = obj({ keypoints: { skeleton: "person_17", points: [[0, 0, 0], [5, 5, 2]] } });
    expect(translateObject(o, 10, 20).keypoints?.points[0]).toEqual([0, 0, 0]);
  });

  it("carries a cuboid unchanged: it is in metres in the ego frame, not in pixels", () => {
    const cuboid = { center: [12, -1.5, 0.8], size: [1.8, 4.2, 1.5], yaw: 0.3, yaw_source: "lane" };
    expect(translateObject(obj({ cuboid_3d: cuboid }), 14, 14).cuboid_3d).toEqual(cuboid);
  });

  it("shares no array with the source, so editing the copy cannot reach the original", () => {
    const src = obj({
      mask: [[0, 0, 4, 0, 4, 4]], polyline: [[0, 0], [1, 1]], bbox_amodal: [0, 0, 9, 9], attrs: { helmet: [true] },
      keypoints: { skeleton: "person_17", points: [[0, 0, 0], [1, 1, 2]] },
      cuboid_3d: { center: [1, 2, 3], size: [1, 1, 1], yaw: 0 },
    });
    const out = translateObject(src, 0, 0);
    expect(out).toEqual(src);
    expect(out.bbox).not.toBe(src.bbox);
    expect(out.mask[0]).not.toBe(src.mask[0]);
    expect(out.polyline?.[0]).not.toBe(src.polyline?.[0]);
    expect(out.bbox_amodal).not.toBe(src.bbox_amodal);
    expect(out.attrs).not.toBe(src.attrs);
    expect(out.keypoints?.points[0]).not.toBe(src.keypoints?.points[0]);
    expect(out.cuboid_3d?.center).not.toBe(src.cuboid_3d?.center);
    expect(out.cuboid_3d?.size).not.toBe(src.cuboid_3d?.size);
  });
});

describe("canNudge", () => {
  it("is true for an ordinary object", () => {
    expect(canNudge(obj())).toBe(true);
  });

  it("is false for a locked one", () => {
    expect(canNudge(obj({ locked: true }))).toBe(false);
  });

  it("is false for a cuboid, which a pixel offset cannot place", () => {
    expect(canNudge(obj({ cuboid_3d: { center: [1, 2, 3], size: [1, 1, 1], yaw: 0 } }))).toBe(false);
  });

  it("is true where the cuboid field is present and empty", () => {
    expect(canNudge(obj({ cuboid_3d: null }))).toBe(true);
  });
});
