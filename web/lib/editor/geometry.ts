// Moving an object in image space, for every geometry kind the editor stores.
//
// The editor had exactly one way to move anything: drag the box. That moved `bbox` and nothing else, which
// was already wrong for an object carrying a mask, and there was no keyboard path at all. An arrow-key
// nudge and a paste both need the same thing, a translation that carries every field that is a position,
// so it lives here once rather than as two partial copies that each forget a different field. The old
// paste forgot three: polyline, cuboid_3d and bbox_amodal were dropped on the floor.

import type { EdObject } from "@/components/editor/useEditor";

/**
 * Whether an object's position can be changed by a 2D offset without making it lie.
 *
 * A cuboid lives in the ego frame, in metres. Sliding its 2D box by a pixel says nothing about where the
 * 3D box went, and leaving the 3D box where it was while the 2D one moves produces an object whose two
 * geometries disagree. So a cuboid object is skipped, explicitly, rather than half moved.
 */
export function canNudge(o: EdObject): boolean {
  return !o.locked && !o.cuboid_3d;
}

/**
 * The same object, `dx, dy` image pixels away. Every array in the result is a fresh copy.
 *
 * Rotation is about the box centre, so it travels with the box and needs no change. Keypoints that were
 * never placed (v = 0) sit at the origin by convention and stay there: moving them would turn "not
 * labelled" into a point at (dx, dy). The cuboid is copied and not moved, for the reason above.
 */
export function translateObject(o: EdObject, dx: number, dy: number): EdObject {
  const out: EdObject = {
    ...o,
    bbox: [o.bbox[0] + dx, o.bbox[1] + dy, o.bbox[2] + dx, o.bbox[3] + dy],
    mask: o.mask.map((poly) => poly.map((v, i) => v + (i % 2 === 0 ? dx : dy))),
    attrs: { ...o.attrs },
  };
  if (o.bbox_amodal?.length === 4) {
    const a = o.bbox_amodal;
    out.bbox_amodal = [a[0] + dx, a[1] + dy, a[2] + dx, a[3] + dy];
  }
  if (o.polyline) out.polyline = o.polyline.map((pt) => [pt[0] + dx, pt[1] + dy]);
  if (o.keypoints) {
    out.keypoints = {
      skeleton: o.keypoints.skeleton,
      points: o.keypoints.points.map((pt) => (pt[2] > 0 ? [pt[0] + dx, pt[1] + dy, pt[2]] : pt.slice())),
    };
  }
  if (o.cuboid_3d) {
    out.cuboid_3d = { ...o.cuboid_3d, center: o.cuboid_3d.center.slice(), size: o.cuboid_3d.size.slice() };
  }
  return out;
}
