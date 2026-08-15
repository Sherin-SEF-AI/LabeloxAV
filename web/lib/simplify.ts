// Polygon simplification, and which vertex handles are worth drawing.
//
// SAM returns a mask traced at pixel resolution: a segmented car arrives as several hundred vertices, most
// of them a fraction of a pixel apart. Two costs follow. The stored polygon is an order of magnitude larger
// than the shape it describes, in every export and every payload. And the editor draws one draggable handle
// per vertex, so selecting that car covers it in a solid band of circles: the object disappears under its
// own controls, and grabbing the one vertex that is wrong means hitting a target overlapped by twenty
// others.
//
// Douglas-Peucker is the right shape of answer because its tolerance is in the units of the thing being
// simplified. A tolerance of one image pixel removes vertices that cannot be seen at 100% zoom and keeps
// every corner that can.

type Pt = [number, number];

function toPoints(flat: readonly number[]): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i + 1 < flat.length; i += 2) pts.push([flat[i], flat[i + 1]]);
  return pts;
}

/** Perpendicular distance from p to the segment ab, in the same units as the coordinates. */
function segmentDistance(p: Pt, a: Pt, b: Pt): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  if (dx === 0 && dy === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  // Clamped projection, so a point beyond either end measures to the end rather than to the infinite line.
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

function douglasPeucker(pts: Pt[], tolerance: number): Pt[] {
  if (pts.length <= 2) return pts;
  let worst = 0;
  let index = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = segmentDistance(pts[i], pts[0], pts[pts.length - 1]);
    if (d > worst) { worst = d; index = i; }
  }
  if (worst <= tolerance) return [pts[0], pts[pts.length - 1]];
  const left = douglasPeucker(pts.slice(0, index + 1), tolerance);
  const right = douglasPeucker(pts.slice(index), tolerance);
  return [...left.slice(0, -1), ...right];
}

/** Below this many vertices there is nothing to gain and a small polygon can only be damaged. */
export const MIN_POLYGON_POINTS = 3;

/**
 * Simplify a flattened [x,y,x,y,...] polygon, keeping at least a triangle.
 *
 * The ring is opened at its first vertex and closed again afterwards, so the seam is treated as a real
 * corner rather than as two unrelated endpoints, which is what would flatten the shape there.
 */
export function simplifyPolygon(flat: readonly number[], tolerance = 1): number[] {
  const pts = toPoints(flat);
  if (pts.length <= MIN_POLYGON_POINTS || tolerance <= 0) return [...flat];
  const closed = pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1];
  const ring = closed ? pts : [...pts, pts[0]];
  let out = douglasPeucker(ring, tolerance);
  if (!closed) out = out.slice(0, -1);
  if (out.length < MIN_POLYGON_POINTS) return [...flat];
  return out.flat();
}

/** Simplify every ring of a mask. */
export function simplifyMask(polys: readonly (readonly number[])[], tolerance = 1): number[][] {
  return polys.map((poly) => simplifyPolygon(poly, tolerance));
}

/** Handles closer together than this on screen overlap each other and cannot be aimed at separately. */
export const MIN_HANDLE_SPACING_PX = 9;

/**
 * The indices (into the flattened polygon) of the vertices worth drawing a handle for at this zoom.
 *
 * Indices rather than points, because a handle drags the vertex it belongs to: a decimated list that
 * renumbered them would move the wrong vertex, which is a silent corruption of somebody's mask. Zooming in
 * brings the skipped vertices back, so nothing is unreachable, it is only undrawn where it would be
 * unhittable anyway.
 */
export function handleIndices(flat: readonly number[], scale: number,
                              minSpacingPx = MIN_HANDLE_SPACING_PX): number[] {
  const n = Math.floor(flat.length / 2);
  const out: number[] = [];
  if (n === 0) return out;
  const s = Number.isFinite(scale) && scale > 0 ? scale : 1;
  let lastX = Number.NaN;
  let lastY = Number.NaN;
  for (let i = 0; i < n; i++) {
    const x = flat[i * 2];
    const y = flat[i * 2 + 1];
    if (out.length === 0 || Math.hypot(x - lastX, y - lastY) * s >= minSpacingPx) {
      out.push(i * 2);
      lastX = x;
      lastY = y;
    }
  }
  // A polygon whose every vertex is inside one handle's width still needs something to grab.
  if (out.length === 0) out.push(0);
  return out;
}
