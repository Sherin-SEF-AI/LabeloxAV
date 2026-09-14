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
  // RDP always keeps its two endpoints, and on a closed ring both of those are vertex 0, so the vertex
  // just before the seam survives whether or not it says anything. On a square whose corner sits at
  // vertex 0 that leaves a fifth vertex in the middle of an edge. `core/polygons.py` does the same pass
  // for the same reason: the server simplifies again at write time, and a client that stopped one vertex
  // short would draw a different outline from the one being stored.
  out = dropCollinear(out, tolerance);
  if (out.length < MIN_POLYGON_POINTS) return [...flat];
  return out.flat();
}

/**
 * Drop vertices lying on the line through their two neighbours, treating the array as a closed ring.
 *
 * Only the seam vertices can survive RDP this way, so at most a couple are ever removed. The pass covers
 * the whole ring because that is simpler than reasoning about which two they were.
 */
function dropCollinear(points: Pt[], tolerance: number): Pt[] {
  const n = points.length;
  if (n <= MIN_POLYGON_POINTS) return points;
  const keep = new Array<boolean>(n).fill(true);
  let kept = n;
  for (let i = 0; i < n; i++) {
    if (kept <= MIN_POLYGON_POINTS) break;
    // Neighbours among the vertices still kept, so removing one does not make the next look redundant
    // against a vertex that has itself already gone.
    let prev = (i - 1 + n) % n;
    while (!keep[prev] && prev !== i) prev = (prev - 1 + n) % n;
    let next = (i + 1) % n;
    while (!keep[next] && next !== i) next = (next + 1) % n;
    if (prev === i || next === i) break;
    if (segmentDistance(points[i], points[prev], points[next]) <= tolerance) {
      keep[i] = false;
      kept--;
    }
  }
  return points.filter((_, i) => keep[i]);
}

/** Simplify every ring of a mask. */
export function simplifyMask(polys: readonly (readonly number[])[], tolerance = 1): number[][] {
  return polys.map((poly) => simplifyPolygon(poly, tolerance));
}

// Bounds and fraction mirrored from core/polygons.py. The server simplifies again when the mask is
// written, so a client using a different tolerance would show an outline that differs from the stored
// one until the next reload.
export const TOLERANCE_FRAC = 0.005;
export const MIN_TOLERANCE_PX = 0.35;
export const MAX_TOLERANCE_PX = 4.0;
/** Rings below this area are left exactly as drawn; see the measurement in core/polygons.py. */
export const MIN_SIMPLIFY_AREA_PX = 64;

/**
 * The tolerance for one ring, from its own perimeter.
 *
 * A fixed pixel tolerance cannot be right for both a 20px sign and a 900px bus: on the first it removes
 * real shape, on the second it leaves hundreds of vertices tracing a straight edge.
 */
export function sizeTolerance(flat: readonly number[]): number {
  const pts = toPoints(flat);
  if (pts.length < 2) return MIN_TOLERANCE_PX;
  let per = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    per += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return Math.min(MAX_TOLERANCE_PX, Math.max(MIN_TOLERANCE_PX, per * TOLERANCE_FRAC));
}

function ringArea(flat: readonly number[]): number {
  const pts = toPoints(flat);
  if (pts.length < 3) return 0;
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return Math.abs(a) / 2;
}

/**
 * Simplify a mask exactly as the server will when it stores it.
 *
 * Use this rather than `simplifyMask` with a fixed tolerance anywhere the result is about to be saved:
 * what the annotator approves and what lands on disk should be the same outline.
 */
export function simplifyMaskLikeServer(polys: readonly (readonly number[])[]): number[][] {
  return polys.map((poly) =>
    ringArea(poly) < MIN_SIMPLIFY_AREA_PX ? [...poly] : simplifyPolygon(poly, sizeTolerance(poly)),
  );
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
