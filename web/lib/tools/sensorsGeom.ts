// Geometry for the sensors pages, pure so it can be tested: projecting a world point through a rendered
// reconstruction view (the same camera the server rendered it with), the eight corners of a cuboid, turning a
// click on the top-down view back into a world point, and colouring radar returns by their Doppler.

export type Mat = number[][];

export type ViewCamera = { K: Mat; T_world_cam: Mat; width: number; height: number; ortho: boolean };

/** Invert a rigid 4x4 (rotation + translation) without a general inverse. */
export function invertRigid(T: Mat): Mat {
  const R = [[T[0][0], T[1][0], T[2][0]], [T[0][1], T[1][1], T[2][1]], [T[0][2], T[1][2], T[2][2]]];
  const t = [T[0][3], T[1][3], T[2][3]];
  const ti = R.map((row) => -(row[0] * t[0] + row[1] * t[1] + row[2] * t[2]));
  return [[...R[0], ti[0]], [...R[1], ti[1]], [...R[2], ti[2]], [0, 0, 0, 1]];
}

/** A world point in a view's pixels, or null when it is behind the camera. Depth is along the optical axis. */
export function projectToView(cam: ViewCamera, p: number[], inv?: Mat): { u: number; v: number; depth: number } | null {
  const M = inv ?? invertRigid(cam.T_world_cam);
  const x = M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3];
  const y = M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3];
  const z = M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3];
  if (cam.ortho) return { u: cam.K[0][0] * x + cam.K[0][2], v: cam.K[1][1] * y + cam.K[1][2], depth: z };
  if (z <= 0.05) return null;
  return { u: (cam.K[0][0] * x) / z + cam.K[0][2], v: (cam.K[1][1] * y) / z + cam.K[1][2], depth: z };
}

/** The 8 corners of a cuboid (dims = length, width, height; yaw about +z), bottom four first. */
export function cuboidCorners(center: number[], dims: number[], yaw: number): number[][] {
  const [l, w, h] = [dims[0] / 2, dims[1] / 2, dims[2] / 2];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const out: number[][] = [];
  for (const dz of [-h, h]) {
    for (const [dx, dy] of [[l, w], [l, -w], [-l, -w], [-l, w]]) {
      out.push([center[0] + c * dx - s * dy, center[1] + s * dx + c * dy, center[2] + dz]);
    }
  }
  return out;
}

/** The 12 edges of `cuboidCorners`, as index pairs; the first edge is the front of the bottom face. */
export const CUBOID_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
];

/**
 * A pixel of an orthographic top-down view back to the world (x, y). The view's camera looks straight down,
 * so a pixel is camera-frame (u / fx, v / fy), and the camera's axes map that into the world.
 */
export function bevPixelToWorld(cam: ViewCamera, u: number, v: number): { x: number; y: number } {
  const xc = (u - cam.K[0][2]) / cam.K[0][0];
  const yc = (v - cam.K[1][2]) / cam.K[1][1];
  const T = cam.T_world_cam;
  return { x: T[0][0] * xc + T[0][1] * yc + T[0][3], y: T[1][0] * xc + T[1][1] * yc + T[1][3] };
}

/** The world yaw that points "up" on a top-down view (its heading), so a new box starts aligned with the road. */
export function bevUpYaw(cam: ViewCamera): number {
  // image up is the camera's -y axis
  return Math.atan2(-cam.T_world_cam[1][1], -cam.T_world_cam[0][1]);
}

/**
 * Doppler colour: approaching (negative radial speed) blue, receding red, stationary grey, saturating at
 * `full` m/s. Returned as an rgb() string for canvas and SVG.
 */
export function dopplerColour(radialMps: number, full = 10): string {
  const t = Math.max(-1, Math.min(1, radialMps / full));
  const a = Math.abs(t);
  const grey = [150, 156, 166];
  const end = t < 0 ? [70, 150, 255] : [255, 96, 70];
  const mix = grey.map((g, i) => Math.round(g + (end[i] - g) * Math.min(1, a * 1.6)));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

/** Index of the item whose ts_ns is nearest `ts` in a list sorted by ts_ns, or -1 for an empty list. */
export function nearestByTs<T extends { ts_ns: number }>(rows: T[], ts: number): number {
  if (!rows.length) return -1;
  let lo = 0, hi = rows.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].ts_ns < ts) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(rows[lo - 1].ts_ns - ts) <= Math.abs(rows[lo].ts_ns - ts)) return lo - 1;
  return lo;
}

/**
 * The ground height near world (x, y) from a preview cloud ([x, y, z, r, g, b] relative to `anchor`): the 15th
 * percentile of the heights within `radius` metres, so a car or a kerb in the disc does not lift it. Null when
 * fewer than 5 points are near.
 */
export function groundAt(points: Float32Array, count: number, anchor: number[], x: number, y: number,
  radius = 1.5): number | null {
  const lx = x - anchor[0], ly = y - anchor[1], r2 = radius * radius;
  const zs: number[] = [];
  for (let i = 0; i < count; i++) {
    const dx = points[i * 6] - lx, dy = points[i * 6 + 1] - ly;
    if (dx * dx + dy * dy <= r2) zs.push(points[i * 6 + 2]);
  }
  if (zs.length < 5) return null;
  zs.sort((a, b) => a - b);
  return zs[Math.floor(zs.length * 0.15)] + anchor[2];
}
