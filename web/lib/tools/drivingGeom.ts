// Geometry for the driving tools, mirrored from the Python so the browser shows exactly what the server stores:
//  - catmullRom: services/hdmap/bev_lanes.py catmull_rom (the lane spline drawn in the BEV editor and exported)
//  - bevToMetric / metricToBev: services/hdmap/bev_view.py BevView (row 0 is the far edge; x grows to the right)
//  - rotation / project: services/calibration/resolve.py Calibration.R and services/lidar/calib/align.py project
//  - edgeScore / mutualInformation: services/lidar/calib/align.py, so the live score on /calibration/align is the
//    number the auto-alignment optimises and the save records.

export type Pt = [number, number];

/** Uniform Catmull-Rom through the points, `samples` per segment, passing through every point. */
export function catmullRom(points: number[][], samples = 8): Pt[] {
  const p = points.map((q) => [q[0], q[1]] as Pt);
  if (p.length < 3 || samples < 2) return p;
  const ext: Pt[] = [
    [2 * p[0][0] - p[1][0], 2 * p[0][1] - p[1][1]],
    ...p,
    [2 * p[p.length - 1][0] - p[p.length - 2][0], 2 * p[p.length - 1][1] - p[p.length - 2][1]],
  ];
  const out: Pt[] = [];
  for (let i = 1; i < ext.length - 2; i++) {
    const [p0, p1, p2, p3] = [ext[i - 1], ext[i], ext[i + 1], ext[i + 2]];
    for (let s = 0; s < samples; s++) {
      const t = s / samples;
      const t2 = t * t;
      const t3 = t2 * t;
      const f = (k: 0 | 1) => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
        + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3);
      out.push([f(0), f(1)]);
    }
  }
  out.push(p[p.length - 1]);
  return out;
}

export type BevView = { near_m: number; far_m: number; half_width_m: number; px_per_m: number; width: number; height: number };

/** A BEV pixel to (forward, lateral-right) metres from the camera's ground point. */
export function bevToMetric(v: BevView, bx: number, by: number): Pt {
  return [v.far_m - by / v.px_per_m, bx / v.px_per_m - v.half_width_m];
}

/** (forward, lateral-right) metres to a BEV pixel; the exact inverse of bevToMetric. */
export function metricToBev(v: BevView, forward: number, lateral: number): Pt {
  return [(lateral + v.half_width_m) * v.px_per_m, (v.far_m - forward) * v.px_per_m];
}

/** Index of the polyline segment nearest to (x, y) and the distance to it, for inserting a control point. */
export function nearestSegment(points: number[][], x: number, y: number): { index: number; dist: number } {
  let best = { index: -1, dist: Infinity };
  for (let i = 0; i + 1 < points.length; i++) {
    const [ax, ay] = points[i];
    const [bx, by] = points[i + 1];
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy || 1e-9;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2));
    const d = Math.hypot(ax + t * dx - x, ay + t * dy - y);
    if (d < best.dist) best = { index: i, dist: d };
  }
  return best;
}

// ---- camera-LiDAR alignment ------------------------------------------------------------------------------

export type Intrinsics = { fx: number; fy: number; cx: number; cy: number };
export type Extrinsic = { rpy_deg: number[]; xyz_m: number[] };

const R_OPT2EGO = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]];

function matmul(a: number[][], b: number[][]): number[][] {
  return a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
}

/** The 3x3 that maps an ego-shifted row vector to the camera optical frame: cam = (ego - xyz) @ R. */
export function rotation(rpyDeg: number[]): number[][] {
  const [roll, pitch, yaw] = rpyDeg.map((d) => (d * Math.PI) / 180);
  const cz = Math.cos(-yaw);
  const sz = Math.sin(-yaw);
  const cp = Math.cos(-pitch);
  const sp = Math.sin(-pitch);
  const cr = Math.cos(-roll);
  const sr = Math.sin(-roll);
  const rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
  const ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]];
  const rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]];
  return matmul(matmul(matmul(rz, ry), rx), R_OPT2EGO);
}

/**
 * Project interleaved points (stride floats per point, x y z first) into the image. Returns uv (NaN when behind
 * the camera) and the in-image mask, the same as align.project.
 */
export function project(points: Float32Array, stride: number, k: Intrinsics, ex: Extrinsic, width: number,
  height: number): { uv: Float32Array; inside: Uint8Array; depth: Float32Array } {
  const n = Math.floor(points.length / stride);
  const r = rotation(ex.rpy_deg);
  const [tx, ty, tz] = ex.xyz_m;
  const uv = new Float32Array(n * 2);
  const inside = new Uint8Array(n);
  const depth = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = points[i * stride] - tx;
    const y = points[i * stride + 1] - ty;
    const z = points[i * stride + 2] - tz;
    const cx = x * r[0][0] + y * r[1][0] + z * r[2][0];
    const cy = x * r[0][1] + y * r[1][1] + z * r[2][1];
    const cz = x * r[0][2] + y * r[1][2] + z * r[2][2];
    depth[i] = cz;
    if (cz <= 1e-3) {
      uv[i * 2] = NaN;
      uv[i * 2 + 1] = NaN;
      continue;
    }
    const u = (cx / cz) * k.fx + k.cx;
    const v = (cy / cz) * k.fy + k.cy;
    uv[i * 2] = u;
    uv[i * 2 + 1] = v;
    inside[i] = u >= 0 && u < width && v >= 0 && v < height ? 1 : 0;
  }
  return { uv, inside, depth };
}

/** A single-channel field (row-major, width x height) looked up at the nearest pixel, as align.sample. */
export function sampleField(field: ArrayLike<number>, width: number, height: number, u: number, v: number): number {
  const x = Math.min(width - 1, Math.max(0, Math.floor(u)));
  const y = Math.min(height - 1, Math.max(0, Math.floor(v)));
  return field[y * width + x];
}

/** Weighted mean of the edge field (0..1) under the projected boundary points; null when none are in view. */
export function edgeScore(uv: Float32Array, inside: Uint8Array, weights: ArrayLike<number>, field: ArrayLike<number>,
  width: number, height: number): number | null {
  let num = 0;
  let den = 0;
  for (let i = 0; i < inside.length; i++) {
    const w = weights[i];
    if (!inside[i] || w <= 0) continue;
    num += w * sampleField(field, width, height, uv[i * 2], uv[i * 2 + 1]);
    den += w;
  }
  return den > 0 ? num / den : null;
}

/** Normalised mutual information I(a;b) / min(H(a), H(b)) of two samples, histogrammed into `bins` each over the
 * samples' own range (numpy.histogram2d's default). */
export function mutualInformation(a: ArrayLike<number>, b: ArrayLike<number>, bins = 32): number | null {
  const n = a.length;
  if (n < 50) return null;
  const range = (x: ArrayLike<number>) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < x.length; i++) { lo = Math.min(lo, x[i]); hi = Math.max(hi, x[i]); }
    return [lo, hi === lo ? lo + 1 : hi];
  };
  const [alo, ahi] = range(a);
  const [blo, bhi] = range(b);
  const joint = new Float64Array(bins * bins);
  const bin = (v: number, lo: number, hi: number) => Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins));
  for (let i = 0; i < n; i++) joint[bin(a[i], alo, ahi) * bins + bin(b[i], blo, bhi)] += 1;
  const px = new Float64Array(bins);
  const py = new Float64Array(bins);
  for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) { px[i] += joint[i * bins + j] / n; py[j] += joint[i * bins + j] / n; }
  let mi = 0;
  for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) {
    const p = joint[i * bins + j] / n;
    if (p > 0) mi += p * Math.log(p / (px[i] * py[j]));
  }
  const h = (q: Float64Array) => q.reduce((s, v) => (v > 0 ? s - v * Math.log(v) : s), 0);
  const d = Math.min(h(px), h(py));
  return d > 1e-9 ? mi / d : null;
}

/** RMS and max pixel error of clicked pairs (3D point, observed pixel) under an extrinsic. */
export function pairsError(points: number[][], uvObs: number[][], k: Intrinsics, ex: Extrinsic, width: number,
  height: number): { rms: number | null; max: number | null; n: number } {
  if (!points.length) return { rms: null, max: null, n: 0 };
  const flat = new Float32Array(points.flatMap((p) => [p[0], p[1], p[2]]));
  const { uv, depth } = project(flat, 3, k, ex, width, height);
  let s = 0;
  let mx = 0;
  let n = 0;
  for (let i = 0; i < points.length; i++) {
    if (depth[i] <= 1e-3) continue;
    const e = Math.hypot(uv[i * 2] - uvObs[i][0], uv[i * 2 + 1] - uvObs[i][1]);
    s += e * e;
    mx = Math.max(mx, e);
    n++;
  }
  return n ? { rms: Math.sqrt(s / n), max: mx, n } : { rms: null, max: null, n: 0 };
}

/** A depth to a colour ramp (near warm, far cool), for drawing projected points. */
export function depthColor(d: number, near = 3, far = 50): string {
  const t = Math.max(0, Math.min(1, (d - near) / (far - near)));
  const hue = Math.round(10 + t * 210);
  return `hsl(${hue} 95% 55%)`;
}

// ---- paths ------------------------------------------------------------------------------------------------

/** The bounds of a set of [t, x, y] (or [x, y]) paths, padded, for fitting a top-down plot. */
export function pathBounds(paths: number[][][], pad = 5): { minX: number; maxX: number; minY: number; maxY: number } {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const path of paths) for (const p of path) {
    const x = p.length >= 3 ? p[1] : p[0];
    const y = p.length >= 3 ? p[2] : p[1];
    minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  if (!Number.isFinite(minX)) return { minX: -pad, maxX: pad, minY: -pad, maxY: pad };
  return { minX: minX - pad, maxX: maxX + pad, minY: minY - pad, maxY: maxY + pad };
}

/** World/ego metres to plot pixels and back, keeping metres square and +y (north / left) pointing up. */
export function plotTransform(b: { minX: number; maxX: number; minY: number; maxY: number }, w: number, h: number) {
  const s = Math.min(w / (b.maxX - b.minX), h / (b.maxY - b.minY));
  const ox = (w - s * (b.maxX - b.minX)) / 2;
  const oy = (h - s * (b.maxY - b.minY)) / 2;
  return {
    scale: s,
    toPx: (x: number, y: number): Pt => [ox + (x - b.minX) * s, h - oy - (y - b.minY) * s],
    toM: (px: number, py: number): Pt => [b.minX + (px - ox) / s, b.minY + (h - oy - py) / s],
  };
}
