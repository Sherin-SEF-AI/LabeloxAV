// Pure helpers for the 3D labelling pages (/lidar/segment, /lidar/occupancy, label once on /lidar/annotate):
// screen-space selection over projected points, the label buffer with undo, class colours, and voxel maths.
// No three.js here, so every rule is unit-tested (scene3dGeom.test.ts).

export type XY = [number, number];

/** Even-odd rule: is (x, y) inside the polygon? */
export function pointInPolygon(x: number, y: number, poly: XY[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-12) + xi) inside = !inside;
  }
  return inside;
}

/** Points (screen xy interleaved, NaN = off screen / behind camera) inside a polygon, as a 0/1 mask. */
export function selectPolygon(screen: Float32Array, poly: XY[]): Uint8Array {
  const n = screen.length / 2;
  const out = new Uint8Array(n);
  if (poly.length < 3) return out;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of poly) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  for (let i = 0; i < n; i++) {
    const x = screen[i * 2], y = screen[i * 2 + 1];
    if (!(x >= x0 && x <= x1 && y >= y0 && y <= y1)) continue; // also rejects NaN
    if (pointInPolygon(x, y, poly)) out[i] = 1;
  }
  return out;
}

export function selectRect(screen: Float32Array, a: XY, b: XY): Uint8Array {
  const n = screen.length / 2;
  const out = new Uint8Array(n);
  const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
  for (let i = 0; i < n; i++) {
    const x = screen[i * 2], y = screen[i * 2 + 1];
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) out[i] = 1;
  }
  return out;
}

/** Brush: every point within `r` pixels of (cx, cy), OR-ed into `into`. Returns how many it added. */
export function brushInto(into: Uint8Array, screen: Float32Array, cx: number, cy: number, r: number): number {
  const r2 = r * r;
  let added = 0;
  for (let i = 0; i < into.length; i++) {
    if (into[i]) continue;
    const dx = screen[i * 2] - cx, dy = screen[i * 2 + 1] - cy;
    if (dx * dx + dy * dy <= r2) { into[i] = 1; added++; }
  }
  return added;
}

export type Combine = "replace" | "add" | "subtract";

/** Combine a new selection with the current one (shift adds, alt subtracts). */
export function combineMask(cur: Uint8Array | null, next: Uint8Array, how: Combine): Uint8Array {
  if (!cur || how === "replace") return next;
  const out = new Uint8Array(next.length);
  for (let i = 0; i < next.length; i++) out[i] = how === "add" ? (cur[i] | next[i]) : (cur[i] && !next[i] ? 1 : 0);
  return out;
}

export const countMask = (m: Uint8Array | null) => { let c = 0; if (m) for (let i = 0; i < m.length; i++) c += m[i]; return c; };

/**
 * Write `cls` (-1 to erase) onto every selected point. With `onlyUnlabelled`, points that already carry a class
 * keep it, so a broad lasso can fill the gaps around careful work without overwriting it. Returns the count.
 */
export function applyClass(labels: Int16Array, mask: Uint8Array, cls: number, onlyUnlabelled = false): number {
  let n = 0;
  for (let i = 0; i < labels.length; i++) {
    if (!mask[i]) continue;
    if (onlyUnlabelled && labels[i] >= 0) continue;
    if (labels[i] !== cls) { labels[i] = cls; n++; }
  }
  return n;
}

/** Suggested labels (from 2D masks, cuboids or PTv3) merged into the working labels. */
export function mergeSuggestion(labels: Int16Array, suggestion: Int16Array, onlyUnlabelled: boolean): number {
  let n = 0;
  for (let i = 0; i < labels.length && i < suggestion.length; i++) {
    const s = suggestion[i];
    if (s < 0) continue;
    if (onlyUnlabelled && labels[i] >= 0) continue;
    if (labels[i] !== s) { labels[i] = s; n++; }
  }
  return n;
}

export function countByClass(labels: Int16Array): Map<number, number> {
  const m = new Map<number, number>();
  for (let i = 0; i < labels.length; i++) m.set(labels[i], (m.get(labels[i]) ?? 0) + 1);
  return m;
}

/** A bounded undo stack of label snapshots. */
export class LabelHistory {
  private stack: Int16Array[] = [];
  constructor(private readonly max = 30) {}
  push(labels: Int16Array) {
    this.stack.push(labels.slice());
    if (this.stack.length > this.max) this.stack.shift();
  }
  undo(): Int16Array | null { return this.stack.pop() ?? null; }
  get size() { return this.stack.length; }
  clear() { this.stack = []; }
}

/** A stable, well-spread colour per class id (golden-angle hue); -1 is null (caller draws it by height). */
export function classRgb(id: number): [number, number, number] | null {
  if (id < 0) return null;
  const h = ((id * 137.508) % 360) / 360;
  const s = 0.72, l = 0.56;
  const k = (n: number) => (n + h * 12) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)];
}

export const rgbCss = (rgb: [number, number, number] | null) =>
  rgb ? `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})` : "transparent";

// ---- voxels ----

/** The voxel index containing a point, or null outside the grid. */
export function voxelOf(p: number[], origin: number[], voxelM: number, dims: number[]): [number, number, number] | null {
  const v = [0, 1, 2].map((a) => Math.floor((p[a] - origin[a]) / voxelM));
  return v.every((x, a) => x >= 0 && x < dims[a]) ? (v as [number, number, number]) : null;
}

/** A voxel's centre in the grid's frame. */
export const voxelCentre = (v: number[], origin: number[], voxelM: number) => [0, 1, 2].map((a) => origin[a] + (v[a] + 0.5) * voxelM);

/** The inclusive voxel box spanned by two ground points and a z-index range, clipped to the grid. */
export function voxelBox(a: number[], b: number[], zLo: number, zHi: number, origin: number[], voxelM: number,
  dims: number[]): { lo: number[]; hi: number[] } | null {
  const ix = (p: number, ax: number) => Math.floor((p - origin[ax]) / voxelM);
  const lo = [Math.max(0, Math.min(ix(a[0], 0), ix(b[0], 0))), Math.max(0, Math.min(ix(a[1], 1), ix(b[1], 1))), Math.max(0, Math.min(zLo, zHi))];
  const hi = [Math.min(dims[0] - 1, Math.max(ix(a[0], 0), ix(b[0], 0))), Math.min(dims[1] - 1, Math.max(ix(a[1], 1), ix(b[1], 1))),
    Math.min(dims[2] - 1, Math.max(zLo, zHi))];
  return lo.every((v, i) => v <= hi[i]) ? { lo, hi } : null;
}

/** A column of voxels (one BEV cell, z from zLo to zHi). */
export function voxelColumn(ix: number, iy: number, zLo: number, zHi: number): number[][] {
  const out: number[][] = [];
  for (let z = Math.min(zLo, zHi); z <= Math.max(zLo, zHi); z++) out.push([ix, iy, z]);
  return out;
}

/** Road height near (x, y) in a scene: a low percentile of point z within radius r, or null if empty. */
export function groundNear(points: Float32Array, count: number, x: number, y: number, r = 3, q = 0.1): number | null {
  const zs: number[] = [];
  const r2 = r * r;
  for (let i = 0; i < count; i++) {
    const dx = points[i * 4] - x, dy = points[i * 4 + 1] - y;
    if (dx * dx + dy * dy <= r2) zs.push(points[i * 4 + 2]);
  }
  if (zs.length < 5) return null;
  zs.sort((a, b) => a - b);
  return zs[Math.floor(q * (zs.length - 1))];
}
