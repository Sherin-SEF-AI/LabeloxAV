// Pure helpers for the quality tools' pages and overlays (consensus heatmap, evidence boxes, VLM verdicts,
// rare-scene selection). Kept free of React so they are unit-tested in quality.test.ts.

import type { ClassTally, HeatCell, VlmCheck } from "./qualityApi";

export type Tone = "pass" | "warn" | "block" | "muted" | "info";

/** The colour a heatmap cell takes: agreement rate picks the hue, the number of decided clusters its strength,
 *  so a 100% from one cluster reads fainter than a 100% from forty. */
export function heatStyle(cell: HeatCell): { tone: Tone; background: string } {
  if (!cell || cell.rate == null) return { tone: "muted", background: "transparent" };
  const decided = cell.accepted + cell.disputed;
  const tone: Tone = cell.rate >= 0.8 ? "pass" : cell.rate >= 0.5 ? "warn" : "block";
  const alpha = Math.round((0.14 + 0.5 * Math.min(1, decided / 10)) * 100) / 100;
  return { tone, background: `rgb(var(--c-${tone}) / ${alpha})` };
}

/** Per-class totals summed across groups, for the heatmap's last column. */
export function classTotals(cells: HeatCell[][]): (ClassTally & { rate: number | null })[] {
  return cells.map((row) => {
    const t = { accepted: 0, disputed: 0, dropped: 0 };
    for (const c of row) {
      if (!c) continue;
      t.accepted += c.accepted; t.disputed += c.disputed; t.dropped += c.dropped;
    }
    const decided = t.accepted + t.disputed;
    return { ...t, rate: decided ? t.accepted / decided : null };
  });
}

export type Viewport = { scale: number; ox: number; oy: number };

/** An image-space xyxy box placed in screen space under the editor's viewport (screen = o + scale * image). */
export function boxToScreen(bbox: number[], v: Viewport): { x: number; y: number; w: number; h: number } {
  const s = v.scale > 0 && Number.isFinite(v.scale) ? v.scale : 1;
  const [x1, y1, x2, y2] = bbox;
  return { x: v.ox + x1 * s, y: v.oy + y1 * s, w: Math.max(0, (x2 - x1) * s), h: Math.max(0, (y2 - y1) * s) };
}

/** The scale that fits an image of (w, h) into a box of (bw, bh), never enlarging past 1. */
export function fitScale(w: number, h: number, bw: number, bh: number): number {
  if (!(w > 0 && h > 0 && bw > 0 && bh > 0)) return 1;
  return Math.min(1, bw / w, bh / h);
}

/** A window of an image around a box, padded, clamped to the image: what a zoomed evidence view shows. */
export function focusWindow(boxes: number[][], w: number, h: number, pad = 0.6): number[] {
  if (!boxes.length || !(w > 0 && h > 0)) return [0, 0, w, h];
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const b of boxes) {
    x1 = Math.min(x1, b[0]); y1 = Math.min(y1, b[1]); x2 = Math.max(x2, b[2]); y2 = Math.max(y2, b[3]);
  }
  const px = Math.max(40, (x2 - x1) * pad), py = Math.max(40, (y2 - y1) * pad);
  return [Math.max(0, x1 - px), Math.max(0, y1 - py), Math.min(w, x2 + px), Math.min(h, y2 + py)];
}

/** Colour token per evidence role: the flagged object stands out, what it was compared to is secondary. */
export function roleTone(role: string): Tone {
  if (role === "subject") return "block";
  if (role === "overlaps" || role === "contains") return "warn";
  return "info";
}

export function verdictTone(v: VlmCheck["verdict"] | undefined): Tone {
  return v === "correct" ? "pass" : v === "incorrect" ? "block" : v === "unsure" ? "warn" : "muted";
}

export function verdictLabel(c: Pick<VlmCheck, "verdict" | "alternative">): string {
  if (c.verdict === "correct") return "Looks right";
  if (c.verdict === "incorrect") return c.alternative ? `Looks wrong, maybe ${c.alternative.replace(/_/g, " ")}` : "Looks wrong";
  return "Not sure";
}

/** One sentence for a merge's counts, the way the page reports a finished or previewed merge. */
export function mergeSummary(c: Record<string, number> | null | undefined): string {
  if (!c) return "";
  const parts = [`${c.accepted ?? 0} agreed`, `${c.disputed ?? 0} sent to review`];
  if (c.dropped) parts.push(`${c.dropped} model-only boxes nobody drew`);
  if (c.created != null) parts.push(`${c.created} new consensus labels`);
  if (c.removed) parts.push(`${c.removed} stale ones removed`);
  return parts.join(", ");
}

/** Toggle an id in a selection set, returning a new set (React state stays immutable). */
export function toggleId(sel: Set<string>, id: string): Set<string> {
  const next = new Set(sel);
  if (next.has(id)) next.delete(id); else next.add(id);
  return next;
}

/** Scene axes worth showing on a rare-frame tile, as "night · rain". */
export function sceneLine(scene: Record<string, unknown> | null | undefined): string {
  if (!scene) return "";
  return ["time_of_day", "weather", "road_type"].map((k) => scene[k]).filter((v) => typeof v === "string" && v)
    .join(" · ");
}

/** Where a frame sits in the ranking, as "rarest 0.1%": a percentile of 0.9999 printed as "rarer than 100%" says
 *  nothing, while the share of frames at least as rare does. */
export function rarestShare(percentile: number | null | undefined): string {
  if (percentile == null || !Number.isFinite(percentile)) return "";
  const share = Math.max(0, 1 - percentile) * 100;
  if (share < 0.1) return "rarest 0.1%";
  return `rarest ${share < 10 ? share.toFixed(1) : Math.round(share)}%`;
}
