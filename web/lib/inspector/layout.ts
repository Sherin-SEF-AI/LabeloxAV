// Arranging inspector panels, which until now could not be arranged at all.
//
// Panels rendered into `grid-cols-2 auto-rows-[minmax(200px,1fr)]`. Every panel was the same size and sat
// wherever adding it happened to put it, so a camera got the same area as a one-line raw message and the 3D
// panel was clipped by a cell sized for a plot. Order was insertion order and nothing could change it.
//
// This is the smallest model that fixes that and still round-trips through a saved layout: each panel keeps
// a column span and a row span, and the list order is the layout order. Two integers per panel, no tree, no
// nested splits.
//
// A split tree is what Foxglove has and it is genuinely more expressive, but it makes every operation a tree
// rewrite and every saved layout a schema, and the thing people actually do here is "make the 3D view bigger
// and put it first". Spans express that. If nested splits are wanted later this is a superset boundary to
// grow from, not a wall.
//
// The functions are pure and return new arrays, so the page can hold panels in state and the arrangement can
// be tested without a DOM, which is how the rest of web/lib is tested.

import type { InspectorPanel } from "@/lib/api";

/** A panel with its arrangement. `span` is columns, `rows` is height units. */
export type Arranged = InspectorPanel & { span?: number; rows?: number };

export const COLUMNS = 2;
export const MIN_SPAN = 1;
export const MAX_SPAN = COLUMNS;
export const MIN_ROWS = 1;
// Four row units is roughly a full viewport at the 200px row height. Beyond that a panel is not big, it is
// the only thing on screen, and the grid stops being a layout.
export const MAX_ROWS = 4;

export function spanOf(p: Arranged): number {
  return clamp(p.span ?? 1, MIN_SPAN, MAX_SPAN);
}

export function rowsOf(p: Arranged): number {
  return clamp(p.rows ?? 1, MIN_ROWS, MAX_ROWS);
}

function clamp(v: number, lo: number, hi: number): number {
  // A non-finite value means the state is not trustworthy, so it gets the smallest panel. Clamping upward
  // would hand the biggest cell on screen to whichever panel had the most broken state.
  if (!Number.isFinite(v)) return lo;
  return Math.min(hi, Math.max(lo, Math.round(v)));
}

/**
 * Move the panel at `from` so it sits at `to`, closing the gap behind it.
 *
 * Splice semantics rather than swap. Dragging a panel from the end to the front should push everything else
 * down by one, which is what a person means by moving it; a swap would send whatever was first to the end
 * and scramble an arrangement somebody had just built.
 */
export function movePanel<T>(panels: T[], from: number, to: number): T[] {
  if (from === to || from < 0 || to < 0 || from >= panels.length || to >= panels.length) return panels;
  const next = panels.slice();
  const [moved] = next.splice(from, 1);
  next.splice(to, 0, moved);
  return next;
}

/** Set one panel's column span, clamped to the grid. */
export function setSpan(panels: Arranged[], id: string, span: number): Arranged[] {
  return panels.map((p) => (p.id === id ? { ...p, span: clamp(span, MIN_SPAN, MAX_SPAN) } : p));
}

/** Set one panel's height in row units, clamped. */
export function setRows(panels: Arranged[], id: string, rows: number): Arranged[] {
  return panels.map((p) => (p.id === id ? { ...p, rows: clamp(rows, MIN_ROWS, MAX_ROWS) } : p));
}

/** Toggle a panel between one column and the full width, which is the resize people reach for first. */
export function cycleSpan(panels: Arranged[], id: string): Arranged[] {
  const p = panels.find((x) => x.id === id);
  if (!p) return panels;
  return setSpan(panels, id, spanOf(p) === MAX_SPAN ? MIN_SPAN : MAX_SPAN);
}

/** Grid placement for one panel. */
export function panelStyle(p: Arranged): { gridColumn: string; gridRow: string } {
  return { gridColumn: `span ${spanOf(p)}`, gridRow: `span ${rowsOf(p)}` };
}

/**
 * The arrangement fields to persist, alongside whatever identifies the panel.
 *
 * Explicit rather than saving the panel object, because `id` is a render-time counter (`p7`) that means
 * nothing on reload, and persisting it would make two layouts collide the moment both were opened.
 */
export function toSaved(p: Arranged): { type: string; topic?: string; field?: string; span: number; rows: number } {
  return { type: p.type, topic: p.topic, field: p.field, span: spanOf(p), rows: rowsOf(p) };
}

/**
 * Restore a saved layout, giving each panel a fresh render id.
 *
 * Tolerates panels saved before spans existed: they come back as 1x1, which is exactly how they looked when
 * they were saved, so an old layout opens unchanged rather than rearranged.
 */
export function fromSaved(saved: Arranged[], mkId: () => string): Arranged[] {
  return saved.map((p) => ({ ...p, id: mkId(), span: spanOf(p), rows: rowsOf(p) }));
}
