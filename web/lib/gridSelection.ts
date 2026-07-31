// Cursor and selection arithmetic for a keyboard-driven review grid.
//
// A grid is faster than one-crop-at-a-time only if the hands never leave the keyboard, which means the
// cursor, the range select and the wrap behaviour have to be exactly right. They are also the part that
// breaks silently: an off-by-one at a row end applies a verdict to the wrong crop, and the reviewer has no
// way to notice, because the tile they were looking at is gone either way.
//
// So it lives here, as arithmetic over indices, and not inside a component where it could only be tested by
// rendering one.

export type Dir = "left" | "right" | "up" | "down";

/** Where the cursor lands after a keypress. Never leaves the grid. */
export function moveCursor(index: number, dir: Dir, count: number, cols: number): number {
  if (count <= 0) return 0;
  const c = Math.max(1, cols);
  switch (dir) {
    case "left":
      // Wrapping to the previous row is what makes left and right a single reading order rather than a
      // per-row one, which is how somebody scanning a grid actually moves.
      return Math.max(0, index - 1);
    case "right":
      return Math.min(count - 1, index + 1);
    case "up":
      // Clamped, not wrapped: at the top row there is nothing above, and jumping to the bottom would move
      // the cursor a screen away from where the reviewer is looking.
      return index - c >= 0 ? index - c : index;
    case "down": {
      const next = index + c;
      // A partial last row must still be reachable from the row above it.
      if (next < count) return next;
      return index < count - 1 && Math.floor((count - 1) / c) > Math.floor(index / c) ? count - 1 : index;
    }
  }
}

/** The indices a shift-range covers, inclusive, in either direction. */
export function rangeBetween(anchor: number, cursor: number): number[] {
  const [lo, hi] = anchor <= cursor ? [anchor, cursor] : [cursor, anchor];
  return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
}

/** Toggle one index in a selection. */
export function toggle(selection: ReadonlySet<number>, index: number): Set<number> {
  const next = new Set(selection);
  if (next.has(index)) next.delete(index);
  else next.add(index);
  return next;
}

/**
 * What a verdict applies to: the selection when there is one, otherwise the single tile under the cursor.
 *
 * The fallback is the whole point of the mode. Most tiles are correct and get one keystroke each; a
 * selection is for the occasional run of wrong ones. Requiring a selection first would put two keystrokes
 * on the common case to save one on the rare one.
 */
export function targetIndices(selection: ReadonlySet<number>, cursor: number): number[] {
  return selection.size > 0 ? [...selection].sort((a, b) => a - b) : [cursor];
}

/**
 * Where the cursor goes after N tiles are decided and removed from the queue.
 *
 * Clamped to the new length, so deciding the last tile leaves the cursor on the new last tile rather than
 * past the end. Returning 0 there instead would send somebody who just finished back to the beginning.
 */
export function cursorAfterRemoval(cursor: number, removed: readonly number[], newCount: number): number {
  if (newCount <= 0) return 0;
  const before = removed.filter((i) => i < cursor).length;
  return Math.max(0, Math.min(newCount - 1, cursor - before));
}
