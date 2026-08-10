// How many tool groups stay in the strip, and how many move into an overflow flyout.
//
// The editor's layout rule is that the strip is one row forever and absorbs new capability by grouping,
// never by growing. It held the first half of that and not the second: the row is `overflow-x-auto`, so at
// 1280px (a 13in laptop) the last group scrolled out of sight with no scrollbar, no chevron, and nothing to
// say a tool was there at all. A tool you cannot see is a tool you do not have, even though its hotkey still
// works, and the whole point of a visible strip is that it teaches what is available.
//
// So groups that do not fit collapse into a single overflow button. That is the same mechanism the strip
// already uses for alternates within a group, applied one level up, which is why it needs no new visual
// vocabulary.
//
// Pure and width-driven so it can be tested without a DOM. The component measures, this decides.

export const GAP_PX = 6;          // gap-1.5 between groups
// The collapsed "more tools" button plus the gap in front of it, measured in the browser at 37px + 6px.
// Understating this is not a rounding error: the button is what the evicted groups make room for, so a
// short reserve puts the row back over its container and reintroduces the clip.
export const OVERFLOW_BTN_PX = 43;

/** Width of `n` groups laid out in a row, gaps included. */
function rowWidth(widths: number[], n: number, gap: number): number {
  if (n <= 0) return 0;
  let w = 0;
  for (let i = 0; i < n; i++) w += widths[i] + (i > 0 ? gap : 0);
  return w;
}

/**
 * Number of leading groups that fit in `available` px.
 *
 * Returns `widths.length` when everything fits, which is the case that must not reserve room for an overflow
 * button that will never be drawn. Anything less means the remainder belongs in the flyout.
 *
 * A zero or negative `available` means the container has not been laid out yet (the first paint, or a hidden
 * tab). Showing everything is the safe answer there: the row is scrollable, so an over-full strip is
 * recoverable, while an empty one looks like the tools are gone.
 */
export function fitCount(widths: number[], available: number,
                         gap: number = GAP_PX, overflowPx: number = OVERFLOW_BTN_PX): number {
  if (!widths.length) return 0;
  // Unmeasured groups report 0. Deciding from those would collapse a strip that actually fits.
  if (available <= 0 || widths.some((w) => w <= 0)) return widths.length;

  if (rowWidth(widths, widths.length, gap) <= available) return widths.length;

  const budget = available - overflowPx;
  let n = 0;
  while (n < widths.length && rowWidth(widths, n + 1, gap) <= budget) n++;
  return n;
}

/** Split groups into the ones that stay and the ones that collapse, preserving order. */
export function splitGroups<T>(groups: T[], widths: number[], available: number,
                               gap: number = GAP_PX, overflowPx: number = OVERFLOW_BTN_PX):
                               { visible: T[]; hidden: T[] } {
  const n = fitCount(widths, available, gap, overflowPx);
  return { visible: groups.slice(0, n), hidden: groups.slice(n) };
}
