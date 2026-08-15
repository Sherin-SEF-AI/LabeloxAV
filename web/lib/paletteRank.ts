// Ordering for the command palette, which had none.
//
// The palette filtered destinations with a substring test over label plus hint and showed them in menu
// order. Typing "jobs" therefore put Projects first, because its hint reads "assign jobs, stages,
// scorecards" and Projects sits above Jobs in the Label menu. Enter went to Projects. Somebody looking for
// the page called Jobs, typing its exact name, was sent somewhere else.
//
// Hints have to stay searchable: they are how a person finds "Curation" while thinking "active learning".
// So the fix is not to stop matching them, it is to rank. A destination whose name IS what was typed
// outranks one that merely mentions it in prose, and the palette is keyboard-first, so first place is the
// answer for anyone who types and presses Enter without looking.

export type Destination = { href: string; label: string; hint?: string };

// Lower is better. The gaps are deliberate: every exact-name match sorts above every prefix match, and so
// on down, so a longer list can never let a weaker kind of match overtake a stronger one.
const EXACT = 0;
const PREFIX = 1;
const WORD = 2;
const INFIX = 3;
const HINT = 4;
const NONE = 99;

function score(d: Destination, q: string): number {
  const label = d.label.toLowerCase();
  if (label === q) return EXACT;
  if (label.startsWith(q)) return PREFIX;
  // A word boundary inside the label: "queue" should find "Review queue" ahead of anything that only
  // mentions queues in its description.
  if (new RegExp(`\\b${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`).test(label)) return WORD;
  if (label.includes(q)) return INFIX;
  if ((d.hint ?? "").toLowerCase().includes(q)) return HINT;
  return NONE;
}

/**
 * Destinations matching `q`, best first. An empty query is every destination in menu order, because with
 * nothing typed there is nothing to rank by and menu order is the grouping people already know.
 *
 * Ties keep their original order, so the result is stable and the palette does not reshuffle between
 * keystrokes that do not change the ranking.
 */
export function rankDestinations<T extends Destination>(items: readonly T[], q: string): T[] {
  const needle = q.trim().toLowerCase();
  if (!needle) return [...items];
  return items
    .map((d, i) => ({ d, i, s: score(d, needle) }))
    .filter((r) => r.s !== NONE)
    .sort((a, b) => a.s - b.s || a.i - b.i)
    .map((r) => r.d);
}
