// What the properties panel counts, in one place, because the panel and the page footer were free to
// disagree about it.
//
// The design this implements draws a three-segment review bar labelled "confirmed / reviewed / low conf".
// Two of those three cannot be computed from the data:
//
//   - There is no "reviewed" state. `ck_object_state` admits review, auto_accept, accepted, rejected,
//     annotate and submitted, and nothing else; lib/reviewVerdict.ts documents a bug that came from
//     inventing a state outside that list, where every undo raised a CheckViolationError and the schema
//     defending itself was the only reason the corpus survived.
//   - "low conf" is a confidence threshold, not a state. An object can be auto_accept AND low conf, so a
//     bar mixing the two double-counts and its segments do not sum to the total.
//
// So the bar ships three buckets that are mutually exclusive and do sum: who ruled on this object.
// `accepted` is a person, `auto_accept` is the gate and nobody since, everything else is open work. That
// is the same distinction StateBadge already draws with a solid-versus-dashed border, for the same
// reason: the two acceptances rendered identically and an object sitting at 0.34 looked exactly like one
// a reviewer had signed off.
//
// Low confidence keeps its own separate readout on the objects card, which is also where the design puts
// it.

import type { EdObject } from "../useEditor";

// The threshold the "conf < 0.5" selection chip passes as its value, and the one the reducer's `lowConf`
// branch defaults to. Exported so the chip and the header badge read the same number: two literals here
// means the header says "4 low conf" while the chip selects 7, and nothing in the UI explains the gap.
export const LOW_CONF = 0.5;

// The label-quality bands. These appeared twice in the page, on the selected-object badge and on the
// per-row dot, and splitting them across two components is how they drift apart.
export const QUALITY_GOOD = 0.4;
export const QUALITY_WEAK = 0.25;

export type ReviewCounts = {
  total: number;
  /** state "accepted": a person ruled on it. Same predicate as the page footer's "N confirmed". */
  confirmed: number;
  /** state "auto_accept": the gate accepted it and no human has looked since. */
  auto: number;
  /** review, annotate, submitted, rejected, and anything the server sends that we do not know. */
  open: number;
};

export function reviewCounts(objects: EdObject[]): ReviewCounts {
  let confirmed = 0;
  let auto = 0;
  for (const o of objects) {
    if (o.state === "accepted") confirmed += 1;
    else if (o.state === "auto_accept") auto += 1;
  }
  // Derived rather than counted in a third branch, so the three can never fail to sum to the total no
  // matter what state string the server introduces next.
  return { total: objects.length, confirmed, auto, open: objects.length - confirmed - auto };
}

// `conf ?? 1` matches the reducer's own defence in useEditor. EdObject types conf as required, so this
// should be unreachable, but the reducer guards it and a silent divergence between the two would show up
// as a count that is off by one with no way to see why.
export function lowConfCount(objects: EdObject[]): number {
  return objects.filter((o) => (o.conf ?? 1) < LOW_CONF).length;
}

export type QualityTone = "good" | "weak" | "bad";

export function qualityTone(q: number): QualityTone {
  return q >= QUALITY_GOOD ? "good" : q >= QUALITY_WEAK ? "weak" : "bad";
}

// Segment widths as percentages. Pulled out of the component because a frame with no objects divides by
// zero, and a NaN in a style attribute renders as a bar of whatever width the browser last had.
export function reviewWidths(c: ReviewCounts): { confirmed: number; auto: number; open: number } {
  if (c.total <= 0) return { confirmed: 0, auto: 0, open: 0 };
  const pct = (n: number) => (n / c.total) * 100;
  return { confirmed: pct(c.confirmed), auto: pct(c.auto), open: pct(c.open) };
}
