// What the correction dialog should preselect, and what it should say when it finds nothing.
//
// The dialog offers to apply one correction to every object that shares the mistake. Two rules decide
// whether it helps or does damage.
//
// A candidate a person has already ruled on is shown but never preselected. Bulk tooling that quietly
// overwrites somebody's decision is worse than bulk tooling that does nothing, and hiding those candidates
// instead would leave the operator wondering why the count does not match what they can see.
//
// And an empty list has three different meanings that used to render as one sentence: nothing was similar
// enough, the object has no embedding so nothing was compared, or the search failed. Only the first is a
// statement about similarity. The middle one is what the feature actually hit for its entire existence,
// while the dialog said "no similar objects above the threshold".

import type { CorrectionCandidate, CorrectionSuggestion } from "./types";

/**
 * The candidates to tick on open.
 *
 * Everything that is not already at the corrected value and that no human has ruled on. A person can still
 * tick a human-reviewed candidate deliberately; what they cannot do is apply to one without noticing.
 */
export function defaultSelection(candidates: readonly CorrectionCandidate[],
                                 excludeTrackId?: string | null): Set<string> {
  return new Set(candidates.filter((c) => !c.already && !c.human
                                          && !onExcludedTrack(c, excludeTrackId))
                           .map((c) => c.object_id));
}

/**
 * Whether this candidate is a frame of the track the correction was just fanned across.
 *
 * Those frames are already carrying the corrected class, and they arrive here because they are the most
 * visually similar things in the corpus to the object that was just fixed. Leaving them ticked would apply
 * a second time through `bulkReview`, which writes `source = "human"`, and 92 frames nobody looked at would
 * start claiming human authorship.
 */
export function onExcludedTrack(c: CorrectionCandidate, excludeTrackId?: string | null): boolean {
  return !!excludeTrackId && !!c.track_id && String(c.track_id) === String(excludeTrackId);
}

export type EmptyReason = { headline: string; detail: string | null };

/**
 * What to show when there is nothing to apply to. Null when there are candidates.
 *
 * `examined` separates a strict slider from an empty corpus: neighbours were found and then thresholded away
 * is a different situation from no neighbours existing, and the fix differs too.
 */
export function emptyReason(sug: CorrectionSuggestion | null): EmptyReason | null {
  if (!sug || sug.candidates.length > 0) return null;
  if (sug.reason && !sug.reason.startsWith("no objects above")) {
    return { headline: sug.reason, detail: null };
  }
  if (sug.examined && sug.examined > 0) {
    return {
      headline: "no similar objects above the threshold",
      detail: `${sug.examined} neighbours were examined; lower the similarity slider to see them`,
    };
  }
  return { headline: "no similar objects found", detail: null };
}

/** The count for the header: how many share the mistake, not how many rows came back. */
export function applicableCount(candidates: readonly CorrectionCandidate[],
                                excludeTrackId?: string | null): number {
  return candidates.filter((c) => !c.already && !onExcludedTrack(c, excludeTrackId)).length;
}

/**
 * Candidates grouped by the class they currently carry, largest group first.
 *
 * The candidate set spans classes now, because one systematic error is usually spread over several: the
 * relabel agent put objects into `bmtc_bus_shelter` from `bus`, `traffic_sign` and `hoarding`. Grouping is
 * what turns a grid of crops into a statement about which lineages are involved.
 */
export function byCurrentClass(candidates: readonly CorrectionCandidate[]): [string, CorrectionCandidate[]][] {
  const groups = new Map<string, CorrectionCandidate[]>();
  for (const c of candidates) {
    const g = groups.get(c.class_name);
    if (g) g.push(c); else groups.set(c.class_name, [c]);
  }
  return [...groups.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
}
