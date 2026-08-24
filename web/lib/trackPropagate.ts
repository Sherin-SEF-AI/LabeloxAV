// Deciding whether a class correction should travel, and saying what happened when it does.
//
// Measured before this existed: 413 tracks carried an unambiguous human class, and of the 44,097 objects
// on them only 5,798 had it. The median track is 93 frames and the median number of frames a person
// actually touched is 1, so 86.9% of every correction ever made sat on one frame while the other 92 kept
// the detector's guess. Renaming an object fixed the frame in front of you and nothing else.
//
// Pure, and separate from the editor, because the two rules worth testing are predicates: when NOT to
// call, and what the operator is told. Both are easy to get subtly wrong and invisible when wrong.

export type PropagationTarget = {
  id: string;
  track_id?: string | null;
  isNew?: boolean;
};

export type PropagationResult = {
  relabeled: number;
  class_name: string;
  clamped?: boolean;
  skipped_human?: string[];
  id_switch_events?: number;
  run_id?: string | null;
};

/**
 * Whether changing this object's class should fan out across its track.
 *
 * The negative cases each correspond to a real thing in the corpus rather than defensive noise:
 * 2.0% of objects carry no track_id and have no siblings to fix, and a box drawn in this session has not
 * been saved yet so it has no track at all.
 */
export function shouldPropagate(obj: PropagationTarget | null, previousClassName: string | null,
                                nextClassName: string): boolean {
  if (!obj) return false;
  // A geometry-only save must not relabel a track. The editor's save path sends class on every dirty
  // object, so "the payload has a class" is not the question; "did the class change" is.
  if (!previousClassName || previousClassName === nextClassName) return false;
  if (obj.isNew || obj.id.startsWith("tmp-")) return false;
  if (!obj.track_id) return false;
  return true;
}

/** What the operator is told. Clauses appear only when they carry information. */
export function propagationMessage(r: PropagationResult): string {
  const frames = `${r.relabeled} more frame${r.relabeled === 1 ? "" : "s"}`;
  let msg = `${r.class_name} applied to ${frames} on this track`;
  const held = r.skipped_human?.length ?? 0;
  if (held > 0) {
    msg += `, ${held} left alone (a person had already labelled ${held === 1 ? "it" : "them"})`;
  }
  // An annotator's approval is a submission, not ground truth. Saying so is the difference between the
  // workflow being visible and the annotator thinking their edit counted for more than it did.
  if (r.clamped) msg += ", saved for review";
  // Where the tracker lost the object and picked it up again, which is where a propagated label could have
  // landed on a different physical object. 9,139 tracks carry at least one. Named rather than blocked,
  // because blocking refuses the fix on most of the corpus, and the undo is in the same toast.
  const sw = r.id_switch_events ?? 0;
  if (sw > 0) {
    msg += ` · re-identified after an occlusion at ${sw} point${sw === 1 ? "" : "s"}, worth a look`;
  }
  return msg;
}

/** The message for an object that has no track, so silence is not mistaken for the change travelling. */
export const UNTRACKED_NOTE = "this object is not tracked, so the change applies to this frame only";
