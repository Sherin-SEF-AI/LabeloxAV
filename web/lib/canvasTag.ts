// What a box says about itself on the canvas.
//
// The canvas drew every object as a coloured rectangle and nothing else. Class and confidence lived in the
// properties panel, one object at a time, so the question a reviewer actually has ("which of these forty
// boxes is the model unsure about") could only be answered by clicking forty boxes. The model's own
// uncertainty is the single most useful thing to see while reviewing, and it was the one thing not shown.
//
// The rules here exist because a label on every box at every zoom is worse than none: text wider than the
// box it describes covers the object, and a dense frame becomes unreadable.

export type EditorTagInput = {
  class_name: string;
  conf: number;
  state: string;
  isNew?: boolean;
};

export type CanvasTag = {
  /** Whether to draw anything at all. */
  show: boolean;
  text: string;
  /** Below the review threshold: the box worth looking at first. */
  low: boolean;
};

/** Under this many screen pixels of box width, the tag is wider than what it describes. */
export const MIN_TAG_BOX_PX = 44;

/** The confidence below which a detection is worth a second look, matching the review queue's own cut. */
export const LOW_CONF = 0.5;

export function objectTag(o: EditorTagInput, boxWidthPx: number, lowThreshold = LOW_CONF): CanvasTag {
  const settled = o.state === "accepted" || o.state === "rejected";
  const hide = { show: false, text: "", low: false };
  if (!Number.isFinite(boxWidthPx) || boxWidthPx < MIN_TAG_BOX_PX) return hide;
  // A box being drawn right now has no confidence to report and the pointer is already on it.
  if (o.isNew) return hide;
  // A settled object carries a human's answer, so the model's number is history and the class is the fact.
  const conf = Number.isFinite(o.conf) ? Math.min(1, Math.max(0, o.conf)) : 0;
  const text = settled ? o.class_name : `${o.class_name} ${Math.round(conf * 100)}`;
  return { show: true, text, low: !settled && conf < lowThreshold };
}
