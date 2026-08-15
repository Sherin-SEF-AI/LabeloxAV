// How finished a neighbouring frame is, for the filmstrip.
//
// The strip showed each frame's object count and nothing about its state, so a reviewer working through a
// session could not see which frames they had already confirmed or where they had stopped. Stepping back to
// check something meant reopening frames that were finished, and the only way to tell was to open one and
// look at the objects.
//
// Three states rather than done/not-done, because the partly confirmed frame is the interesting one: it is
// where somebody was interrupted, and it is invisible under a boolean.

export type TileState = "empty" | "untouched" | "partial" | "done";

export type TileProgress = {
  state: TileState;
  /** 0..1, for the bar under the tile. Always 0 when there is nothing to confirm. */
  frac: number;
  /** What the tile says out loud, for the title and the screen reader. */
  label: string;
};

export function tileProgress(nObjects: number, nConfirmed: number): TileProgress {
  const total = Math.max(0, Math.trunc(nObjects));
  // Clamped rather than trusted: a confirmed count above the total would draw a bar past the tile, and the
  // two numbers are counted per state on the server, where a state added later would land outside both.
  const done = Math.min(total, Math.max(0, Math.trunc(nConfirmed)));
  if (total === 0) return { state: "empty", frac: 0, label: "no objects" };
  if (done === 0) return { state: "untouched", frac: 0, label: `${total} objects, none confirmed` };
  if (done >= total) return { state: "done", frac: 1, label: `${total} objects, all confirmed` };
  return { state: "partial", frac: done / total, label: `${done} of ${total} confirmed` };
}
