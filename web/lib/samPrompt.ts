// Accumulating the clicks that make up one SAM prompt.
//
// Every layer below this has supported multi-point prompts from the start. `SegmentIn` carries `points`
// and `labels`, `sam_service.segment` forwards both to the model, the endpoint refuses a request whose
// labels and points differ in length, and the canvas has always sent label 0 on shift-click.
//
// What did not exist was the list. `runSam` opened by committing the pending candidate, so every click
// ended the previous mask and began a new one, and a shift-click therefore sent a single negative point
// with no positive anchor - a prompt that says only where the object is NOT, which SAM cannot act on.
// The result looked like negative clicks were unsupported when in fact they were never assembled.
//
// The arithmetic lives here rather than in the page because the interesting case is a refusal, and a
// refusal is exactly the kind of thing that is easy to get wrong and impossible to see: the mask simply
// does not change, and the annotator cannot tell whether they missed or the tool ignored them.

export type SamPrompt = { points: number[][]; labels: number[] };

export type AddResult =
  | { ok: true; prompt: SamPrompt }
  | { ok: false; reason: string };

/** An empty prompt. */
export const EMPTY: SamPrompt = { points: [], labels: [] };

/**
 * Add one click to a prompt.
 *
 * Returns the new prompt, or a refusal with a sentence to show. The one refusal is a subtraction with
 * nothing to subtract from: SAM needs at least one positive point before a negative one means anything,
 * and sending the negative alone returns either nothing or an arbitrary region.
 */
export function addPoint(prompt: SamPrompt | null, pt: number[], label: number): AddResult {
  const prev = prompt ?? EMPTY;
  if (label === 0 && !prev.labels.includes(1)) {
    return {
      ok: false,
      reason: "shift-click removes part of a mask - click the object first, then shift-click what to exclude",
    };
  }
  return { ok: true, prompt: { points: [...prev.points, pt], labels: [...prev.labels, label] } };
}

/** Drop the last click, for taking back a misplaced one without starting over. */
export function undoPoint(prompt: SamPrompt | null): SamPrompt | null {
  if (!prompt || prompt.points.length <= 1) return null;
  return { points: prompt.points.slice(0, -1), labels: prompt.labels.slice(0, -1) };
}

/** How the prompt reads on screen: "3 include, 1 exclude". */
export function describe(prompt: SamPrompt | null): string {
  if (!prompt || !prompt.points.length) return "";
  const inc = prompt.labels.filter((l) => l === 1).length;
  const exc = prompt.labels.length - inc;
  return exc ? `${inc} include, ${exc} exclude` : `${inc} include`;
}
