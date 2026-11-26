// The pure parts of the video and editor tools (agent A): the Sweep order, what a Track-mask click asks
// the server for, and which fan-out candidates start ticked. Kept out of the 2,500-line frame page so each
// rule can be tested without mounting the editor.

import type { FanoutPreview, TrackRequest } from "./videoApi";

// ---- Sweep: one key per decision over a frame's auto-labels ----

/** States a machine label sits in until a person rules on it. A person's own label is never in a sweep. */
export const SWEEP_STATES: ReadonlySet<string> = new Set(["review", "annotate", "auto_accept"]);

type Sweepable = { id: string; state: string; bbox: number[]; isNew?: boolean; visible?: boolean };

/**
 * The frame's undecided machine labels in reading order: left to right, then top to bottom.
 *
 * Left to right because that is how the eye crosses a dashcam frame, and a sweep that jumps about makes the
 * person hunt for the highlighted box before they can judge it. `decided` holds what this sweep already
 * ruled on but whose state has not come back from the server yet (a relabel waits for the autosave).
 */
export function sweepQueue(objects: readonly Sweepable[], decided: ReadonlySet<string> = new Set()): string[] {
  return objects
    .filter((o) => !o.isNew && o.visible !== false && SWEEP_STATES.has(o.state) && !decided.has(o.id))
    .slice()
    .sort((a, b) => (a.bbox[0] - b.bbox[0]) || (a.bbox[1] - b.bbox[1]))
    .map((o) => o.id);
}

/** The next object to put in front of the person: the one after `current` in the queue, else the first. */
export function nextSweepTarget(queue: readonly string[], current: string | null): string | null {
  if (!queue.length) return null;
  const i = current == null ? -1 : queue.indexOf(current);
  // A decided object has already left the queue, so "after it" is the head; a skipped one is still in it,
  // so the walk moves on and wraps round to what was skipped.
  if (i < 0) return queue[0];
  return queue[(i + 1) % queue.length];
}

// ---- Track mask: what a click asks for ----

export type LastTrack = { track_id: string; window_frame_ids: string[] } | null;

/**
 * The request a click with the Track-mask tool sends.
 *
 * A click on a frame inside the window of the track this tool just made is a correction of that track:
 * it re-prompts it (the server replays the earlier clicks and adds this one). Anywhere else it starts a new
 * track of the class in hand. Shift-click is a negative point, as with the SAM point tool.
 */
export function trackRequestForClick(args: {
  frameId: string; pt: number[]; label: number; className: string | null; last: LastTrack;
  before: number; after: number;
}): TrackRequest | null {
  const prompt = { frame_id: args.frameId, points: [[args.pt[0], args.pt[1]]], labels: [args.label ? 1 : 0] };
  if (args.last && args.last.window_frame_ids.includes(args.frameId)) {
    return { frame_id: args.frameId, track_id: args.last.track_id, prompts: [prompt], before: args.before,
             after: args.after };
  }
  // A negative click with nothing to correct says what the object is not, which is no place to start.
  if (!args.label || !args.className) return null;
  return { frame_id: args.frameId, class_name: args.className, prompts: [prompt], before: args.before,
           after: args.after };
}

// ---- Fan-out: what starts ticked ----

export type FanoutGroup = "track" | "lookalikes" | "confusion";
export type FanoutTicks = Record<FanoutGroup, string[]>;

/**
 * The candidates ticked when the card opens: every selectable member of the track and of the same
 * confusion, and the look-alikes too. A person's label, or one already at the new class, is never ticked.
 */
export function defaultTicks(p: FanoutPreview): FanoutTicks {
  const pick = (xs: { object_id: string; selectable: boolean }[]) => xs.filter((x) => x.selectable).map((x) => x.object_id);
  return { track: pick(p.track.objects), lookalikes: pick(p.lookalikes.objects), confusion: pick(p.confusion.objects) };
}

export function fanoutCount(t: FanoutTicks): number {
  return t.track.length + t.lookalikes.length + t.confusion.length;
}

export function toggleTick(t: FanoutTicks, group: FanoutGroup, id: string): FanoutTicks {
  const has = t[group].includes(id);
  return { ...t, [group]: has ? t[group].filter((x) => x !== id) : [...t[group], id] };
}

/** One sentence for the result, naming what was skipped so a short count is never a mystery. */
export function fanoutMessage(r: { track: number; others: number; skipped_human?: string[]; refused?: unknown[];
                                   skipped_stale?: unknown[]; class_name: string }): string {
  const parts = [`${r.track + r.others} relabelled ${r.class_name}`];
  if (r.track) parts.push(`${r.track} on the track`);
  if (r.others) parts.push(`${r.others} look-alike or same-confusion`);
  const skipped = (r.skipped_human?.length ?? 0) + (r.refused?.length ?? 0) + (r.skipped_stale?.length ?? 0);
  if (skipped) parts.push(`${skipped} left alone (a person's label, refused by the ontology, or changed meanwhile)`);
  return parts.join("; ");
}
