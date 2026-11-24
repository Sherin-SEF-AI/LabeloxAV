// What a keystroke means in the frame editor, decided without React.
//
// This was a 110-line closure inside the page's effect. It could only be exercised by mounting the whole
// editor, so it was never exercised at all, and it read the selection through a dependency array that had
// drifted ten names behind what the closure actually used. The decision is pulled out here as a function
// of the event and a plain description of the editor's state; the page keeps the part that needs the
// page, which is carrying the decision out.
//
// The order of the branches is the order the handler always had. It is load-bearing in two places and
// both are commented where they occur.

import { MODES, toolForHotkey } from "./registry";
import { actionForHotkey, effectiveModes, type Overrides } from "./keymap";

export type KeyLike = {
  key: string;
  code: string;
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey?: boolean;
  /** True for the events a held key generates after the first. */
  repeat?: boolean;
};

export type KeyContext = {
  /** Focus is somewhere text is typed. Nothing below Space fires then. */
  typing: boolean;
  /** Focus is in a widget that moves with the arrows itself, or something already handled this event. */
  arrowsOwned: boolean;
  mode: string;
  overrides: Overrides;
  tool: string;
  /** How many objects a selection-wide action would reach. */
  selectionCount: number;
  /** How many objects are on the clipboard. */
  clipboardCount: number;
  /** The primary selection is a two or three wheeler, which is what the helmet and occupant keys need. */
  ridable: boolean;
  /** Something is half made: a SAM candidate, a SAM prompt, or a pose being placed. */
  pending: boolean;
  /** Clicks in the SAM prompt being refined. */
  samPoints: number;
  /** Points placed so far in the pose being drawn. */
  kpDraftPoints: number;
  /** Classes the 1 to 9 keys can reach. */
  classCount: number;
  /** The object canvas is what is on screen, rather than the lane editor, the point cloud or a table. */
  objectCanvas: boolean;
};

export type KeyIntent =
  | { t: "pan" }
  | { t: "undo" } | { t: "redo" }
  | { t: "save" } | { t: "saveAs" }
  | { t: "selectBy"; how: "all" | "sameClass" | "invert" | "none" }
  | { t: "console" }
  | { t: "copy" } | { t: "paste" }
  | { t: "mode"; key: string }
  | { t: "tool"; tool: string }
  | { t: "action"; id: string }
  | { t: "zoom"; factor: number }
  | { t: "acceptRest" } | { t: "finishKeypoints" } | { t: "acceptCandidate" }
  | { t: "nextRisk" }
  | { t: "cancelPending" }
  | { t: "samUndo" }
  | { t: "delete" }
  | { t: "frame"; dir: "prev" | "next" }
  | { t: "relabel"; index: number }
  | { t: "nudge"; dx: number; dy: number; coalesce: boolean };

export type Resolved = { intent: KeyIntent; preventDefault: boolean };

const hit = (intent: KeyIntent, preventDefault = false): Resolved => ({ intent, preventDefault });

const ARROWS: Record<string, [number, number]> = {
  ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1],
};

/** One image pixel per press, ten with Shift. */
export const NUDGE_PX = 1;
export const NUDGE_BIG_PX = 10;

const RIDABLE_ONLY = new Set(["toggleHelmets", "bumpOccupants"]);

export function resolveKey(e: KeyLike, c: KeyContext): Resolved | null {
  // Space pans, and is the one key checked before the typing guard's early return: both read the guard.
  if (e.code === "Space" && !c.typing) return hit({ t: "pan" }, true);
  if (c.typing) return null;

  const mod = e.metaKey || e.ctrlKey;
  const k = e.key.toLowerCase();

  if (mod && k === "z") return hit(e.shiftKey ? { t: "redo" } : { t: "undo" }, true);
  // Shift makes it save-as: a named, server-side state that survives a refresh, which the ordinary save
  // does not create and undo cannot get back to.
  if (mod && k === "s") return hit(e.shiftKey ? { t: "saveAs" } : { t: "save" }, true);
  // Selection, on the modifiers people already expect from every other editor.
  if (mod && k === "a") return hit({ t: "selectBy", how: e.shiftKey ? "sameClass" : "all" }, true);
  if (mod && k === "i") return hit({ t: "selectBy", how: "invert" }, true);
  // Backslash: near Enter on every layout, and unclaimed. Toggling the console has to be reachable
  // without leaving the drawing hand, or it is a panel people open once.
  if (!mod && e.key === "\\") return hit({ t: "console" }, true);
  // Copy does not prevent the default, so with nothing selected (and so nothing to copy here) the
  // browser's own copy of a text selection still works.
  if (mod && k === "c") return c.selectionCount > 0 ? hit({ t: "copy" }) : null;
  if (mod && k === "v") return c.clipboardCount > 0 ? hit({ t: "paste" }, true) : null;
  if (mod) return null;

  // Shift+1..7 switches mode (plain 1..9 stays the quick-relabel shortcut). Read from `code`, because
  // Shift+1 arrives as "!" in `key` and as something else again on a non-US layout.
  if (e.shiftKey && /^Digit[1-9]$/.test(e.code)) {
    const target = MODES[Number(e.code.slice(5)) - 1];
    return target ? hit({ t: "mode", key: target.key }, true) : null;
  }

  const arrow = ARROWS[e.key];
  if (arrow) {
    // Alt+arrow is the browser's back and forward. A menu, a tab strip or a slider that has focus moves
    // with the arrows itself, and nudging the canvas underneath it would be a second, invisible effect of
    // one keypress.
    if (e.altKey || c.arrowsOwned || !c.objectCanvas || c.selectionCount === 0) return null;
    const step = e.shiftKey ? NUDGE_BIG_PX : NUDGE_PX;
    // Prevented so the page does not scroll under the canvas. `coalesce` is the key-repeat flag: a held
    // key is one gesture and should undo as one.
    return hit({ t: "nudge", dx: arrow[0] * step, dy: arrow[1] * step, coalesce: !!e.repeat }, true);
  }

  // Tools and single-key actions, resolved for the current mode with the annotator's rebindings applied.
  // No collision can exist between the two inside a mode (the keymap refuses to create one), so the order
  // here decides nothing; tools are first only because they always were.
  if (k.length === 1) {
    const tool = toolForHotkey(c.mode, k, effectiveModes(c.overrides));
    if (tool) return hit({ t: "tool", tool });
    const action = actionForHotkey(c.mode, k, c.overrides);
    // On a truck the helmet key would set an attribute the truck's scope does not admit, and the server
    // would reject the save with no visible cause. So it is the key doing nothing, not the key failing.
    if (action) return RIDABLE_ONLY.has(action) && !c.ridable ? null : hit({ t: "action", id: action });
  }

  if (e.key === "=" || e.key === "+") return hit({ t: "zoom", factor: 1.2 });
  if (e.key === "-") return hit({ t: "zoom", factor: 1 / 1.2 });
  if (e.key === "Enter") {
    // Shift is the "and the rest" modifier here, matching the menu. Plain Enter still commits the
    // in-progress mask or keypoint, which is the more frequent thing by far.
    if (e.shiftKey) return hit({ t: "acceptRest" });
    if (c.tool === "keypoint" && c.kpDraftPoints > 0) return hit({ t: "finishKeypoints" });
    return hit({ t: "acceptCandidate" });
  }
  if (e.key === "Tab") return hit({ t: "nextRisk" }, true);
  if (e.key === "Escape") {
    // Escape clears in the order a person means it: the in-progress thing first, then the selection.
    // Dropping both at once loses a selection somebody was still using.
    if (c.pending) return hit({ t: "cancelPending" });
    return c.selectionCount > 0 ? hit({ t: "selectBy", how: "none" }) : null;
  }
  // Take back the last SAM click without starting the mask over. Checked before the delete binding
  // below, because while a SAM prompt is open Backspace can only mean this.
  if (e.key === "Backspace" && c.samPoints > 0) return hit({ t: "samUndo" }, true);
  if (e.key === "Delete" || e.key === "Backspace") return c.selectionCount > 0 ? hit({ t: "delete" }) : null;
  if (e.key === "[") return hit({ t: "frame", dir: "prev" });
  if (e.key === "]") return hit({ t: "frame", dir: "next" });
  if (/^[1-9]$/.test(e.key)) {
    const index = parseInt(e.key, 10) - 1;
    return index < c.classCount ? hit({ t: "relabel", index }) : null;
  }
  return null;
}

// ---- where the keystroke landed -----------------------------------------------------------------------

// Structural rather than `HTMLElement`, so the two guards below can be tested without a DOM.
type TargetLike = {
  tagName?: string;
  isContentEditable?: boolean;
  closest?: (selector: string) => unknown;
};

const EDITABLE = '[contenteditable=""],[contenteditable="true"],[contenteditable="plaintext-only"]';

/**
 * Whether the event came from somewhere a person types.
 *
 * The handler checked three tag names. A contenteditable region is a text field with none of them, so
 * typing a `b` into one switched the canvas to the box tool, and Backspace deleted the selected object.
 */
export function isTypingTarget(target: unknown): boolean {
  const el = target as TargetLike | null;
  if (!el || typeof el !== "object") return false;
  if (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA") return true;
  if (el.isContentEditable) return true;
  // jsdom, and an element inside an editable ancestor on some engines, do not report isContentEditable.
  return typeof el.closest === "function" && !!el.closest(EDITABLE);
}

const ARROW_WIDGETS = '[role="menu"],[role="menubar"],[role="listbox"],[role="tablist"],[role="slider"],'
  + '[role="radiogroup"],[role="tree"],[role="grid"],[role="dialog"]';

/** Whether focus is inside a widget whose own keyboard model uses the arrow keys. */
export function ownsArrowKeys(target: unknown): boolean {
  const el = target as TargetLike | null;
  return !!el && typeof el === "object" && typeof el.closest === "function" && !!el.closest(ARROW_WIDGETS);
}
