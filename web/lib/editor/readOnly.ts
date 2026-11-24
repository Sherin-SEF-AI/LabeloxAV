// What a view-only editor still does, in one place.
//
// The editor had no frame-wide read-only mode: an object could be locked, but nothing said "this whole
// frame is somebody else's right now". Checking for that at each button would be forty checks and the
// forty-first feature would forget. So the rule sits at the three points every edit already passes
// through, and each consults this file:
//
//   * the reducer's dispatch (components/editor/useEditor.ts): nothing changes the working copy
//   * the key resolver's output (components/editor/useEditorKeys.ts): no key means an edit
//   * the tool handed to the canvas: nothing but the pointer and the ruler
//
// What stays allowed is what a person watching needs: selecting an object to read its properties, hiding
// one to see behind it, panning, zooming, measuring, moving between frames, copying.
//
// Deliberately an allowlist on the way in and a denylist nowhere: an action or intent added later is
// refused in view-only mode until somebody decides it is a view.

import type { Action, Tool } from "@/components/editor/useEditor";

import type { KeyIntent } from "./keys";

/** Tools that change nothing: the pointer and the ruler. */
const VIEW_TOOLS: ReadonlySet<string> = new Set<Tool>(["select", "measure"]);

export function isViewTool(tool: string): boolean {
  return VIEW_TOOLS.has(tool);
}

/** The tool the canvas should run: the chosen one, or the pointer when that one would draw. */
export function effectiveTool(tool: Tool, readOnly: boolean): Tool {
  return readOnly && !isViewTool(tool) ? "select" : tool;
}

// Reducer actions that leave the annotations alone. `load`, `reload`, `saved` and `reviewed` are the
// server's state arriving, not the person's edit: a save that was already in flight when the lock was
// lost still has to be able to report what it wrote, or the working copy would disagree with the server
// about which objects exist. `setLocked` and `setVisible` are this viewer's own view of the list.
const VIEW_ACTIONS: ReadonlySet<Action["t"]> = new Set<Action["t"]>([
  "load", "reload", "saved", "reviewed",
  "viewport", "select", "selectMany", "toggleSelect", "selectBy", "setVisible", "setLocked",
]);

/**
 * Whether the reducer may run this action while the editor is view-only.
 *
 * Undo, redo and jump are refused along with the edits. They do not touch the server, but they do
 * replace the working copy, and the promise made to somebody who has just lost the lock is that what
 * they had not saved is still exactly as they left it.
 */
export function actionAllowed(a: Action): boolean {
  if (a.t === "tool") return isViewTool(a.tool);
  // Clearing a pending SAM candidate is tidying up; proposing one is the start of an edit.
  if (a.t === "candidate") return a.polys === null;
  return VIEW_ACTIONS.has(a.t);
}

// Single-key actions (ACTIONS in ./keymap.ts) that are views. Everything else there confirms, reviews
// or sets an attribute.
const VIEW_KEY_ACTIONS: ReadonlySet<string> = new Set(["fit"]);

/** Whether a resolved keystroke may be carried out while the editor is view-only. */
export function intentAllowed(intent: KeyIntent): boolean {
  switch (intent.t) {
    case "pan": case "selectBy": case "console": case "copy": case "mode": case "zoom":
    case "nextRisk": case "frame": case "cancelPending":
      return true;
    case "tool":
      return isViewTool(intent.tool);
    case "action":
      return VIEW_KEY_ACTIONS.has(intent.id);
    default:
      // undo, redo, save, saveAs, paste, acceptRest, finishKeypoints, acceptCandidate, samUndo, delete,
      // relabel, nudge, and whatever is added to KeyIntent after this was written.
      return false;
  }
}

// Menu bar entries (the page's EDITOR_EVENT handlers) that are views or navigation. Three of them only
// point at where a tool lives (changeClass, reanalyse, autolabelFrame), and the drive-level ones open the
// session picker, whose jobs are not this frame's lock to guard.
const VIEW_MENU_EVENTS: ReadonlySet<string> = new Set([
  "nextRisk", "prevFrame", "nextFrame", "openDrives", "inspect", "issue", "changeClass",
  "hide", "show", "lock",
  "selectAll", "selectNone", "selectInvert", "selectSameClass", "selectUnreviewed", "selectNew",
  "selectLowConf", "selectRejected",
  "fit", "zoomIn", "zoomOut", "togglePanel", "dynamics",
  "reanalyse", "autolabelFrame", "autolabelDrive", "stopAutolabel",
]);

/** Whether a menu bar entry may run while the editor is view-only. */
export function menuEventAllowed(id: string): boolean {
  return VIEW_MENU_EVENTS.has(id);
}

// Right-click menu entries that are views. The rest reclassify, review, propagate, delete or save.
const VIEW_CANVAS_MENU: ReadonlySet<string> = new Set([
  "lock", "hide", "hideMany", "showMany", "lockMany", "viewTrack", "copyId",
  "selectAll", "selectNone", "invert", "lowConf", "fit", "showPanel", "issue",
]);

/** Whether a canvas context-menu entry may run while the editor is view-only. */
export function canvasMenuAllowed(key: string): boolean {
  return VIEW_CANVAS_MENU.has(key);
}

/** What to say when an edit is refused. Short: it is shown on every refused keystroke. */
export function readOnlyNotice(holderName: string | null): string {
  return holderName ? `view only: ${holderName} is editing this frame` : "view only: checking who is editing this frame";
}
