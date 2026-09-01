// The frame editor's menu bar, defined as data.
//
// The editor grew a mode rail, a tool strip, a floating layers widget, a properties rail with three tool
// tabs, a canvas console and a right-click menu, and between them they still did not add up to a map of
// what the editor can do. Some things had only a keyboard shortcut, some only a panel that can be
// collapsed, and a few only existed on another page you had to know about. A menu bar is the one surface
// where everything is listed whether or not you can currently reach it, which is what makes it possible to
// learn the tool rather than be shown it.
//
// Items dispatch a window event rather than taking a callback, which is the pattern
// components/editor/CanvasConsole.tsx already documents: the editor owns the keymap and the state, so the
// menu raises an intent and the page decides what it means. That keeps this file free of React and free of
// every handler's dependencies.

import type { Menu } from "./menus";

/** Every event this menu can raise. The page listens for exactly these. */
export const EDITOR_EVENT = "lbx:editor:";

const ev = (name: string) => `${EDITOR_EVENT}${name}`;

export const EDITOR_MENUS: Menu[] = [
  {
    key: "frame", label: "Frame",
    items: [
      { key: "save", label: "Save", shortcut: "Cmd S", event: ev("save") },
      { key: "saveAs", label: "Save as a named state...", shortcut: "Cmd Shift S", event: ev("saveAs") },
      { key: "confirm", label: "Confirm frame and move on", shortcut: "Enter", event: ev("confirmFrame"),
        separatorBefore: true },
      { key: "acceptRest", label: "Accept the rest of this frame", shortcut: "Shift Enter",
        event: ev("acceptRest") },
      { key: "nextRisk", label: "Jump to the riskiest unreviewed object", shortcut: "Tab",
        event: ev("nextRisk") },
      { key: "prev", label: "Previous frame", shortcut: "[", event: ev("prevFrame") },
      { key: "next", label: "Next frame", shortcut: "]", event: ev("nextFrame") },
      { key: "drive", label: "Switch drive...", event: ev("openDrives"), separatorBefore: true },
      { key: "inspect", label: "Open this moment in the Inspector", event: ev("inspect") },
      { key: "issue", label: "Raise an issue on this frame", event: ev("issue") },
    ],
  },
  {
    key: "edit", label: "Edit",
    items: [
      { key: "undo", label: "Undo", shortcut: "Cmd Z", event: ev("undo") },
      { key: "redo", label: "Redo", shortcut: "Cmd Shift Z", event: ev("redo") },
      { key: "class", label: "Change class of the selection...", event: ev("changeClass"),
        separatorBefore: true },
      { key: "accept", label: "Accept the selection", shortcut: "A", event: ev("accept") },
      { key: "reject", label: "Reject the selection", shortcut: "X", event: ev("reject") },
      { key: "hide", label: "Hide the selection", event: ev("hide"), separatorBefore: true },
      { key: "show", label: "Show the selection", event: ev("show") },
      { key: "lock", label: "Lock the selection", event: ev("lock") },
      { key: "delete", label: "Delete the selection", shortcut: "Del", event: ev("delete"),
        separatorBefore: true },
    ],
  },
  {
    key: "select", label: "Select",
    items: [
      { key: "all", label: "All", shortcut: "Cmd A", event: ev("selectAll") },
      { key: "none", label: "None", shortcut: "Esc", event: ev("selectNone") },
      { key: "invert", label: "Invert", shortcut: "Cmd I", event: ev("selectInvert") },
      { key: "sameClass", label: "Same class as the selection", shortcut: "Cmd Shift A",
        event: ev("selectSameClass"), separatorBefore: true },
      { key: "unreviewed", label: "Unreviewed", event: ev("selectUnreviewed") },
      { key: "new", label: "Drawn in this session", event: ev("selectNew") },
      { key: "lowConf", label: "Unsure ones (conf < 0.5)", event: ev("selectLowConf") },
      { key: "rejected", label: "Rejected", event: ev("selectRejected") },
    ],
  },
  {
    key: "view", label: "View",
    items: [
      { key: "fit", label: "Fit frame to view", shortcut: "F", event: ev("fit") },
      { key: "zoomIn", label: "Zoom in", shortcut: "+", event: ev("zoomIn") },
      { key: "zoomOut", label: "Zoom out", shortcut: "-", event: ev("zoomOut") },
      { key: "panel", label: "Properties panel", event: ev("togglePanel"), separatorBefore: true },
      { key: "console", label: "Canvas console", event: "lbx:canvas-console" },
      { key: "palette", label: "Command palette", shortcut: "Cmd K", event: "lbx:palette" },
      { key: "shortcuts", label: "Keyboard shortcuts", shortcut: "?", event: "lbx:shortcuts",
        separatorBefore: true },
    ],
  },
  {
    key: "run", label: "Run",
    items: [
      // The agent actions, which until now lived only inside one tab of the properties rail. A rail that
      // can be collapsed is not where the list of what the machine can do should live.
      { key: "autolabelFrame", label: "Auto-label this frame", event: ev("autolabelFrame") },
      { key: "autolabelDrive", label: "Auto-label this whole drive...", event: ev("autolabelDrive") },
      { key: "stopAutolabel", label: "Stop the running auto-label", event: ev("stopAutolabel") },
      { key: "segRoad", label: "Segment the road surface", event: ev("segRoad"), separatorBefore: true },
      { key: "lanes", label: "Propose lanes", event: ev("proposeLanes") },
      { key: "dynamics", label: "Recompute dynamics for this drive", event: ev("dynamics") },
      { key: "reanalyse", label: "Re-check redaction and labels", event: ev("reanalyse"),
        separatorBefore: true },
    ],
  },
];
