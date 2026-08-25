// What a right-click on the canvas should offer, decided from what was under the pointer.
//
// Every action here already existed and lived somewhere else: the properties rail, the tool strip, a
// keyboard shortcut, or a page you had to navigate to. That is fine when you are setting up and wrong when
// you are working, because the thing you want to act on is under the cursor and the control for it is four
// hundred pixels away in a panel you may have collapsed.
//
// Kept pure and out of the component for the reason the other pickers are: what appears when is a decision
// with real cases in it (nothing selected, one object, several, an object somebody else locked), and each
// of those is a sentence you can assert rather than a screenshot you have to look at.

export type CanvasMenuItem = {
  key: string;
  label: string;
  /** Shown right-aligned. The point is to teach the shortcut, not to hide it in a help overlay. */
  hint?: string;
  /** Present and greyed rather than absent, so the menu is a stable map of what exists. */
  disabled?: boolean;
  /** Why it is greyed. Without this a disabled row is a dead end. */
  why?: string;
  danger?: boolean;
  /** Opens a submenu instead of firing. */
  submenu?: "class";
};

export type CanvasMenuSection = { key: string; items: CanvasMenuItem[] };

export type CanvasTarget = {
  /** The object under the pointer, if any. */
  object: { id: string; class_name: string; locked?: boolean; visible: boolean; isNew?: boolean;
            track_id?: string | null; state: string } | null;
  /** How many objects are selected right now. */
  selectedCount: number;
  /** Whether the right-clicked object is part of that selection. */
  targetInSelection: boolean;
};

/**
 * The menu for this click.
 *
 * Three shapes, because three different things are being acted on: a multi-selection the click landed
 * inside, a single object, or the frame itself. Mixing them produces a menu where half the rows silently
 * apply to something other than what you aimed at.
 */
export function canvasMenu(t: CanvasTarget): CanvasMenuSection[] {
  if (t.object && t.selectedCount > 1 && t.targetInSelection) return bulkMenu(t.selectedCount);
  if (t.object) return objectMenu(t.object);
  return frameMenu(t.selectedCount);
}

function objectMenu(o: NonNullable<CanvasTarget["object"]>): CanvasMenuSection[] {
  // A locked object is locked against being changed, not against being read or unlocked. Greying the
  // unlock row too would make the lock unreachable from the only place it is visible.
  const lockedWhy = "this object is locked; unlock it first";
  const newWhy = "save the frame first, this object does not exist on the server yet";
  return [
    { key: "label", items: [
      { key: "class", label: "change class", hint: "1-9", submenu: "class",
        disabled: !!o.locked, why: o.locked ? lockedWhy : undefined },
      { key: "accept", label: "accept", hint: "A", disabled: !!o.locked,
        why: o.locked ? lockedWhy : undefined },
      { key: "reject", label: "reject", hint: "X", disabled: !!o.locked,
        why: o.locked ? lockedWhy : undefined },
    ] },
    { key: "state", items: [
      { key: "lock", label: o.locked ? "unlock" : "lock" },
      { key: "hide", label: o.visible ? "hide" : "show" },
    ] },
    { key: "track", items: [
      { key: "propagate", label: "auto-track forward", disabled: !!o.isNew,
        why: o.isNew ? newWhy : undefined },
      { key: "viewTrack", label: "open this track", disabled: !o.track_id,
        why: o.track_id ? undefined : "this object is not part of a track" },
      { key: "copyId", label: "copy object id" },
    ] },
    { key: "danger", items: [
      { key: "delete", label: "delete", hint: "Del", danger: true, disabled: !!o.locked,
        why: o.locked ? lockedWhy : undefined },
    ] },
  ];
}

function bulkMenu(n: number): CanvasMenuSection[] {
  return [
    // Reclassifying a multi-selection had no control anywhere. `relabelSelected` acts on the primary
    // selection only, and the bulk toolbar offered hide, show, lock and delete, so marquee-selecting
    // twelve boxes and pressing a class key changed exactly one of them.
    { key: "label", items: [
      { key: "class", label: `change class of ${n}`, submenu: "class" },
      { key: "acceptMany", label: `accept ${n}` },
      { key: "rejectMany", label: `reject ${n}` },
    ] },
    { key: "state", items: [
      { key: "hideMany", label: `hide ${n}` },
      { key: "showMany", label: `show ${n}` },
      { key: "lockMany", label: `lock ${n}` },
    ] },
    { key: "danger", items: [
      { key: "deleteMany", label: `delete ${n}`, hint: "Del", danger: true },
    ] },
  ];
}

function frameMenu(selectedCount: number): CanvasMenuSection[] {
  return [
    { key: "select", items: [
      { key: "selectAll", label: "select all", hint: "Cmd A" },
      { key: "selectNone", label: "clear selection", hint: "Esc",
        disabled: selectedCount === 0, why: "nothing is selected" },
      { key: "invert", label: "invert selection", hint: "Cmd I" },
      { key: "lowConf", label: "select unsure ones (conf < 0.5)" },
    ] },
    { key: "view", items: [
      { key: "fit", label: "fit frame to view", hint: "F" },
      // Not the layers widget: that already floats on the canvas. The properties rail is the one thing on
      // this screen that can be collapsed out of reach, and collapsing it is the default below 1100px.
      { key: "showPanel", label: "show properties panel" },
    ] },
    { key: "frame", items: [
      { key: "save", label: "save", hint: "Cmd S" },
      { key: "issue", label: "raise an issue on this frame" },
    ] },
  ];
}

/** Every enabled key in a menu, in order, for keyboard walking. */
export function enabledKeys(sections: CanvasMenuSection[]): string[] {
  return sections.flatMap((s) => s.items.filter((i) => !i.disabled).map((i) => i.key));
}

/**
 * Where to draw the menu so it stays on screen.
 *
 * A context menu opened near the right or bottom edge is the normal case, not the edge case: the canvas
 * fills the window and people right-click what they are looking at, which is often near an edge.
 */
export function menuPosition(at: { x: number; y: number }, size: { w: number; h: number },
                             viewport: { w: number; h: number }): { left: number; top: number } {
  const pad = 8;
  const left = at.x + size.w + pad > viewport.w ? Math.max(pad, at.x - size.w) : at.x;
  const top = at.y + size.h + pad > viewport.h ? Math.max(pad, viewport.h - size.h - pad) : at.y;
  return { left, top };
}
