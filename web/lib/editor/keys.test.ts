import { describe, expect, it } from "vitest";

import { MODES } from "./registry";
import {
  NUDGE_BIG_PX, NUDGE_PX, isTypingTarget, ownsArrowKeys, resolveKey, type KeyContext, type KeyLike,
} from "./keys";

// The editor's keyboard handler was a closure inside an effect inside a 2,200-line page, so the only way
// to find out what a key did was to press it. Every branch it had is a sentence here.

const key = (k: string, over: Partial<KeyLike> = {}): KeyLike => ({
  key: k, code: k.length === 1 && /[a-z]/i.test(k) ? `Key${k.toUpperCase()}` : k,
  shiftKey: false, metaKey: false, ctrlKey: false, altKey: false, repeat: false, ...over,
});
const cmd = (k: string, over: Partial<KeyLike> = {}) => key(k, { metaKey: true, ...over });

const ctx = (over: Partial<KeyContext> = {}): KeyContext => ({
  typing: false, arrowsOwned: false, mode: "objects", overrides: {}, tool: "select",
  selectionCount: 0, clipboardCount: 0, ridable: false, pending: false, samPoints: 0, kpDraftPoints: 0,
  classCount: 9, objectCanvas: true, ...over,
});

const intent = (e: KeyLike, c: KeyContext = ctx()) => resolveKey(e, c)?.intent ?? null;
const prevented = (e: KeyLike, c: KeyContext = ctx()) => resolveKey(e, c)?.preventDefault ?? null;

describe("while typing", () => {
  const typing = ctx({ typing: true, selectionCount: 3, clipboardCount: 1 });

  it("nothing fires: not a tool, not delete, not a nudge, not undo", () => {
    for (const e of [key("b"), key("Delete"), key("Backspace"), key("ArrowLeft"), key("ArrowUp", { shiftKey: true }),
                     cmd("z"), cmd("v"), cmd("a"), key("Enter"), key("Escape"), key("Tab"), key("["), key("1"),
                     key(" ", { code: "Space" })]) {
      expect(resolveKey(e, typing), e.key).toBeNull();
    }
  });
});

describe("space", () => {
  it("pans, and stops the focused button from being clicked", () => {
    expect(resolveKey(key(" ", { code: "Space" }), ctx())).toEqual({ intent: { t: "pan" }, preventDefault: true });
  });
});

describe("chords", () => {
  it("undo and redo", () => {
    expect(intent(cmd("z"))).toEqual({ t: "undo" });
    expect(intent(cmd("Z", { shiftKey: true }))).toEqual({ t: "redo" });
    expect(intent(key("z", { ctrlKey: true }))).toEqual({ t: "undo" });
    expect(prevented(cmd("z"))).toBe(true);
  });

  it("save, and save-as with shift", () => {
    expect(intent(cmd("s"))).toEqual({ t: "save" });
    expect(intent(cmd("S", { shiftKey: true }))).toEqual({ t: "saveAs" });
    expect(prevented(cmd("s"))).toBe(true);
  });

  it("select all, same class and invert", () => {
    expect(intent(cmd("a"))).toEqual({ t: "selectBy", how: "all" });
    expect(intent(cmd("A", { shiftKey: true }))).toEqual({ t: "selectBy", how: "sameClass" });
    expect(intent(cmd("i"))).toEqual({ t: "selectBy", how: "invert" });
  });

  it("copy needs a selection, and leaves the browser's own copy alone", () => {
    expect(resolveKey(cmd("c"), ctx({ selectionCount: 4 }))).toEqual({ intent: { t: "copy" }, preventDefault: false });
    expect(resolveKey(cmd("c"), ctx())).toBeNull();
  });

  it("paste needs something on the clipboard", () => {
    expect(resolveKey(cmd("v"), ctx({ clipboardCount: 2 }))).toEqual({ intent: { t: "paste" }, preventDefault: true });
    expect(resolveKey(cmd("v"), ctx())).toBeNull();
  });

  it("any other chord is the browser's", () => {
    // Cmd+B must not select the box tool, Cmd+R must not select measure, Cmd+Backspace must not delete.
    for (const k of ["b", "r", "f", "1", "Backspace", "Enter", "ArrowLeft", "[", "\\"]) {
      expect(resolveKey(cmd(k), ctx({ selectionCount: 1 })), k).toBeNull();
    }
  });

  it("backslash toggles the console", () => {
    expect(resolveKey(key("\\"), ctx())).toEqual({ intent: { t: "console" }, preventDefault: true });
  });
});

describe("modes", () => {
  it("shift plus a digit switches mode, read from the physical key", () => {
    // Shift+1 arrives as "!" in `key`. The digit is in `code`.
    expect(intent(key("!", { code: "Digit1", shiftKey: true }))).toEqual({ t: "mode", key: MODES[0].key });
    expect(intent(key("&", { code: "Digit7", shiftKey: true }))).toEqual({ t: "mode", key: MODES[6].key });
    expect(prevented(key("!", { code: "Digit1", shiftKey: true }))).toBe(true);
  });

  it("a digit with no mode behind it does nothing, and does not fall through to relabel", () => {
    expect(resolveKey(key("(", { code: "Digit9", shiftKey: true }), ctx())).toBeNull();
  });
});

describe("tools", () => {
  it("resolve from the registry for the current mode", () => {
    expect(intent(key("b"))).toEqual({ t: "tool", tool: "box" });
    expect(intent(key("k"))).toEqual({ t: "tool", tool: "amodal" });
    expect(intent(key("k"), ctx({ mode: "pose" }))).toEqual({ t: "tool", tool: "keypoint" });
  });

  it("ignore caps lock and shift", () => {
    expect(intent(key("B"))).toEqual({ t: "tool", tool: "box" });
    expect(intent(key("B", { shiftKey: true }))).toEqual({ t: "tool", tool: "box" });
  });

  it("do not exist in a mode that does not offer them", () => {
    expect(resolveKey(key("b"), ctx({ mode: "review" }))).toBeNull();
  });

  it("read the rebound key, and stop answering to the old one", () => {
    const rebound = ctx({ overrides: { "tool:box": "n" } });
    expect(intent(key("n"), rebound)).toEqual({ t: "tool", tool: "box" });
    expect(resolveKey(key("b"), rebound)).toBeNull();
  });

  it("do not prevent the default, as they never did", () => {
    expect(prevented(key("b"))).toBe(false);
  });
});

describe("single-key actions", () => {
  it("`a` confirms the frame, except in Review where it accepts the selected object", () => {
    expect(intent(key("a"))).toEqual({ t: "action", id: "confirmFrame" });
    expect(intent(key("a"), ctx({ mode: "review" }))).toEqual({ t: "action", id: "reviewAccept" });
  });

  it("`x` rejects in Review and is nothing anywhere else", () => {
    expect(intent(key("x"), ctx({ mode: "review" }))).toEqual({ t: "action", id: "reviewReject" });
    expect(resolveKey(key("x"), ctx())).toBeNull();
  });

  it("`f` fits", () => {
    expect(intent(key("f"))).toEqual({ t: "action", id: "fit" });
  });

  it("the helmet and occupant keys work on a two or three wheeler only", () => {
    // On a truck they would set an attribute its scope does not admit, and the save would be rejected
    // with no visible cause. So the key does nothing rather than failing later.
    expect(intent(key("h"), ctx({ ridable: true }))).toEqual({ t: "action", id: "toggleHelmets" });
    expect(intent(key("o"), ctx({ ridable: true }))).toEqual({ t: "action", id: "bumpOccupants" });
    expect(resolveKey(key("h"), ctx())).toBeNull();
    expect(resolveKey(key("o"), ctx())).toBeNull();
  });

  it("read the rebound key", () => {
    const rebound = ctx({ overrides: { "action:fit": "z" } });
    expect(intent(key("z"), rebound)).toEqual({ t: "action", id: "fit" });
    expect(resolveKey(key("f"), rebound)).toBeNull();
  });
});

describe("arrow keys nudge the selection", () => {
  const sel = ctx({ selectionCount: 2 });

  it("one image pixel in the direction pressed", () => {
    expect(intent(key("ArrowLeft"), sel)).toEqual({ t: "nudge", dx: -NUDGE_PX, dy: 0, coalesce: false });
    expect(intent(key("ArrowRight"), sel)).toEqual({ t: "nudge", dx: NUDGE_PX, dy: 0, coalesce: false });
    expect(intent(key("ArrowUp"), sel)).toEqual({ t: "nudge", dx: 0, dy: -NUDGE_PX, coalesce: false });
    expect(intent(key("ArrowDown"), sel)).toEqual({ t: "nudge", dx: 0, dy: NUDGE_PX, coalesce: false });
    expect(NUDGE_PX).toBe(1);
  });

  it("ten with shift", () => {
    expect(intent(key("ArrowRight", { shiftKey: true }), sel)).toEqual({ t: "nudge", dx: NUDGE_BIG_PX, dy: 0, coalesce: false });
    expect(intent(key("ArrowUp", { shiftKey: true }), sel)).toEqual({ t: "nudge", dx: 0, dy: -NUDGE_BIG_PX, coalesce: false });
    expect(NUDGE_BIG_PX).toBe(10);
  });

  it("a held key is marked as a continuation, so it undoes as one gesture", () => {
    expect(intent(key("ArrowRight", { repeat: true }), sel)).toEqual({ t: "nudge", dx: 1, dy: 0, coalesce: true });
  });

  it("stops the page scrolling under the canvas", () => {
    expect(prevented(key("ArrowDown"), sel)).toBe(true);
  });

  it("does nothing with nothing selected, and leaves the key to the browser", () => {
    expect(resolveKey(key("ArrowLeft"), ctx())).toBeNull();
  });

  it("leaves alt+arrow to the browser, which is back and forward", () => {
    expect(resolveKey(key("ArrowLeft", { altKey: true }), sel)).toBeNull();
  });

  it("does not nudge while a menu, a tab strip or a slider is using the arrows", () => {
    expect(resolveKey(key("ArrowDown"), ctx({ selectionCount: 2, arrowsOwned: true }))).toBeNull();
  });

  it("does not nudge objects that are not on screen", () => {
    // Lanes, the point cloud and the events table replace the object canvas.
    expect(resolveKey(key("ArrowDown"), ctx({ selectionCount: 2, objectCanvas: false }))).toBeNull();
  });
});

describe("the rest of the handler, in its old order", () => {
  it("zoom", () => {
    expect(intent(key("="))).toEqual({ t: "zoom", factor: 1.2 });
    expect(intent(key("+", { shiftKey: true, code: "Equal" }))).toEqual({ t: "zoom", factor: 1.2 });
    expect(intent(key("-"))).toEqual({ t: "zoom", factor: 1 / 1.2 });
  });

  it("enter keeps the mask in progress, finishes a pose, or with shift accepts the rest", () => {
    expect(intent(key("Enter"))).toEqual({ t: "acceptCandidate" });
    expect(intent(key("Enter"), ctx({ tool: "keypoint", kpDraftPoints: 4 }))).toEqual({ t: "finishKeypoints" });
    expect(intent(key("Enter"), ctx({ tool: "keypoint", kpDraftPoints: 0 }))).toEqual({ t: "acceptCandidate" });
    expect(intent(key("Enter", { shiftKey: true }), ctx({ tool: "keypoint", kpDraftPoints: 4 }))).toEqual({ t: "acceptRest" });
  });

  it("tab walks the risk order and keeps focus where it is", () => {
    expect(resolveKey(key("Tab"), ctx())).toEqual({ intent: { t: "nextRisk" }, preventDefault: true });
  });

  it("escape cancels what is in progress first, and only then clears the selection", () => {
    expect(intent(key("Escape"), ctx({ pending: true, selectionCount: 3 }))).toEqual({ t: "cancelPending" });
    expect(intent(key("Escape"), ctx({ selectionCount: 3 }))).toEqual({ t: "selectBy", how: "none" });
    expect(resolveKey(key("Escape"), ctx())).toBeNull();
  });

  it("backspace takes back a SAM click before it deletes anything", () => {
    expect(resolveKey(key("Backspace"), ctx({ samPoints: 2, selectionCount: 1 })))
      .toEqual({ intent: { t: "samUndo" }, preventDefault: true });
  });

  it("delete and backspace delete the selection, however many are in it", () => {
    expect(intent(key("Delete"), ctx({ selectionCount: 1 }))).toEqual({ t: "delete" });
    expect(intent(key("Backspace"), ctx({ selectionCount: 6 }))).toEqual({ t: "delete" });
    // Delete is not Backspace: an open SAM prompt does not claim it.
    expect(intent(key("Delete"), ctx({ selectionCount: 1, samPoints: 2 }))).toEqual({ t: "delete" });
    expect(resolveKey(key("Delete"), ctx())).toBeNull();
  });

  it("brackets step frames", () => {
    expect(intent(key("["))).toEqual({ t: "frame", dir: "prev" });
    expect(intent(key("]"))).toEqual({ t: "frame", dir: "next" });
  });

  it("1 to 9 relabel, as far as there are classes", () => {
    expect(intent(key("1"))).toEqual({ t: "relabel", index: 0 });
    expect(intent(key("9"))).toEqual({ t: "relabel", index: 8 });
    expect(resolveKey(key("5"), ctx({ classCount: 3 }))).toBeNull();
    expect(resolveKey(key("0"), ctx())).toBeNull();
  });

  it("a key nothing claims is left alone", () => {
    for (const k of ["z", "q", "F5", "Home", "Shift", ";"]) expect(resolveKey(key(k), ctx()), k).toBeNull();
  });
});

describe("where the keystroke landed", () => {
  const el = (tagName: string, over: Record<string, unknown> = {}) => ({ tagName, closest: () => null, ...over });

  it("an input, a select and a textarea are typing", () => {
    for (const tag of ["INPUT", "SELECT", "TEXTAREA"]) expect(isTypingTarget(el(tag)), tag).toBe(true);
  });

  it("a contenteditable region is typing, which the three tag names missed", () => {
    expect(isTypingTarget(el("DIV", { isContentEditable: true }))).toBe(true);
  });

  it("so is anything inside one, on an engine that does not say so itself", () => {
    const seen: string[] = [];
    const inside = el("SPAN", { closest: (sel: string) => { seen.push(sel); return {}; } });
    expect(isTypingTarget(inside)).toBe(true);
    expect(seen[0]).toContain('[contenteditable="true"]');
    expect(seen[0]).toContain('[contenteditable=""]');
  });

  it("a button, the body and the canvas are not", () => {
    for (const tag of ["BUTTON", "BODY", "CANVAS", "DIV"]) expect(isTypingTarget(el(tag)), tag).toBe(false);
  });

  it("nothing, a window and a string are not", () => {
    for (const t of [null, undefined, "x", 3, {}]) expect(isTypingTarget(t)).toBe(false);
  });

  it("a menu, a tab strip, a listbox and a slider own the arrow keys", () => {
    const seen: string[] = [];
    const inMenu = el("BUTTON", { closest: (sel: string) => { seen.push(sel); return {}; } });
    expect(ownsArrowKeys(inMenu)).toBe(true);
    for (const role of ["menu", "menubar", "listbox", "tablist", "slider", "dialog"]) {
      expect(seen[0], role).toContain(`[role="${role}"]`);
    }
  });

  it("anything else does not", () => {
    expect(ownsArrowKeys(el("BODY"))).toBe(false);
    for (const t of [null, undefined, "x", {}]) expect(ownsArrowKeys(t)).toBe(false);
  });
});
