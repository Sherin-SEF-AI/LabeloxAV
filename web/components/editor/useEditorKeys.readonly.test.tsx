import { fireEvent, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetKeymap } from "@/lib/editor/keymap";
import type { KeyIntent } from "@/lib/editor/keys";

import { useEditorKeys, type EditorKeyState } from "./useEditorKeys";

// The keyboard on a view-only frame, on the real window listener. Which intents are views is decided
// and tested in lib/editor/readOnly; this is that the listener asks, and asks at the moment of the key.

const base: EditorKeyState = {
  mode: "objects", tool: "select", selectionCount: 2, clipboardCount: 1, ridable: true, pending: false,
  samPoints: 0, kpDraftPoints: 0, classCount: 9, objectCanvas: true,
};

let run: ReturnType<typeof vi.fn>;
let onPan: ReturnType<typeof vi.fn>;
let onBlocked: ReturnType<typeof vi.fn>;
let readOnly = true;

function Harness() {
  useEditorKeys({ state: () => base, run, onPan, readOnly: () => readOnly, onBlocked });
  return <div />;
}

beforeEach(() => {
  window.localStorage.clear(); resetKeymap();
  run = vi.fn(); onPan = vi.fn(); onBlocked = vi.fn(); readOnly = true;
});

const ran = () => run.mock.calls.map((c) => (c[0] as KeyIntent).t);
const blocked = () => onBlocked.mock.calls.map((c) => (c[0] as KeyIntent).t);
const key = (init: KeyboardEventInit) => fireEvent.keyDown(document.body, init);

describe("keys on a view-only frame", () => {
  it("refuses save, and still stops the browser's own save dialog", () => {
    render(<Harness />);
    const notPrevented = key({ key: "s", code: "KeyS", metaKey: true });
    expect(notPrevented).toBe(false);
    expect(ran()).toEqual([]);
    expect(blocked()).toEqual(["save"]);
  });

  it("refuses paste, delete, nudge, relabel, undo and the drawing tools", () => {
    render(<Harness />);
    key({ key: "v", code: "KeyV", ctrlKey: true });
    key({ key: "Delete", code: "Delete" });
    key({ key: "Backspace", code: "Backspace" });
    key({ key: "ArrowLeft", code: "ArrowLeft" });
    key({ key: "3", code: "Digit3" });
    key({ key: "z", code: "KeyZ", metaKey: true });
    key({ key: "z", code: "KeyZ", metaKey: true, shiftKey: true });
    key({ key: "b", code: "KeyB" });
    key({ key: "a", code: "KeyA" });
    key({ key: "Enter", code: "Enter", shiftKey: true });
    expect(ran()).toEqual([]);
    expect(blocked()).toEqual(
      ["paste", "delete", "delete", "nudge", "relabel", "undo", "redo", "tool", "action", "acceptRest"]);
  });

  it("says so once for a held key, not once per repeat", () => {
    render(<Harness />);
    key({ key: "ArrowRight", code: "ArrowRight" });
    for (let i = 0; i < 10; i++) key({ key: "ArrowRight", code: "ArrowRight", repeat: true });
    expect(ran()).toEqual([]);
    expect(onBlocked).toHaveBeenCalledTimes(1);
  });

  it("still pans, zooms, selects, copies and changes frame", () => {
    render(<Harness />);
    key({ key: " ", code: "Space" });
    key({ key: "=", code: "Equal" });
    key({ key: "a", code: "KeyA", metaKey: true });
    key({ key: "c", code: "KeyC", metaKey: true });
    key({ key: "]", code: "BracketRight" });
    key({ key: "Escape", code: "Escape" });
    key({ key: "f", code: "KeyF" });
    expect(onPan).toHaveBeenCalledWith(true);
    expect(ran()).toEqual(["zoom", "selectBy", "copy", "frame", "selectBy", "action"]);
    expect(onBlocked).not.toHaveBeenCalled();
  });

  it("reads the lock at the moment of the key, so losing and regaining it needs no re-render", () => {
    render(<Harness />);
    readOnly = false;
    key({ key: "Delete", code: "Delete" });
    readOnly = true;
    key({ key: "Delete", code: "Delete" });
    readOnly = false;
    key({ key: "s", code: "KeyS", ctrlKey: true });
    expect(ran()).toEqual(["delete", "save"]);
    expect(blocked()).toEqual(["delete"]);
  });
});
