import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetKeymap, setBinding } from "@/lib/editor/keymap";
import type { KeyIntent } from "@/lib/editor/keys";

import { useEditorKeys, type EditorKeyState } from "./useEditorKeys";

// What a key means is tested in lib/editor/keys.test.ts without a DOM. This is the part that needs one:
// the listener is really on window, it really reads where the event landed, and it really sees state that
// changed after it was attached. The old handler got the last of those wrong by construction: it closed
// over a render's values and listed ten fewer dependencies than it read.

const base: EditorKeyState = {
  mode: "objects", tool: "select", selectionCount: 0, clipboardCount: 0, ridable: false, pending: false,
  samPoints: 0, kpDraftPoints: 0, classCount: 9, objectCanvas: true,
};

function Harness({ state, run, onPan }: {
  state: Partial<EditorKeyState>; run: (i: KeyIntent) => void; onPan: (b: boolean) => void;
}) {
  const overrides = useEditorKeys({ state: () => ({ ...base, ...state }), run, onPan });
  return (
    <div>
      <input aria-label="search" />
      <textarea aria-label="note" />
      <div aria-label="rich" contentEditable="true" suppressContentEditableWarning>text</div>
      <div role="menu"><button>menu item</button></div>
      <button>plain button</button>
      <output>{JSON.stringify(overrides)}</output>
    </div>
  );
}

let run: ReturnType<typeof vi.fn>;
let onPan: ReturnType<typeof vi.fn>;
beforeEach(() => { window.localStorage.clear(); resetKeymap(); run = vi.fn(); onPan = vi.fn(); });

const mount = (state: Partial<EditorKeyState> = {}) => render(<Harness state={state} run={run} onPan={onPan} />);
const intents = () => run.mock.calls.map((c) => c[0]);

describe("the editor keymap on window", () => {
  it("runs what a key means", () => {
    mount();
    fireEvent.keyDown(document.body, { key: "b", code: "KeyB" });
    expect(intents()).toEqual([{ t: "tool", tool: "box" }]);
  });

  it("nudges the selection with the arrows, and prevents the scroll", () => {
    mount({ selectionCount: 2 });
    const notPrevented = fireEvent.keyDown(document.body, { key: "ArrowLeft", code: "ArrowLeft" });
    fireEvent.keyDown(document.body, { key: "ArrowDown", code: "ArrowDown", shiftKey: true });
    fireEvent.keyDown(document.body, { key: "ArrowDown", code: "ArrowDown", repeat: true });
    expect(notPrevented).toBe(false);
    expect(intents()).toEqual([
      { t: "nudge", dx: -1, dy: 0, coalesce: false },
      { t: "nudge", dx: 0, dy: 10, coalesce: false },
      { t: "nudge", dx: 0, dy: 1, coalesce: true },
    ]);
  });

  it("does nothing while typing in an input, a textarea or a contenteditable", () => {
    mount({ selectionCount: 2 });
    for (const label of ["search", "note", "rich"]) {
      const el = screen.getByLabelText(label);
      fireEvent.keyDown(el, { key: "ArrowLeft", code: "ArrowLeft" });
      fireEvent.keyDown(el, { key: "Backspace", code: "Backspace" });
      fireEvent.keyDown(el, { key: "b", code: "KeyB" });
      fireEvent.keyDown(el, { key: " ", code: "Space" });
    }
    expect(run).not.toHaveBeenCalled();
    expect(onPan).not.toHaveBeenCalled();
  });

  it("leaves the arrows to a menu that has focus, and still nudges from a plain button", () => {
    mount({ selectionCount: 1 });
    fireEvent.keyDown(screen.getByRole("button", { name: "menu item" }), { key: "ArrowDown", code: "ArrowDown" });
    expect(run).not.toHaveBeenCalled();
    fireEvent.keyDown(screen.getByRole("button", { name: "plain button" }), { key: "ArrowDown", code: "ArrowDown" });
    expect(intents()).toEqual([{ t: "nudge", dx: 0, dy: 1, coalesce: false }]);
  });

  it("leaves the arrows alone when something nearer has already handled the event", () => {
    mount({ selectionCount: 1 });
    const claim = (e: Event) => e.preventDefault();
    document.addEventListener("keydown", claim);
    fireEvent.keyDown(document.body, { key: "ArrowDown", code: "ArrowDown" });
    document.removeEventListener("keydown", claim);
    expect(run).not.toHaveBeenCalled();
  });

  it("pans while space is held and stops when it is released", () => {
    mount();
    fireEvent.keyDown(document.body, { key: " ", code: "Space" });
    fireEvent.keyUp(document.body, { key: " ", code: "Space" });
    fireEvent.keyUp(document.body, { key: "b", code: "KeyB" });
    expect(onPan.mock.calls).toEqual([[true], [false]]);
    expect(run).not.toHaveBeenCalled();
  });

  it("ignores a key that means nothing, without preventing it", () => {
    mount();
    expect(fireEvent.keyDown(document.body, { key: "q", code: "KeyQ" })).toBe(true);
    expect(run).not.toHaveBeenCalled();
  });

  it("sees state that changed after the listener was attached", () => {
    const view = mount({ selectionCount: 0 });
    fireEvent.keyDown(document.body, { key: "Delete", code: "Delete" });
    expect(run).not.toHaveBeenCalled();
    view.rerender(<Harness state={{ selectionCount: 3 }} run={run} onPan={onPan} />);
    fireEvent.keyDown(document.body, { key: "Delete", code: "Delete" });
    expect(intents()).toEqual([{ t: "delete" }]);
  });

  it("resolves a rebound key at once, and returns the overrides for whatever prints keys", () => {
    mount();
    act(() => { setBinding("tool:box", "n"); });
    fireEvent.keyDown(document.body, { key: "n", code: "KeyN" });
    fireEvent.keyDown(document.body, { key: "b", code: "KeyB" });
    expect(intents()).toEqual([{ t: "tool", tool: "box" }]);
    expect(screen.getByRole("status")).toHaveTextContent('{"tool:box":"n"}');
  });

  it("reads the keymap stored on an earlier visit", () => {
    window.localStorage.setItem("lbx.editor.keymap", JSON.stringify({ "tool:box": "n" }));
    mount();
    fireEvent.keyDown(document.body, { key: "n", code: "KeyN" });
    expect(intents()).toEqual([{ t: "tool", tool: "box" }]);
  });

  it("stops listening when the editor unmounts", () => {
    const view = mount();
    view.unmount();
    fireEvent.keyDown(document.body, { key: "b", code: "KeyB" });
    expect(run).not.toHaveBeenCalled();
  });
});
