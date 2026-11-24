import { describe, expect, it, vi } from "vitest";

import { type Action, type EdObject, type EditGate, type EditorState, editorReducer, gateDispatch, isDirty } from "./useEditor";

// The door every change to the working copy goes through. View-only is enforced here, so these drive the
// real reducer through the real gate: what matters is not that a function returned false but that the
// state a person had is the state they still have.

function obj(id: string, over: Partial<EdObject> = {}): EdObject {
  return {
    id, class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, version: 3, ...over,
  };
}

function editor(objects: EdObject[]) {
  let state: EditorState = {
    objects, deleted: [], selectedId: null, selectedIds: [], tool: "select",
    viewport: { scale: 1, ox: 0, oy: 0 }, candidate: null, touched: [], past: [], future: [],
  };
  const gate: EditGate = { readOnly: false, onBlocked: vi.fn() };
  const dispatch = gateDispatch((a: Action) => { state = editorReducer(state, a); }, () => gate);
  return { dispatch, gate, get state() { return state; }, blocked: gate.onBlocked as ReturnType<typeof vi.fn> };
}

describe("an editable frame", () => {
  it("passes every action through", () => {
    const e = editor([obj("a")]);
    e.dispatch({ t: "update", id: "a", patch: { bbox: [5, 5, 20, 20] } });
    e.dispatch({ t: "add", obj: obj("tmp-1", { isNew: true }) });
    expect(e.state.objects.map((o) => o.id)).toEqual(["a", "tmp-1"]);
    expect(e.state.objects[0].bbox).toEqual([5, 5, 20, 20]);
    expect(e.blocked).not.toHaveBeenCalled();
  });

  it("with no gate at all behaves as the bare reducer", () => {
    let state: EditorState = editorReducer(undefined as unknown as EditorState,
      { t: "load", objects: [obj("a")], viewport: { scale: 1, ox: 0, oy: 0 } });
    const dispatch = gateDispatch((a) => { state = editorReducer(state, a); }, () => undefined);
    dispatch({ t: "delete", id: "a" });
    expect(state.objects).toEqual([]);
  });
});

describe("a view-only frame", () => {
  it("refuses draw, move, delete, paste, confirm and undo, and says so each time", () => {
    const e = editor([obj("a"), obj("b")]);
    e.dispatch({ t: "select", id: "a" });
    e.gate.readOnly = true;
    const before = e.state;

    const edits: Action[] = [
      { t: "add", obj: obj("tmp-1", { isNew: true }) },                      // draw
      { t: "addMany", objs: [obj("tmp-2", { isNew: true })] },               // paste
      { t: "update", id: "a", patch: { bbox: [1, 1, 2, 2] } },               // move / resize / relabel
      { t: "nudge", ids: ["a"], dx: 5, dy: 0 },                              // arrow keys
      { t: "delete", id: "a" },
      { t: "deleteMany", ids: ["a", "b"] },
      { t: "acceptAll" }, { t: "acceptRest" },
      { t: "undo" }, { t: "redo" }, { t: "jump", at: 1 },
      { t: "tool", tool: "box" },
      { t: "candidate", polys: [[0, 0, 4, 0, 4, 4]] },
    ];
    for (const a of edits) e.dispatch(a);

    expect(e.state).toBe(before);
    expect(e.blocked).toHaveBeenCalledTimes(edits.length);
    expect(e.blocked.mock.calls.map((c) => c[0].t)).toEqual(edits.map((a) => a.t));
  });

  it("still lets a viewer select, hide, change the view and pick the ruler", () => {
    const e = editor([obj("a"), obj("b")]);
    e.gate.readOnly = true;
    e.dispatch({ t: "select", id: "b" });
    e.dispatch({ t: "setVisible", ids: ["a"], visible: false });
    e.dispatch({ t: "viewport", viewport: { scale: 2, ox: 10, oy: 10 } });
    e.dispatch({ t: "tool", tool: "measure" });
    expect(e.state.selectedId).toBe("b");
    expect(e.state.objects[0].visible).toBe(false);
    expect(e.state.viewport.scale).toBe(2);
    expect(e.state.tool).toBe("measure");
    expect(e.blocked).not.toHaveBeenCalled();
    // None of that is an edit: there is still nothing to save.
    expect(isDirty(e.state)).toBe(false);
  });

  it("keeps unsaved work exactly as it was when the lock is lost mid-edit", () => {
    const e = editor([obj("a"), obj("b")]);
    // Editing: a moved box, a new box, a deleted box, and two steps of history.
    e.dispatch({ t: "update", id: "a", patch: { bbox: [5, 5, 20, 20] } });
    e.dispatch({ t: "add", obj: obj("tmp-1", { isNew: true, class_name: "bus" }) });
    e.dispatch({ t: "delete", id: "b" });
    const held = e.state;
    expect(isDirty(held)).toBe(true);

    // Taken over. Everything the person now presses is refused...
    e.gate.readOnly = true;
    e.dispatch({ t: "undo" });
    e.dispatch({ t: "update", id: "a", patch: { bbox: [0, 0, 1, 1] } });
    e.dispatch({ t: "delete", id: "tmp-1" });

    // ...and what they had is untouched: the same objects, the same pending delete, the same history.
    expect(e.state).toBe(held);
    expect(e.state.objects.find((o) => o.id === "a")).toMatchObject({ bbox: [5, 5, 20, 20], dirty: true });
    expect(e.state.objects.find((o) => o.id === "tmp-1")).toMatchObject({ isNew: true, class_name: "bus" });
    expect(e.state.deleted).toEqual(["b"]);
    expect(e.state.past).toHaveLength(3);
    expect(isDirty(e.state)).toBe(true);

    // The lock comes back, and editing carries on from the same place.
    e.gate.readOnly = false;
    e.dispatch({ t: "undo" });
    expect(e.state.deleted).toEqual([]);
    expect(e.state.objects.map((o) => o.id)).toContain("b");
  });

  it("lets a save that was already in flight report what the server wrote", () => {
    const e = editor([obj("a")]);
    e.dispatch({ t: "add", obj: obj("tmp-1", { isNew: true }) });
    e.gate.readOnly = true;
    // The request left before the lock was lost; its answer arrives after.
    e.dispatch({ t: "saved", remap: { "tmp-1": "srv-9" }, versions: { "tmp-1": 1 } });
    expect(e.state.objects.map((o) => o.id)).toEqual(["a", "srv-9"]);
    expect(isDirty(e.state)).toBe(false);
    e.dispatch({ t: "reviewed", id: "a", state: "accepted", version: 4 });
    expect(e.state.objects[0]).toMatchObject({ state: "accepted", version: 4 });
    expect(e.blocked).not.toHaveBeenCalled();
  });

  it("lets the frame be read again from the server", () => {
    const e = editor([obj("a")]);
    e.gate.readOnly = true;
    e.dispatch({ t: "reload", objects: [obj("a", { class_name: "truck", version: 9 }), obj("c")] });
    expect(e.state.objects.map((o) => [o.id, o.class_name])).toEqual([["a", "truck"], ["c", "sedan"]]);
  });
});
