import { describe, expect, it } from "vitest";

import { type Action, type EdObject, type EditorState, editorReducer } from "./useEditor";

// The editor reducer is the most intricate pure module in the app and had no tests at all, while being the
// one place a bug silently corrupts annotations rather than throwing.
//
// It gains multi-select and per-object hide/lock here. Selection was a single `selectedId`, so there was no
// marquee, no group move, and no bulk delete or class change from the canvas; every batch action had to go
// through the agent bar. `visible` existed on the object and was honoured by the canvas, but was hardcoded
// true at all nine construction sites and no UI ever toggled it, so the capability was built and unreachable.

function obj(id: string, over: Partial<EdObject> = {}): EdObject {
  return {
    id, class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, ...over,
  };
}

function state(objects: EdObject[], over: Partial<EditorState> = {}): EditorState {
  return {
    objects, deleted: [], selectedId: null, selectedIds: [], tool: "select",
    viewport: { scale: 1, ox: 0, oy: 0 }, candidate: null, touched: [], past: [], future: [], ...over,
  };
}

const run = (s: EditorState, ...actions: Action[]) => actions.reduce(editorReducer, s);

describe("single selection", () => {
  it("selecting one object replaces any multi-selection", () => {
    const s = run(state([obj("a"), obj("b")]),
      { t: "selectMany", ids: ["a", "b"] },
      { t: "select", id: "a" });
    expect(s.selectedIds).toEqual(["a"]);
    expect(s.selectedId).toBe("a");
  });

  it("clearing the selection empties both", () => {
    const s = run(state([obj("a")]), { t: "select", id: "a" }, { t: "select", id: null });
    expect(s.selectedId).toBeNull();
    expect(s.selectedIds).toEqual([]);
  });
});

describe("multi-selection", () => {
  it("a marquee selects every id it swept", () => {
    const s = run(state([obj("a"), obj("b"), obj("c")]), { t: "selectMany", ids: ["a", "c"] });
    expect(s.selectedIds).toEqual(["a", "c"]);
  });

  it("the primary selection stays set so single-object panels keep working", () => {
    // selectedIds is additive to selectedId rather than replacing it, so the properties panel and the
    // geometry handles need no change and operate on the primary selection.
    const s = run(state([obj("a"), obj("b")]), { t: "selectMany", ids: ["a", "b"] });
    expect(s.selectedId).toBe("a");
  });

  it("additive selection unions without duplicating", () => {
    const s = run(state([obj("a"), obj("b"), obj("c")]),
      { t: "selectMany", ids: ["a", "b"] },
      { t: "selectMany", ids: ["b", "c"], additive: true });
    expect([...s.selectedIds].sort()).toEqual(["a", "b", "c"]);
  });

  it("non-additive selection replaces the previous one", () => {
    const s = run(state([obj("a"), obj("b")]),
      { t: "selectMany", ids: ["a"] },
      { t: "selectMany", ids: ["b"] });
    expect(s.selectedIds).toEqual(["b"]);
  });

  it("toggling adds then removes an id", () => {
    let s = run(state([obj("a"), obj("b")]), { t: "selectMany", ids: ["a"] });
    s = run(s, { t: "toggleSelect", id: "b" });
    expect([...s.selectedIds].sort()).toEqual(["a", "b"]);
    s = run(s, { t: "toggleSelect", id: "a" });
    expect(s.selectedIds).toEqual(["b"]);
  });

  it("everything selected is marked touched, so confirm-frame accepts it", () => {
    // "confirm frame" only accepts objects the annotator actually engaged with; a multi-select that did not
    // record them would silently drop the whole batch from the confirmation.
    const s = run(state([obj("a"), obj("b")]), { t: "selectMany", ids: ["a", "b"] });
    expect([...s.touched].sort()).toEqual(["a", "b"]);
  });
});

describe("locking", () => {
  it("a locked object cannot be swept into a marquee selection", () => {
    // Excluded at selection rather than at each action, so a bulk delete cannot remove the very object the
    // lock was protecting.
    const s = run(state([obj("a"), obj("b", { locked: true })]),
      { t: "selectMany", ids: ["a", "b"] });
    expect(s.selectedIds).toEqual(["a"]);
  });

  it("locking an already-selected object deselects it", () => {
    const s = run(state([obj("a"), obj("b")]),
      { t: "selectMany", ids: ["a", "b"] },
      { t: "setLocked", ids: ["b"], locked: true });
    expect(s.selectedIds).toEqual(["a"]);
    expect(s.objects.find((o) => o.id === "b")?.locked).toBe(true);
  });

  it("unlocking makes an object selectable again", () => {
    const s = run(state([obj("a", { locked: true })]),
      { t: "setLocked", ids: ["a"], locked: false },
      { t: "selectMany", ids: ["a"] });
    expect(s.selectedIds).toEqual(["a"]);
  });
});

describe("visibility", () => {
  it("hiding and showing flips the flag the canvas already honours", () => {
    let s = run(state([obj("a"), obj("b")]), { t: "setVisible", ids: ["a"], visible: false });
    expect(s.objects.find((o) => o.id === "a")?.visible).toBe(false);
    expect(s.objects.find((o) => o.id === "b")?.visible).toBe(true);
    s = run(s, { t: "setVisible", ids: ["a"], visible: true });
    expect(s.objects.find((o) => o.id === "a")?.visible).toBe(true);
  });

  it("hiding does not mark the object dirty", () => {
    // Visibility is a view concern. Marking it dirty would queue a pointless save every time someone hid a
    // box to see what was behind it.
    const s = run(state([obj("a")]), { t: "setVisible", ids: ["a"], visible: false });
    expect(s.objects[0].dirty).toBeFalsy();
  });

  it("hiding does not consume an undo step", () => {
    const s = run(state([obj("a")]), { t: "setVisible", ids: ["a"], visible: false });
    expect(s.past).toEqual([]);
  });

  it("hiding many at once affects exactly those ids", () => {
    const s = run(state([obj("a"), obj("b"), obj("c")]),
      { t: "setVisible", ids: ["a", "c"], visible: false });
    expect(s.objects.map((o) => o.visible)).toEqual([false, true, false]);
  });
});
