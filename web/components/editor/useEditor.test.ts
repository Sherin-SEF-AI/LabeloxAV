import { describe, expect, it, vi } from "vitest";

import { type Action, type EdObject, type EditorState, editorReducer, isDirty, selectionIds } from "./useEditor";

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

describe("touched: what 'Confirm frame' will actually accept", () => {
  // acceptAll confirms only objects in `touched`, which is right - it is what stops Confirm rubber-stamping
  // auto-labels nobody looked at. But two paths never added to it, so the guard quietly ate real work.

  it("a rule-based selection marks its objects touched", () => {
    // "select all low-confidence" then Confirm used to accept nothing at all: selectBy was the one
    // selection path that never touched anything, so acceptAll had an empty set to work from.
    const s = run(
      state([obj("a", { conf: 0.2 }), obj("b", { conf: 0.9 })]),
      { t: "selectBy", how: "lowConf", value: 0.5 },
    );
    expect(s.selectedIds).toEqual(["a"]);
    expect(s.touched).toContain("a");

    const confirmed = editorReducer(s, { t: "acceptAll" });
    expect(confirmed.objects.find((o) => o.id === "a")?.state).toBe("accepted");
    expect(confirmed.objects.find((o) => o.id === "b")?.state).toBe("review");
  });

  it("keeps an object touched across the autosave that renames it", () => {
    // A new object's id changes on save. `touched` held the old tmp- id and was not remapped, so an object
    // the annotator had drawn and edited stopped counting as reviewed the moment autosave landed.
    const drawn = obj("tmp-1", { isNew: true });
    const s = run(
      state([drawn], { touched: ["tmp-1"] }),
      { t: "saved", remap: { "tmp-1": "real-1" }, versions: { "tmp-1": 1 } },
    );
    expect(s.objects[0].id).toBe("real-1");
    expect(s.touched).toEqual(["real-1"]);
  });

  it("leaves an untouched auto-label alone", () => {
    // The property that makes the gate worth having in the first place.
    const s = editorReducer(state([obj("a"), obj("b")]), { t: "acceptAll" });
    expect(s.objects.every((o) => o.state === "review")).toBe(true);
  });
});

describe("acceptRest: finishing the frame the risk order left behind", () => {
  // The deliberate counterpart to acceptAll, and the two must not become the same action.
  //
  // acceptAll confirms what the annotator looked at, which is what stops Confirm rubber-stamping
  // auto-labels. That leaves no way to say "I have dealt with the three that were wrong and the other
  // fourteen are fine" - the mean frame carries 16.7 objects and the busiest 187 - so the sentence cost
  // one click per object and nobody said it.

  it("accepts exactly the objects nobody touched", () => {
    const s = editorReducer(
      state([obj("a"), obj("b"), obj("c")], { touched: ["a"] }),
      { t: "acceptRest" },
    );
    // b and c were not looked at individually and are now accepted; a was, and acceptAll owns it.
    expect(s.objects.find((o) => o.id === "b")?.state).toBe("accepted");
    expect(s.objects.find((o) => o.id === "c")?.state).toBe("accepted");
    expect(s.objects.find((o) => o.id === "a")?.state).toBe("review");
  });

  it("marks them dirty so they persist through the normal save path", () => {
    // Without this the state change lives only in the browser: nothing is written, the frame looks
    // finished, and reloading it brings every object back at `review`.
    const s = editorReducer(state([obj("a")]), { t: "acceptRest" });
    expect(s.objects[0].dirty).toBe(true);
  });

  it("does not touch an object the annotator drew", () => {
    // A new box is human-authored on creation. Sweeping it up here would restate that as a review
    // verdict on somebody else's work.
    const s = editorReducer(state([obj("tmp-1", { isNew: true }), obj("b")]), { t: "acceptRest" });
    expect(s.objects.find((o) => o.id === "tmp-1")?.state).toBe("review");
    expect(s.objects.find((o) => o.id === "tmp-1")?.dirty).toBeFalsy();
  });

  it("is a no-op when there is nothing left, rather than a history entry", () => {
    // Pressing it twice must not stack undo steps that undo nothing, which is how an undo stack stops
    // being usable.
    const first = editorReducer(state([obj("a")]), { t: "acceptRest" });
    const second = editorReducer(first, { t: "acceptRest" });
    expect(second).toBe(first);
  });

  it("leaves an already-accepted object alone", () => {
    const s = editorReducer(state([obj("a", { state: "accepted" }), obj("b")]), { t: "acceptRest" });
    expect(s.objects.find((o) => o.id === "a")?.dirty).toBeFalsy();
    expect(s.objects.find((o) => o.id === "b")?.state).toBe("accepted");
  });

  it("is undoable in one step", () => {
    const s = editorReducer(state([obj("a"), obj("b")]), { t: "acceptRest" });
    const back = editorReducer(s, { t: "undo" });
    expect(back.objects.every((o) => o.state === "review")).toBe(true);
  });
});

describe("tmpId: the idempotency key a new object is saved under", () => {
  // The save sends this id as idem_key and the server returns the existing object for a repeated key on the
  // same frame. A counter that restarted per page load made the first box drawn after reopening a frame
  // collide with one saved on an earlier visit, and the new box was written over the old one.
  it("does not repeat across page loads", async () => {
    const first = await import("./useEditor");
    const a = [first.tmpId(), first.tmpId()];
    vi.resetModules();
    const second = await import("./useEditor");
    const b = [second.tmpId(), second.tmpId()];
    expect(new Set([...a, ...b]).size).toBe(4);
    expect(a[0].replace(/\d+$/, "")).not.toBe(b[0].replace(/\d+$/, ""));
  });

  it("keeps the tmp- prefix that marks an unsaved object", async () => {
    const m = await import("./useEditor");
    expect(m.tmpId()).toMatch(/^tmp-[0-9a-f]{12}-\d+$/);
  });
});

describe("undo across an autosave", () => {
  // Autosave writes an edit and the server bumps the object's version. Undo then restores an older
  // snapshot, and that snapshot must carry the version the server now holds: with the stale one, the next
  // save was refused as a conflict with the editor's own earlier write, and the undo was never saved.
  it("restores the old geometry with the server's current version", () => {
    const s = run(
      state([obj("a", { version: 3, bbox: [0, 0, 10, 10] })]),
      { t: "update", id: "a", patch: { bbox: [0, 0, 20, 20] } },
      { t: "saved", remap: {}, versions: { a: 4 } },
      { t: "undo" },
    );
    const a = s.objects.find((o) => o.id === "a");
    expect(a?.bbox).toEqual([0, 0, 10, 10]);
    expect(a?.version).toBe(4);
  });

  it("gives redo the current version too", () => {
    const s = run(
      state([obj("a", { version: 3 })]),
      { t: "update", id: "a", patch: { bbox: [0, 0, 20, 20] } },
      { t: "undo" },
      { t: "saved", remap: {}, versions: { a: 5 } },
      { t: "redo" },
    );
    expect(s.objects.find((o) => o.id === "a")?.version).toBe(5);
  });
});

describe("selection across the autosave that renames a new object", () => {
  // A new object's tmp- id becomes its server id on save. The selection kept the tmp- id, so the object
  // just drawn or pasted silently stopped being selected and Delete did nothing to it.
  it("keeps the new object selected", () => {
    const s = run(
      state([obj("a")]),
      { t: "add", obj: obj("tmp-x-1", { isNew: true }) },
      { t: "saved", remap: { "tmp-x-1": "real-9" }, versions: { "tmp-x-1": 1 } },
    );
    expect(s.selectedId).toBe("real-9");
    expect(s.objects.some((o) => o.id === "real-9")).toBe(true);
  });

  it("remaps a multi-selection", () => {
    const s = run(
      state([obj("a"), obj("tmp-x-2", { isNew: true })], { selectedIds: ["a", "tmp-x-2"], selectedId: "a" }),
      { t: "saved", remap: { "tmp-x-2": "real-2" } },
    );
    expect(s.selectedIds).toEqual(["a", "real-2"]);
    expect(s.selectedId).toBe("a");
  });
});

describe("selectionIds: what a selection-wide action reaches", () => {
  // The keyboard read `selectedId` and the menus read `selectedIds`, so Delete and the context menu's
  // delete disagreed about what was selected. This is the one answer both now use.

  it("is the multi-selection", () => {
    const s = run(state([obj("a"), obj("b"), obj("c")]), { t: "selectMany", ids: ["a", "c"] });
    expect(selectionIds(s)).toEqual(["a", "c"]);
  });

  it("includes a primary selection the multi-selection has not caught up with", () => {
    expect(selectionIds(state([obj("a"), obj("b")], { selectedId: "b", selectedIds: ["a"] }))).toEqual(["a", "b"]);
  });

  it("drops ids whose object is gone", () => {
    expect(selectionIds(state([obj("a")], { selectedId: "gone", selectedIds: ["a", "gone"] }))).toEqual(["a"]);
  });

  it("is empty when nothing is selected", () => {
    expect(selectionIds(state([obj("a")]))).toEqual([]);
  });

  it("a newly drawn object is the whole selection, not an addition to the last sweep", () => {
    // `add` set the primary and left the multi-selection on whatever was swept before, so the bulk bar
    // said "2 selected" about two other boxes and a bulk delete would have taken them with it.
    const s = run(state([obj("a"), obj("b")]),
      { t: "selectMany", ids: ["a", "b"] },
      { t: "add", obj: obj("tmp-1", { isNew: true }) });
    expect(s.selectedIds).toEqual(["tmp-1"]);
    expect(selectionIds(s)).toEqual(["tmp-1"]);
  });

  it("a redo does not bring an object back half selected", () => {
    // Undo removed the object and left its id in the multi-selection. Redo restored the object, and it
    // was then selected for bulk actions with no primary and nothing on screen to say so.
    const s = run(state([obj("a")]),
      { t: "add", obj: obj("tmp-1", { isNew: true }) },
      { t: "undo" },
      { t: "redo" });
    expect(s.objects.some((o) => o.id === "tmp-1")).toBe(true);
    expect(selectionIds(s)).toEqual([]);
  });
});

describe("deleteMany: a bulk delete is one edit", () => {
  it("removes every id it is given", () => {
    const s = editorReducer(state([obj("a"), obj("b"), obj("c")]), { t: "deleteMany", ids: ["a", "c"] });
    expect(s.objects.map((o) => o.id)).toEqual(["b"]);
    expect(s.deleted).toEqual(["a", "c"]);
  });

  it("is undone in one step", () => {
    // Looping `delete` made one entry per object: undoing a six-box delete was six presses, and stopping
    // after three left a frame nobody had chosen.
    const s = editorReducer(state([obj("a"), obj("b"), obj("c")]), { t: "deleteMany", ids: ["a", "b", "c"] });
    expect(s.past).toHaveLength(1);
    expect(s.past[0].label).toBe("deleted 3 objects");
    const back = editorReducer(s, { t: "undo" });
    expect(back.objects.map((o) => o.id)).toEqual(["a", "b", "c"]);
  });

  it("names the object when there is only one", () => {
    const s = editorReducer(state([obj("a", { class_name: "bus" })]), { t: "deleteMany", ids: ["a"] });
    expect(s.past[0].label).toBe("deleted bus");
  });

  it("does not queue a server delete for an object that was never saved", () => {
    const s = editorReducer(state([obj("tmp-1", { isNew: true }), obj("b")]), { t: "deleteMany", ids: ["tmp-1", "b"] });
    expect(s.deleted).toEqual(["b"]);
  });

  it("leaves a locked object where it is", () => {
    // A lock is a promise that a bulk action cannot reach the object, whatever the selection holds.
    const s = editorReducer(state([obj("a"), obj("b", { locked: true })]), { t: "deleteMany", ids: ["a", "b"] });
    expect(s.objects.map((o) => o.id)).toEqual(["b"]);
  });

  it("clears what it deleted from the selection and keeps the rest", () => {
    const s = run(state([obj("a"), obj("b"), obj("c")]),
      { t: "selectMany", ids: ["a", "b", "c"] },
      { t: "deleteMany", ids: ["a", "b"] });
    expect(s.selectedIds).toEqual(["c"]);
    expect(s.selectedId).toBeNull();
  });

  it("keeps a primary selection it did not delete", () => {
    const s = run(state([obj("a"), obj("b")]), { t: "select", id: "b" }, { t: "deleteMany", ids: ["a"] });
    expect(s.selectedId).toBe("b");
  });

  it("is a no-op, not a history entry, when nothing can be deleted", () => {
    const s0 = state([obj("a", { locked: true })]);
    expect(editorReducer(s0, { t: "deleteMany", ids: ["a", "nope"] })).toBe(s0);
  });

  it("a single delete also leaves the multi-selection", () => {
    const s = run(state([obj("a"), obj("b")]), { t: "selectMany", ids: ["a", "b"] }, { t: "delete", id: "a" });
    expect(s.selectedIds).toEqual(["b"]);
  });
});

describe("addMany: a paste is one edit", () => {
  const pasted = [obj("tmp-1", { isNew: true }), obj("tmp-2", { isNew: true, class_name: "bus" })];

  it("adds every object and selects exactly those", () => {
    const s = run(state([obj("a")]), { t: "select", id: "a" }, { t: "addMany", objs: pasted });
    expect(s.objects.map((o) => o.id)).toEqual(["a", "tmp-1", "tmp-2"]);
    expect(s.selectedIds).toEqual(["tmp-1", "tmp-2"]);
    expect(s.selectedId).toBe("tmp-1");
  });

  it("is undone in one step, however many it carried", () => {
    const s = editorReducer(state([obj("a")]), { t: "addMany", objs: pasted, label: "pasted 2 objects" });
    expect(s.past).toHaveLength(1);
    expect(s.past[0].label).toBe("pasted 2 objects");
    expect(editorReducer(s, { t: "undo" }).objects.map((o) => o.id)).toEqual(["a"]);
  });

  it("labels itself when the caller does not", () => {
    expect(editorReducer(state([]), { t: "addMany", objs: pasted }).past[0].label).toBe("added 2 objects");
    expect(editorReducer(state([]), { t: "addMany", objs: [pasted[1]] }).past[0].label).toBe("added bus");
  });

  it("an empty paste is not an edit", () => {
    const s0 = state([obj("a")]);
    expect(editorReducer(s0, { t: "addMany", objs: [] })).toBe(s0);
  });

  it("makes the frame dirty, so autosave creates them", () => {
    expect(isDirty(editorReducer(state([]), { t: "addMany", objs: pasted }))).toBe(true);
  });
});

describe("nudge: arrow keys move the selection", () => {
  it("moves every id by the offset, in image pixels", () => {
    const s = editorReducer(state([obj("a"), obj("b", { bbox: [5, 5, 9, 9] }), obj("c")]),
      { t: "nudge", ids: ["a", "b"], dx: 1, dy: -10 });
    expect(s.objects.find((o) => o.id === "a")?.bbox).toEqual([1, -10, 11, 0]);
    expect(s.objects.find((o) => o.id === "b")?.bbox).toEqual([6, -5, 10, -1]);
    expect(s.objects.find((o) => o.id === "c")?.bbox).toEqual([0, 0, 10, 10]);
  });

  it("carries the mask, not only the box", () => {
    const s = editorReducer(state([obj("a", { mask: [[0, 0, 10, 0, 10, 10]] })]),
      { t: "nudge", ids: ["a"], dx: 2, dy: 3 });
    expect(s.objects[0].mask).toEqual([[2, 3, 12, 3, 12, 13]]);
  });

  it("marks a saved object dirty and leaves a new one new", () => {
    const s = editorReducer(state([obj("a"), obj("tmp-1", { isNew: true })]),
      { t: "nudge", ids: ["a", "tmp-1"], dx: 1, dy: 0 });
    expect(s.objects.find((o) => o.id === "a")?.dirty).toBe(true);
    expect(s.objects.find((o) => o.id === "tmp-1")?.dirty).toBe(false);
  });

  it("counts as having looked at the object", () => {
    const s = editorReducer(state([obj("a")]), { t: "nudge", ids: ["a"], dx: 1, dy: 0 });
    expect(s.touched).toEqual(["a"]);
  });

  it("is undoable", () => {
    const s = run(state([obj("a")]), { t: "nudge", ids: ["a"], dx: 10, dy: 0 }, { t: "undo" });
    expect(s.objects[0].bbox).toEqual([0, 0, 10, 10]);
  });

  it("a held key is one undo entry", () => {
    // Key-repeat fires thirty times a second. Thirty undo steps to take back one gesture is not an undo.
    let s = editorReducer(state([obj("a")]), { t: "nudge", ids: ["a"], dx: 1, dy: 0 });
    for (let i = 0; i < 29; i++) s = editorReducer(s, { t: "nudge", ids: ["a"], dx: 1, dy: 0, coalesce: true });
    expect(s.objects[0].bbox).toEqual([30, 0, 40, 10]);
    expect(s.past).toHaveLength(1);
    expect(editorReducer(s, { t: "undo" }).objects[0].bbox).toEqual([0, 0, 10, 10]);
  });

  it("separate presses are separate entries", () => {
    const s = run(state([obj("a")]),
      { t: "nudge", ids: ["a"], dx: 1, dy: 0 },
      { t: "nudge", ids: ["a"], dx: 1, dy: 0 });
    expect(s.past).toHaveLength(2);
    expect(s.past[1].label).toBe("nudged sedan");
  });

  it("a repeat does not fold into a nudge of different objects", () => {
    const s = run(state([obj("a"), obj("b")]),
      { t: "nudge", ids: ["a"], dx: 1, dy: 0 },
      { t: "nudge", ids: ["a", "b"], dx: 1, dy: 0, coalesce: true });
    expect(s.past).toHaveLength(2);
    expect(s.past[1].label).toBe("nudged 2 objects");
  });

  it("a repeat does not fold into an edit that was not a nudge", () => {
    const s = run(state([obj("a")]),
      { t: "update", id: "a", patch: { bbox: [0, 0, 20, 20] } },
      { t: "nudge", ids: ["a"], dx: 1, dy: 0, coalesce: true });
    expect(s.past).toHaveLength(2);
  });

  it("a repeat after an undo starts a new entry rather than rewriting history under the redo stack", () => {
    const s = run(state([obj("a")]),
      { t: "nudge", ids: ["a"], dx: 1, dy: 0 },
      { t: "update", id: "a", patch: { bbox: [5, 5, 15, 15] } },
      { t: "undo" },
      { t: "nudge", ids: ["a"], dx: 1, dy: 0, coalesce: true });
    expect(s.past).toHaveLength(2);
    expect(s.future).toEqual([]);
  });

  it("still folds after the autosave that renames the object mid-hold", () => {
    const s = run(state([obj("tmp-1", { isNew: true })]),
      { t: "nudge", ids: ["tmp-1"], dx: 1, dy: 0 },
      { t: "saved", remap: { "tmp-1": "real-1" } },
      { t: "nudge", ids: ["real-1"], dx: 1, dy: 0, coalesce: true });
    expect(s.past).toHaveLength(1);
    expect(s.objects[0].bbox).toEqual([2, 0, 12, 10]);
  });

  it("does not move a locked object", () => {
    const s0 = state([obj("a", { locked: true })]);
    expect(editorReducer(s0, { t: "nudge", ids: ["a"], dx: 1, dy: 0 })).toBe(s0);
  });

  it("skips a cuboid object rather than moving half of it", () => {
    // The 3D box is in the ego frame, in metres. Sliding the 2D box alone would leave the two disagreeing.
    const cuboid = { center: [10, 0, 0.8], size: [1.8, 4.2, 1.5], yaw: 0 };
    const s = editorReducer(state([obj("a", { cuboid_3d: cuboid }), obj("b")]),
      { t: "nudge", ids: ["a", "b"], dx: 4, dy: 0 });
    expect(s.objects.find((o) => o.id === "a")?.bbox).toEqual([0, 0, 10, 10]);
    expect(s.objects.find((o) => o.id === "a")?.cuboid_3d).toBe(cuboid);
    expect(s.objects.find((o) => o.id === "b")?.bbox).toEqual([4, 0, 14, 10]);
    expect(s.past[0].label).toBe("nudged sedan");
  });

  it("a zero offset, or nothing movable, is not an edit", () => {
    const s0 = state([obj("a")]);
    expect(editorReducer(s0, { t: "nudge", ids: ["a"], dx: 0, dy: 0 })).toBe(s0);
    expect(editorReducer(s0, { t: "nudge", ids: ["nope"], dx: 1, dy: 0 })).toBe(s0);
  });
});

describe("reload: the server's copy, after something changed it behind the editor", () => {
  const server = [obj("a", { version: 7, bbox: [1, 1, 9, 9] }), obj("n", { class_name: "bus" })];

  it("replaces the objects and keeps the session", () => {
    const s0 = state([obj("a"), obj("gone")], {
      selectedId: "a", selectedIds: ["a", "gone"], touched: ["a", "gone"], tool: "box",
      viewport: { scale: 2, ox: 5, oy: 6 },
    });
    const s = editorReducer(s0, { t: "reload", objects: server });
    expect(s.objects.map((o) => o.id)).toEqual(["a", "n"]);
    expect(s.objects[0].version).toBe(7);
    // Opening a frame resets these; re-reading one must not.
    expect(s.tool).toBe("box");
    expect(s.viewport).toEqual({ scale: 2, ox: 5, oy: 6 });
    expect(s.selectedId).toBe("a");
    expect(s.selectedIds).toEqual(["a"]);
    expect(s.touched).toEqual(["a"]);
  });

  it("keeps hide, lock and attributes, which the object list does not return", () => {
    const s0 = state([obj("a", { visible: false, locked: true, attrs: { occupant_count: 2 } })]);
    const a = editorReducer(s0, { t: "reload", objects: server }).objects[0];
    expect(a.visible).toBe(false);
    expect(a.locked).toBe(true);
    expect(a.attrs).toEqual({ occupant_count: 2 });
  });

  it("drops a selection whose object the server no longer has", () => {
    const s = editorReducer(state([obj("gone")], { selectedId: "gone", selectedIds: ["gone"] }),
      { t: "reload", objects: server });
    expect(s.selectedId).toBeNull();
    expect(s.selectedIds).toEqual([]);
  });

  it("drops history, whose snapshots hold versions the server has moved past", () => {
    const s = run(state([obj("a")]),
      { t: "update", id: "a", patch: { bbox: [0, 0, 20, 20] } },
      { t: "saved", remap: {} },
      { t: "reload", objects: server });
    expect(s.past).toEqual([]);
    expect(s.future).toEqual([]);
    expect(isDirty(s)).toBe(false);
  });
});
