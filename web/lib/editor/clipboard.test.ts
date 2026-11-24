import { beforeEach, describe, expect, it } from "vitest";

import type { EdObject } from "@/components/editor/useEditor";

import {
  CLIPBOARD_STORAGE_KEY, PASTE_STEP, copyObjects, materialize, parseClipboard, pasteObjects, pasteOffset,
  readClipboard, resetClipboardForTests, toClipItems,
} from "./clipboard";

// The clipboard was one object in a ref on a page that remounts on every frame change. These tests are
// the three things that were wrong with that, as sentences: it copied one of many, it lost three geometry
// kinds on paste, and it did not survive the navigation it was most wanted across.

function obj(id: string, over: Partial<EdObject> = {}): EdObject {
  return {
    id, class_id: 3, class_name: "truck", bbox: [10, 10, 50, 40], mask: [], attrs: {},
    conf: 0.42, state: "review", visible: true, ...over,
  };
}

function memoryStorage(seed: Record<string, string> = {}) {
  const data = new Map(Object.entries(seed));
  return {
    data,
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => { data.set(k, v); },
  };
}

let n = 0;
const newId = () => `tmp-test-${++n}`;

beforeEach(() => { resetClipboardForTests(); n = 0; });

describe("what goes on the clipboard", () => {
  it("carries every geometry kind the editor stores", () => {
    const full = obj("a", {
      rot: 15, mask: [[0, 0, 4, 0, 4, 4]], polyline: [[0, 0], [9, 9]], bbox_amodal: [0, 0, 80, 60],
      keypoints: { skeleton: "person_17", points: [[1, 1, 2]] },
      cuboid_3d: { center: [10, 0, 1], size: [2, 4, 1.5], yaw: 0.2, yaw_source: "lane" },
      attrs: { occupant_count: 2 },
    });
    const [item] = toClipItems([full]);
    expect(item).toEqual({
      class_id: 3, class_name: "truck", bbox: [10, 10, 50, 40], rot: 15, mask: [[0, 0, 4, 0, 4, 4]],
      polyline: [[0, 0], [9, 9]], bbox_amodal: [0, 0, 80, 60],
      keypoints: { skeleton: "person_17", points: [[1, 1, 2]] },
      cuboid_3d: { center: [10, 0, 1], size: [2, 4, 1.5], yaw: 0.2, yaw_source: "lane" },
      attrs: { occupant_count: 2 },
    });
  });

  it("carries nothing that says which object it was", () => {
    // A paste is a new object a person put there. It must not arrive on another frame claiming the source
    // frame's track, its server id, its version or the review somebody gave the original.
    const [item] = toClipItems([obj("server-id-1", { track_id: "track-9", version: 4, state: "accepted", locked: true, dirty: true })]);
    for (const key of ["id", "track_id", "version", "state", "conf", "locked", "dirty", "visible", "isNew", "quality_score"]) {
      expect(item, key).not.toHaveProperty(key);
    }
  });

  it("is a copy, so editing the source afterwards does not change what will be pasted", () => {
    const src = obj("a", { mask: [[0, 0, 4, 0, 4, 4]] });
    const [item] = toClipItems([src]);
    src.bbox[0] = 999;
    src.mask[0][0] = 999;
    expect(item.bbox[0]).toBe(10);
    expect(item.mask[0][0]).toBe(0);
  });

  it("leaves out the optional kinds an object does not have", () => {
    const [item] = toClipItems([obj("a", { bbox_amodal: null, cuboid_3d: null })]);
    expect(Object.keys(item).sort()).toEqual(["attrs", "bbox", "class_id", "class_name", "mask"]);
  });
});

describe("what a paste makes", () => {
  it("makes new human objects with fresh ids", () => {
    const out = materialize(toClipItems([obj("server-1"), obj("server-2")]), 0, newId);
    expect(out.map((o) => o.id)).toEqual(["tmp-test-1", "tmp-test-2"]);
    for (const o of out) {
      expect(o.isNew).toBe(true);
      expect(o.state).toBe("accepted");
      expect(o.conf).toBe(1);
      expect(o.visible).toBe(true);
      expect(o.track_id).toBeUndefined();
      expect(o.version).toBeUndefined();
    }
  });

  it("offsets every kind of geometry together", () => {
    const src = obj("a", {
      mask: [[0, 0, 4, 0, 4, 4]], polyline: [[0, 0], [9, 9]], bbox_amodal: [0, 0, 80, 60],
      keypoints: { skeleton: "person_17", points: [[1, 1, 2], [0, 0, 0]] },
    });
    const [o] = materialize(toClipItems([src]), 14, newId);
    expect(o.bbox).toEqual([24, 24, 64, 54]);
    expect(o.mask).toEqual([[14, 14, 18, 14, 18, 18]]);
    expect(o.polyline).toEqual([[14, 14], [23, 23]]);
    expect(o.bbox_amodal).toEqual([14, 14, 94, 74]);
    expect(o.keypoints?.points).toEqual([[15, 15, 2], [0, 0, 0]]);
  });

  it("keeps a cuboid, which the old paste dropped", () => {
    const cuboid = { center: [10, 0, 1], size: [2, 4, 1.5], yaw: 0.2 };
    const [o] = materialize(toClipItems([obj("a", { cuboid_3d: cuboid })]), 14, newId);
    expect(o.cuboid_3d).toEqual(cuboid);
  });

  it("pasting twice gives two independent objects", () => {
    const items = toClipItems([obj("a", { mask: [[0, 0, 4, 0, 4, 4]] })]);
    const [first] = materialize(items, 0, newId);
    const [second] = materialize(items, 0, newId);
    first.mask[0][0] = 999;
    expect(second.mask[0][0]).toBe(0);
    expect(items[0].mask[0][0]).toBe(0);
  });
});

describe("where a paste lands", () => {
  const clip = (over = {}) => ({ frameId: "f1", items: [], pastedOn: null, pastes: 0, ...over });

  it("is one step away on the frame it was copied from, or it would hide its own source", () => {
    expect(pasteOffset(clip(), "f1")).toBe(PASTE_STEP);
  });

  it("is in place on another frame, which is the point of carrying it there", () => {
    expect(pasteOffset(clip(), "f2")).toBe(0);
  });

  it("steps on with each further paste on the same frame", () => {
    expect(pasteOffset(clip({ pastedOn: "f1", pastes: 2 }), "f1")).toBe(PASTE_STEP * 3);
    expect(pasteOffset(clip({ pastedOn: "f2", pastes: 2 }), "f2")).toBe(PASTE_STEP * 2);
  });

  it("starts again on a frame it has not pasted on yet", () => {
    expect(pasteOffset(clip({ pastedOn: "f2", pastes: 5 }), "f3")).toBe(0);
  });
});

describe("the store", () => {
  it("copies every object it is given", () => {
    const store = memoryStorage();
    expect(copyObjects("f1", [obj("a"), obj("b"), obj("c")], store)).toBe(3);
    expect(readClipboard(store)?.items).toHaveLength(3);
    expect(readClipboard(store)?.frameId).toBe("f1");
  });

  it("copying nothing keeps what was there", () => {
    const store = memoryStorage();
    copyObjects("f1", [obj("a")], store);
    expect(copyObjects("f1", [], store)).toBe(0);
    expect(readClipboard(store)?.items).toHaveLength(1);
  });

  it("pastes all of them, in place on the next frame and stepping after that", () => {
    const store = memoryStorage();
    copyObjects("f1", [obj("a"), obj("b", { bbox: [0, 0, 5, 5] })], store);
    const first = pasteObjects("f2", newId, store);
    expect(first.map((o) => o.bbox)).toEqual([[10, 10, 50, 40], [0, 0, 5, 5]]);
    const second = pasteObjects("f2", newId, store);
    expect(second[0].bbox).toEqual([24, 24, 64, 54]);
    expect(new Set([...first, ...second].map((o) => o.id)).size).toBe(4);
  });

  it("offsets the first paste on the source frame", () => {
    const store = memoryStorage();
    copyObjects("f1", [obj("a")], store);
    expect(pasteObjects("f1", newId, store)[0].bbox).toEqual([24, 24, 64, 54]);
    expect(pasteObjects("f1", newId, store)[0].bbox).toEqual([38, 38, 78, 68]);
  });

  it("a new copy starts the stepping over", () => {
    const store = memoryStorage();
    copyObjects("f1", [obj("a")], store);
    pasteObjects("f2", newId, store);
    copyObjects("f1", [obj("a")], store);
    expect(pasteObjects("f2", newId, store)[0].bbox).toEqual([10, 10, 50, 40]);
  });

  it("pastes nothing when nothing was copied", () => {
    expect(pasteObjects("f1", newId, memoryStorage())).toEqual([]);
  });

  it("survives the page being loaded again, through sessionStorage", () => {
    // Module state survives the route change between frames. This is the other case: a reload of the tab,
    // where the module starts empty and the copy has to come back from storage.
    const store = memoryStorage();
    copyObjects("f1", [obj("a", { polyline: [[0, 0], [9, 9]] })], store);
    resetClipboardForTests();
    const back = readClipboard(store);
    expect(back?.frameId).toBe("f1");
    expect(back?.items[0].polyline).toEqual([[0, 0], [9, 9]]);
    expect(pasteObjects("f2", newId, store)).toHaveLength(1);
  });

  it("reads storage once, then trusts memory", () => {
    const store = memoryStorage();
    expect(readClipboard(store)).toBeNull();
    store.setItem(CLIPBOARD_STORAGE_KEY, JSON.stringify({ frameId: "f9", items: toClipItems([obj("a")]) }));
    expect(readClipboard(store)).toBeNull();
  });

  it("still copies when storage refuses the write", () => {
    // A few hundred SAM masks can exceed the quota. The copy must hold in memory regardless, because
    // frame-to-frame paste never needed the storage.
    const full = { getItem: () => null, setItem: () => { throw new Error("QuotaExceededError"); } };
    expect(copyObjects("f1", [obj("a")], full)).toBe(1);
    expect(pasteObjects("f2", newId, full)).toHaveLength(1);
  });

  it("is empty, not broken, when storage cannot be read or does not exist", () => {
    const broken = { getItem: () => { throw new Error("SecurityError"); }, setItem: () => {} };
    expect(readClipboard(broken)).toBeNull();
    resetClipboardForTests();
    expect(readClipboard(undefined)).toBeNull();
    expect(copyObjects("f1", [obj("a")], undefined)).toBe(1);
  });

  it("falls back to the browser's sessionStorage, and to nothing where there is none", () => {
    // Node has no sessionStorage: the default argument has to cope with that rather than throw.
    expect(readClipboard()).toBeNull();
    expect(copyObjects("f1", [obj("a")])).toBe(1);
    expect(pasteObjects("f2", newId)).toHaveLength(1);
  });
});

describe("what is believed from storage", () => {
  const good = { class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {} };
  const wrap = (items: unknown[], over = {}) => JSON.stringify({ frameId: "f1", items, ...over });

  it("accepts a clipboard it wrote", () => {
    const c = parseClipboard(wrap([{ ...good, rot: 10, attrs: { a: 1 }, bbox_amodal: [0, 0, 20, 20], polyline: [[0, 0], [1, 1]],
      keypoints: { skeleton: "person_17", points: [[1, 1, 2]] },
      cuboid_3d: { center: [1, 2, 3], size: [1, 1, 1], yaw: 0, yaw_source: "human" } }], { pastedOn: "f2", pastes: 3 }));
    expect(c?.items[0]).toEqual({ ...good, rot: 10, attrs: { a: 1 }, bbox_amodal: [0, 0, 20, 20], polyline: [[0, 0], [1, 1]],
      keypoints: { skeleton: "person_17", points: [[1, 1, 2]] },
      cuboid_3d: { center: [1, 2, 3], size: [1, 1, 1], yaw: 0, yaw_source: "human" } });
    expect(c?.pastedOn).toBe("f2");
    expect(c?.pastes).toBe(3);
  });

  it("rejects what is not a clipboard at all", () => {
    for (const raw of [null, undefined, "", "not json", "null", "42", JSON.stringify({ items: [good] }),
                       JSON.stringify({ frameId: "f1", items: "nope" }), wrap([])]) {
      expect(parseClipboard(raw), String(raw)).toBeNull();
    }
  });

  it("drops an item that is not an object's geometry, and keeps the ones that are", () => {
    // What comes out of here becomes annotation geometry. A three-number box must not.
    const c = parseClipboard(wrap([
      good, null, "x", { ...good, bbox: [0, 0, 10] }, { ...good, bbox: [0, 0, "10", 10] }, { ...good, bbox: [0, 0, NaN, 1] },
      { ...good, class_id: "1" }, { ...good, class_name: 5 }, { ...good, mask: [[0, "1"]] }, { ...good, mask: "m" },
    ]));
    expect(c?.items).toEqual([good]);
  });

  it("drops a malformed optional kind rather than the whole item", () => {
    const c = parseClipboard(wrap([{ ...good, attrs: [1], rot: "9", bbox_amodal: [1, 2], polyline: "p",
      keypoints: { skeleton: 1, points: [] }, cuboid_3d: { center: [1], size: "s", yaw: 0 } }]));
    expect(c?.items).toEqual([good]);
  });

  it("keeps a cuboid whose yaw source is missing", () => {
    const c = parseClipboard(wrap([{ ...good, cuboid_3d: { center: [1, 2, 3], size: [1, 1, 1], yaw: 0.5 } }]));
    expect(c?.items[0].cuboid_3d).toEqual({ center: [1, 2, 3], size: [1, 1, 1], yaw: 0.5 });
  });

  it("does not believe a paste count that is not one", () => {
    expect(parseClipboard(wrap([good], { pastes: -2, pastedOn: 7 }))).toMatchObject({ pastes: 0, pastedOn: null });
    expect(parseClipboard(wrap([good], { pastes: 2.7 }))?.pastes).toBe(2);
  });
});
