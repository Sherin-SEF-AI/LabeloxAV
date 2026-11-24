import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import type { Action, EdObject, Tool } from "@/components/editor/useEditor";

import type { KeyIntent } from "./keys";
import {
  actionAllowed, canvasMenuAllowed, effectiveTool, intentAllowed, isViewTool, menuEventAllowed, readOnlyNotice,
} from "./readOnly";

// What a view-only editor still does. The lists below are written out in full on purpose: the rule is an
// allowlist, so the interesting failure is a NEW action, intent or menu entry, and a test that enumerated
// "everything in the union" by reflection would pass it straight through.

const obj: EdObject = {
  id: "a", class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {}, conf: 1,
  state: "review", visible: true,
};
const vp = { scale: 1, ox: 0, oy: 0 };

// Every reducer action, and whether a viewer may dispatch it. Typed as a full record of the union's tags,
// so adding an action to the reducer without deciding this is a compile error here.
const ACTIONS: { [K in Action["t"]]: { action: Extract<Action, { t: K }>; allowed: boolean } } = {
  load: { action: { t: "load", objects: [obj], viewport: vp }, allowed: true },
  reload: { action: { t: "reload", objects: [obj] }, allowed: true },
  saved: { action: { t: "saved", remap: {} }, allowed: true },
  reviewed: { action: { t: "reviewed", id: "a", state: "accepted" }, allowed: true },
  viewport: { action: { t: "viewport", viewport: vp }, allowed: true },
  select: { action: { t: "select", id: "a" }, allowed: true },
  selectMany: { action: { t: "selectMany", ids: ["a"] }, allowed: true },
  toggleSelect: { action: { t: "toggleSelect", id: "a" }, allowed: true },
  selectBy: { action: { t: "selectBy", how: "all" }, allowed: true },
  setVisible: { action: { t: "setVisible", ids: ["a"], visible: false }, allowed: true },
  setLocked: { action: { t: "setLocked", ids: ["a"], locked: true }, allowed: true },
  tool: { action: { t: "tool", tool: "select" }, allowed: true },
  candidate: { action: { t: "candidate", polys: null }, allowed: true },
  add: { action: { t: "add", obj }, allowed: false },
  addMany: { action: { t: "addMany", objs: [obj] }, allowed: false },
  update: { action: { t: "update", id: "a", patch: { bbox: [1, 1, 2, 2] } }, allowed: false },
  acceptAll: { action: { t: "acceptAll" }, allowed: false },
  acceptRest: { action: { t: "acceptRest" }, allowed: false },
  delete: { action: { t: "delete", id: "a" }, allowed: false },
  deleteMany: { action: { t: "deleteMany", ids: ["a"] }, allowed: false },
  nudge: { action: { t: "nudge", ids: ["a"], dx: 1, dy: 0 }, allowed: false },
  undo: { action: { t: "undo" }, allowed: false },
  redo: { action: { t: "redo" }, allowed: false },
  jump: { action: { t: "jump", at: 1 }, allowed: false },
};

describe("reducer actions while view-only", () => {
  for (const [name, { action, allowed }] of Object.entries(ACTIONS)) {
    it(`${allowed ? "allows" : "refuses"} ${name}`, () => {
      expect(actionAllowed(action)).toBe(allowed);
    });
  }

  it("refuses draw, move, delete, paste and every other way of changing an object", () => {
    const refused = Object.entries(ACTIONS).filter(([, v]) => !v.allowed).map(([k]) => k).sort();
    expect(refused).toEqual([
      "acceptAll", "acceptRest", "add", "addMany", "delete", "deleteMany", "jump", "nudge", "redo", "undo", "update",
    ]);
  });

  it("lets the pointer and the ruler be picked, and no drawing tool", () => {
    const tools: Tool[] = ["select", "box", "amodal", "polygon", "polyline", "adverse", "cuboid", "keypoint",
      "measure", "sam-point", "sam-box", "magic-wand", "brush", "eraser", "superpixel"];
    const allowed = tools.filter((tool) => actionAllowed({ t: "tool", tool }));
    expect(allowed).toEqual(["select", "measure"]);
  });

  it("lets a pending SAM candidate be cleared and not proposed", () => {
    expect(actionAllowed({ t: "candidate", polys: null })).toBe(true);
    expect(actionAllowed({ t: "candidate", polys: [[0, 0, 1, 1, 2, 2]] })).toBe(false);
  });
});

describe("the tool the canvas is given", () => {
  it("is the chosen tool when the frame is editable", () => {
    expect(effectiveTool("box", false)).toBe("box");
    expect(effectiveTool("brush", false)).toBe("brush");
  });

  it("falls back to the pointer when view-only and the chosen tool would draw", () => {
    expect(effectiveTool("box", true)).toBe("select");
    expect(effectiveTool("sam-point", true)).toBe("select");
    expect(effectiveTool("measure", true)).toBe("measure");
    expect(effectiveTool("select", true)).toBe("select");
    expect(isViewTool("eraser")).toBe(false);
  });
});

describe("keys while view-only", () => {
  const ALLOWED: KeyIntent[] = [
    { t: "pan" }, { t: "selectBy", how: "all" }, { t: "selectBy", how: "none" }, { t: "console" },
    { t: "copy" }, { t: "mode", key: "review" }, { t: "zoom", factor: 1.2 }, { t: "nextRisk" },
    { t: "frame", dir: "next" }, { t: "frame", dir: "prev" }, { t: "cancelPending" },
    { t: "tool", tool: "select" }, { t: "tool", tool: "measure" }, { t: "action", id: "fit" },
  ];
  const REFUSED: KeyIntent[] = [
    { t: "save" }, { t: "saveAs" }, { t: "paste" }, { t: "delete" },
    { t: "nudge", dx: 1, dy: 0, coalesce: false }, { t: "relabel", index: 0 },
    { t: "undo" }, { t: "redo" }, { t: "acceptRest" }, { t: "acceptCandidate" }, { t: "finishKeypoints" },
    { t: "samUndo" }, { t: "tool", tool: "box" }, { t: "tool", tool: "polygon" }, { t: "tool", tool: "brush" },
    { t: "action", id: "confirmFrame" }, { t: "action", id: "reviewAccept" }, { t: "action", id: "reviewReject" },
    { t: "action", id: "toggleHelmets" }, { t: "action", id: "bumpOccupants" },
  ];

  it("keeps looking, selecting, copying and moving between frames", () => {
    for (const i of ALLOWED) expect(intentAllowed(i), JSON.stringify(i)).toBe(true);
  });

  it("refuses save, paste, delete, move, relabel, undo and every drawing tool", () => {
    for (const i of REFUSED) expect(intentAllowed(i), JSON.stringify(i)).toBe(false);
  });

  it("refuses a single-key action nobody has classified", () => {
    expect(intentAllowed({ t: "action", id: "somethingAddedLater" })).toBe(false);
  });

  it("covers every single-key action the keymap defines", async () => {
    const { ACTIONS: KEY_ACTIONS } = await import("./keymap");
    const decided = new Set([...ALLOWED, ...REFUSED].flatMap((i) => (i.t === "action" ? [i.id] : [])));
    expect(KEY_ACTIONS.map((a) => a.id).filter((id) => !decided.has(id))).toEqual([]);
  });
});

describe("menus while view-only", () => {
  // Read from the page, as the keymap's own coupling test does: the menu ids live in a handler table
  // there, and a new entry must be either a view or refused, by decision rather than by default.
  const page = readFileSync(resolve(__dirname, "../../app/frame/[id]/page.tsx"), "utf8");

  it("refuses every menu bar entry that edits", () => {
    for (const id of ["save", "saveAs", "confirmFrame", "acceptRest", "undo", "redo", "accept", "reject",
                      "delete", "segRoad", "proposeLanes"]) {
      expect(menuEventAllowed(id), id).toBe(false);
    }
  });

  it("keeps navigation, selection and zoom", () => {
    for (const id of ["prevFrame", "nextFrame", "selectAll", "selectNone", "fit", "zoomIn", "zoomOut",
                      "togglePanel", "hide", "show", "nextRisk", "inspect"]) {
      expect(menuEventAllowed(id), id).toBe(true);
    }
  });

  it("has an answer for every entry in the page's menu table, and the page asks", () => {
    const table = page.slice(page.indexOf("const on: Record<string, () => void> = {"));
    const ids = Array.from(table.slice(0, table.indexOf("};")).matchAll(/^\s{6}(\w+): \(\) =>/gm), (m) => m[1]);
    expect(ids.length).toBeGreaterThan(30);
    const editing = ids.filter((id) => !menuEventAllowed(id)).sort();
    expect(editing).toEqual(["accept", "acceptRest", "confirmFrame", "delete", "proposeLanes", "redo", "reject",
                             "save", "saveAs", "segRoad", "undo"]);
    expect(page).toContain("menuEventAllowed(k) ? run : () => { if (!viewOnly()) run(); }");
  });

  it("refuses the right-click entries that edit and keeps the ones that look", () => {
    for (const k of ["class", "accept", "reject", "acceptMany", "rejectMany", "propagate", "delete",
                     "deleteMany", "save"]) {
      expect(canvasMenuAllowed(k), k).toBe(false);
    }
    for (const k of ["hide", "lock", "copyId", "viewTrack", "selectAll", "fit", "showPanel"]) {
      expect(canvasMenuAllowed(k), k).toBe(true);
    }
  });

  it("has an answer for every case the page's canvas menu handles, and the page asks", () => {
    const start = page.indexOf("const onCanvasAction = useCallback(");
    const body = page.slice(start, page.indexOf("// reviewObject is deliberately not listed", start));
    const keys = Array.from(new Set(Array.from(body.matchAll(/case "(\w+)":/g), (m) => m[1])));
    expect(keys.length).toBeGreaterThan(15);
    expect(keys.filter((k) => !canvasMenuAllowed(k)).sort()).toEqual(
      ["accept", "acceptMany", "class", "delete", "deleteMany", "propagate", "reject", "rejectMany", "save"]);
    expect(body).toContain("if (!canvasMenuAllowed(key) && viewOnly()) return;");
  });
});

describe("what the page does with the lock", () => {
  // The three places every edit converges, and the one place the working copy is written back. If any of
  // these stops consulting the lock the unit tests above keep passing while the editor stops being
  // view-only, so the wiring itself is pinned.
  const page = readFileSync(resolve(__dirname, "../../app/frame/[id]/page.tsx"), "utf8");

  it("gates the reducer, the keys and the canvas on the same flag", () => {
    expect(page).toContain("useEditor({ readOnly, onBlocked:");
    expect(page).toContain("readOnly: () => readOnlyRef.current,");
    expect(page).toContain("tool={effectiveTool(st.tool, readOnly)}");
    expect(page).toContain("readOnly={readOnly} candidate={st.candidate}");
  });

  it("does not save, by hand or by autosave, while view-only", () => {
    const save = page.slice(page.indexOf("const save = useCallback(async () => {"));
    expect(save.slice(0, 700)).toContain("if (readOnlyRef.current) return;");
    expect(page).toContain("if (!autosave || !loadedRef.current || !dirty || saving || readOnly) return;");
  });

  it("asks before leaving a view-only frame that still holds unsaved edits", () => {
    expect(page).toContain("if (readOnlyRef.current && isDirty(st)");
  });
});

describe("readOnlyNotice", () => {
  it("names the holder when there is one", () => {
    expect(readOnlyNotice("Asha")).toBe("view only: Asha is editing this frame");
    expect(readOnlyNotice(null)).toMatch(/^view only/);
  });
});
