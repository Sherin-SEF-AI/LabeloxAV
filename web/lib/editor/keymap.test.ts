import { readFileSync } from "node:fs";
import { join } from "node:path";

import { beforeEach, describe, expect, it, vi } from "vitest";

import { GLOBAL, TOOLS } from "@/components/shell/ShortcutOverlay";

import {
  ACTIONS, KEYMAP_STORAGE_KEY, actionForHotkey, bindingKey, bindings, effectiveModes, getKeymap,
  getKeymapServerSnapshot, hydrateKeymap, isBindableKey, keymapCollisions, loadOverrides, rebind, resetKeymap,
  sanitizeOverrides, saveOverrides, setBinding, subscribeKeymap,
} from "./keymap";
import { MODES, groupsForMode, hotkeyCollisions, modeByKey, toolForHotkey, toolsForMode } from "./registry";

// Keys can be rebound now, which adds two ways to end up with a key that does nothing: a rebinding that
// puts two things on one key in a mode, and a stored keymap that outlived the registry it was made for.
// Both are refused here, by the registry's own collision check rather than a second opinion.

function memoryStorage(seed: Record<string, string> = {}) {
  const data = new Map(Object.entries(seed));
  return {
    data,
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => { data.set(k, v); },
    removeItem: (k: string) => { data.delete(k); },
  };
}

beforeEach(() => resetKeymap(memoryStorage()));

describe("the shipped keymap", () => {
  it("has no two bindings on one key inside a mode, actions included", () => {
    // The registry test asserts this for tools. This is the same check with the single-key actions added,
    // which is what would have caught a tool being given `f` or `a`.
    expect(keymapCollisions()).toEqual([]);
  });

  it("lists every tool once, however many modes offer it", () => {
    const ids = bindings().map((b) => b.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids.filter((id) => id === "tool:select")).toHaveLength(1);
    const select = bindings().find((b) => b.id === "tool:select");
    expect(select?.modes).toHaveLength(MODES.length);
  });

  it("lists every tool the registry has and every action", () => {
    const tools = new Set(MODES.flatMap((m) => toolsForMode(m.key)));
    const listed = bindings();
    expect(listed.filter((b) => b.kind === "tool").map((b) => b.id).sort())
      .toEqual([...tools].map((t) => `tool:${t}`).sort());
    expect(listed.filter((b) => b.kind === "action")).toHaveLength(ACTIONS.length);
  });

  it("says which modes a binding is active in", () => {
    const by = Object.fromEntries(bindings().map((b) => [b.id, b]));
    expect(by["tool:amodal"].modes).toEqual(["Objects"]);
    expect(by["tool:measure"].modes).toEqual(["Objects", "Pose and behavior"]);
    expect(by["action:reviewAccept"].modes).toEqual(["Review"]);
    expect(by["action:confirmFrame"].modes).not.toContain("Review");
    expect(by["action:fit"].modes).toHaveLength(MODES.length);
  });

  it("with no overrides, every binding answers to its default", () => {
    for (const b of bindings()) expect(b.key, b.id).toBe(b.default);
    expect(bindingKey({}, "tool:box")).toBe("b");
    expect(bindingKey({}, "tool:no-such-tool")).toBeNull();
  });

  it("with no overrides, the effective registry is the registry", () => {
    expect(effectiveModes()).toEqual(MODES);
  });
});

describe("actions, resolved per mode", () => {
  it("gives `a` to confirm-frame everywhere but Review, and to accept there", () => {
    expect(actionForHotkey("objects", "a")).toBe("confirmFrame");
    expect(actionForHotkey("pose", "A")).toBe("confirmFrame");
    expect(actionForHotkey("review", "a")).toBe("reviewAccept");
  });

  it("gives `x` to reject in Review and to nothing anywhere else", () => {
    expect(actionForHotkey("review", "x")).toBe("reviewReject");
    expect(actionForHotkey("objects", "x")).toBeNull();
  });

  it("resolves the mode-independent actions in every mode", () => {
    for (const m of MODES) {
      expect(actionForHotkey(m.key, "f"), m.key).toBe("fit");
      expect(actionForHotkey(m.key, "h"), m.key).toBe("toggleHelmets");
      expect(actionForHotkey(m.key, "o"), m.key).toBe("bumpOccupants");
    }
  });

  it("returns null for a key no action claims", () => {
    expect(actionForHotkey("objects", "z")).toBeNull();
  });
});

describe("override resolution", () => {
  it("a rebound tool answers to its new key and not its old one", () => {
    const r = rebind({}, "tool:box", "n");
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    const modes = effectiveModes(r.overrides);
    expect(toolForHotkey("objects", "n", modes)).toBe("box");
    expect(toolForHotkey("objects", "b", modes)).toBeNull();
    expect(bindingKey(r.overrides, "tool:box")).toBe("n");
  });

  it("a rebound tool moves in every mode that offers it", () => {
    const r = rebind({}, "tool:measure", "j");
    if (!r.ok) throw new Error(r.reason);
    const modes = effectiveModes(r.overrides);
    expect(toolForHotkey("objects", "j", modes)).toBe("measure");
    expect(toolForHotkey("pose", "j", modes)).toBe("measure");
  });

  it("the tool strip is handed the rebound key to print", () => {
    const r = rebind({}, "tool:box", "n");
    if (!r.ok) throw new Error(r.reason);
    const draw = groupsForMode("objects", effectiveModes(r.overrides)).find((g) => g.key === "draw");
    expect(draw?.tools.find((t) => t.key === "box")?.hotkey).toBe("N");
  });

  it("a rebound action answers to its new key and not its old one", () => {
    const r = rebind({}, "action:fit", "z");
    if (!r.ok) throw new Error(r.reason);
    expect(actionForHotkey("objects", "z", r.overrides)).toBe("fit");
    expect(actionForHotkey("objects", "f", r.overrides)).toBeNull();
  });

  it("is case insensitive, like the keys it replaces", () => {
    const r = rebind({}, "tool:box", "N");
    expect(r.ok && r.overrides).toEqual({ "tool:box": "n" });
  });

  it("frees the old key for something else", () => {
    const first = rebind({}, "tool:box", "n");
    if (!first.ok) throw new Error(first.reason);
    const second = rebind(first.overrides, "tool:polygon", "b");
    expect(second.ok && second.overrides).toEqual({ "tool:box": "n", "tool:polygon": "b" });
  });

  it("rebinding back to the default leaves no override behind", () => {
    const moved = rebind({}, "tool:box", "n");
    if (!moved.ok) throw new Error(moved.reason);
    const back = rebind(moved.overrides, "tool:box", "b");
    expect(back.ok && back.overrides).toEqual({});
  });

  it("does not mutate the overrides it was given", () => {
    const before = Object.freeze({ "tool:box": "n" });
    rebind(before, "tool:polygon", "j");
    expect(before).toEqual({ "tool:box": "n" });
  });

  it("the same letter can serve two modes that never show both tools", () => {
    // `k` is whole-extent in Objects and keypoint in Pose out of the box. Rebinding must allow the same
    // arrangement, or resolving per mode would be a property only the defaults could have.
    const r = rebind({}, "tool:keypoint", "u");
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(toolForHotkey("pose", "u", effectiveModes(r.overrides))).toBe("keypoint");
    expect(toolForHotkey("objects", "u", effectiveModes(r.overrides))).toBe("superpixel");
  });
});

describe("collision rejection", () => {
  it("refuses a key another tool has in the same mode, and says which", () => {
    const r = rebind({}, "tool:box", "g");
    expect(r).toEqual({ ok: false, reason: 'G is already "Polygon" in Objects' });
  });

  it("refuses a tool key an action has", () => {
    const r = rebind({}, "tool:box", "f");
    expect(r.ok).toBe(false);
    expect(!r.ok && r.reason).toMatch(/^F is already "fit the frame to the window" in /);
  });

  it("refuses an action key a tool has in any mode the action is active in", () => {
    const r = rebind({}, "action:fit", "k");
    expect(r.ok).toBe(false);
    expect(!r.ok && r.reason).toMatch(/^K is already /);
  });

  it("refuses two actions on one key in the same mode", () => {
    expect(rebind({}, "action:reviewReject", "a")).toEqual(
      { ok: false, reason: 'A is already "accept selected (Review mode)" in Review' });
  });

  it("checks every mode the binding is in, not only the first", () => {
    // `measure` is in Objects and Pose. `k` is free of collisions with it in neither.
    const r = rebind({}, "tool:measure", "k");
    expect(r.ok).toBe(false);
  });

  it("allows a key that is taken only in a mode the binding is not in", () => {
    // Review's own reject key, moved onto a letter only the Objects strip uses.
    expect(rebind({}, "action:reviewReject", "b").ok).toBe(true);
  });

  it("a refusal changes nothing", () => {
    const before = { "tool:box": "n" };
    const r = rebind(before, "tool:polygon", "n");
    expect(r.ok).toBe(false);
    expect(before).toEqual({ "tool:box": "n" });
  });

  it("is the registry's collision check, run over the rebound registry", () => {
    // Bypass `rebind` to build a keymap it would never produce, and confirm both checks see the clash.
    const clashing = { "tool:box": "g" };
    expect(hotkeyCollisions(effectiveModes(clashing))).toEqual([{ mode: "objects", hotkey: "g", tools: ["box", "polygon"] }]);
    expect(keymapCollisions(clashing)).toEqual(hotkeyCollisions(effectiveModes(clashing)));
  });

  it("refuses the keys the handler gives a fixed meaning", () => {
    for (const key of ["1", "9", "[", "]", "=", "+", "-", "\\", "?", " ", "Enter", "Tab", "Escape", "ArrowLeft", "F5", "é"]) {
      const r = rebind({}, "tool:box", key);
      expect(r.ok, key).toBe(false);
      expect(!r.ok && r.reason, key).toMatch(/is reserved/);
    }
    expect(rebind({}, "tool:box", " ")).toMatchObject({ reason: expect.stringMatching(/^Space is reserved/) });
  });

  it("accepts a letter, zero and the unshifted punctuation", () => {
    for (const key of ["n", "N", "0", ";", "'", ",", ".", "/", "`"]) expect(isBindableKey(key), key).toBe(true);
  });

  it("refuses a binding that does not exist", () => {
    expect(rebind({}, "tool:lasso", "n")).toEqual({ ok: false, reason: "that shortcut cannot be rebound" });
  });
});

describe("a stored keymap is not trusted", () => {
  it("keeps the overrides that are still legal", () => {
    expect(sanitizeOverrides({ "tool:box": "n", "action:fit": "z" })).toEqual({ "tool:box": "n", "action:fit": "z" });
  });

  it("drops a binding the registry no longer has", () => {
    expect(sanitizeOverrides({ "tool:lasso": "n", "tool:box": "j" })).toEqual({ "tool:box": "j" });
  });

  it("drops an override that would collide, rather than load a keymap with two tools on one key", () => {
    // Stored against an older registry in which `g` was free.
    expect(sanitizeOverrides({ "tool:box": "g" })).toEqual({});
    // And one that collides only with an earlier stored entry.
    expect(sanitizeOverrides({ "tool:box": "n", "tool:polygon": "n" })).toEqual({ "tool:box": "n" });
  });

  it("drops reserved keys and non-strings", () => {
    expect(sanitizeOverrides({ "tool:box": "1", "tool:polygon": 7, "tool:polyline": null })).toEqual({});
  });

  it("normalises case and drops a no-op override", () => {
    expect(sanitizeOverrides({ "tool:box": "N", "tool:polygon": "G" })).toEqual({ "tool:box": "n" });
  });

  it("is empty for anything that is not an object", () => {
    for (const raw of [null, undefined, "x", 3, ["tool:box"], true]) expect(sanitizeOverrides(raw)).toEqual({});
  });
});

describe("persistence", () => {
  it("round-trips through storage", () => {
    const store = memoryStorage();
    saveOverrides({ "tool:box": "n" }, store);
    expect(JSON.parse(store.data.get(KEYMAP_STORAGE_KEY) ?? "null")).toEqual({ "tool:box": "n" });
    expect(loadOverrides(store)).toEqual({ "tool:box": "n" });
  });

  it("removes the entry when nothing is overridden, rather than storing an empty object", () => {
    const store = memoryStorage({ [KEYMAP_STORAGE_KEY]: '{"tool:box":"n"}' });
    saveOverrides({}, store);
    expect(store.data.has(KEYMAP_STORAGE_KEY)).toBe(false);
  });

  it("loads the defaults when storage is empty, unreadable or not JSON", () => {
    expect(loadOverrides(memoryStorage())).toEqual({});
    expect(loadOverrides(memoryStorage({ [KEYMAP_STORAGE_KEY]: "{not json" }))).toEqual({});
    expect(loadOverrides({ getItem: () => { throw new Error("SecurityError"); }, setItem: () => {}, removeItem: () => {} })).toEqual({});
    expect(loadOverrides(undefined)).toEqual({});
  });

  it("sanitises what it loads", () => {
    expect(loadOverrides(memoryStorage({ [KEYMAP_STORAGE_KEY]: '{"tool:box":"g","tool:polygon":"j"}' })))
      .toEqual({ "tool:polygon": "j" });
  });

  it("a write that fails does not throw", () => {
    const full = { getItem: () => null, setItem: () => { throw new Error("QuotaExceededError"); }, removeItem: () => { throw new Error("x"); } };
    expect(() => saveOverrides({ "tool:box": "n" }, full)).not.toThrow();
    expect(() => saveOverrides({}, full)).not.toThrow();
    expect(() => saveOverrides({ "tool:box": "n" }, undefined)).not.toThrow();
  });
});

describe("the live keymap", () => {
  it("starts at the defaults, with a stable snapshot for the server render", () => {
    expect(getKeymap()).toEqual({});
    expect(getKeymapServerSnapshot()).toBe(getKeymapServerSnapshot());
    expect(getKeymapServerSnapshot()).toEqual({});
  });

  it("a rebinding is published and persisted", () => {
    const store = memoryStorage();
    const seen = vi.fn();
    const off = subscribeKeymap(seen);
    expect(setBinding("tool:box", "n", store).ok).toBe(true);
    expect(getKeymap()).toEqual({ "tool:box": "n" });
    expect(seen).toHaveBeenCalledTimes(1);
    expect(loadOverrides(store)).toEqual({ "tool:box": "n" });
    off();
    setBinding("tool:box", "j", store);
    expect(seen).toHaveBeenCalledTimes(1);
  });

  it("a refused rebinding publishes nothing and stores nothing", () => {
    const store = memoryStorage();
    const seen = vi.fn();
    const off = subscribeKeymap(seen);
    const r = setBinding("tool:box", "g", store);
    off();
    expect(r.ok).toBe(false);
    expect(getKeymap()).toEqual({});
    expect(seen).not.toHaveBeenCalled();
    expect(store.data.size).toBe(0);
  });

  it("reset returns to the defaults and clears storage", () => {
    const store = memoryStorage();
    setBinding("tool:box", "n", store);
    setBinding("action:fit", "z", store);
    resetKeymap(store);
    expect(getKeymap()).toEqual({});
    expect(store.data.has(KEYMAP_STORAGE_KEY)).toBe(false);
    expect(toolForHotkey("objects", "b", effectiveModes(getKeymap()))).toBe("box");
  });

  it("hydrates from storage, and says nothing when there is nothing new", () => {
    const store = memoryStorage({ [KEYMAP_STORAGE_KEY]: '{"tool:box":"n"}' });
    const seen = vi.fn();
    const off = subscribeKeymap(seen);
    hydrateKeymap(store);
    expect(getKeymap()).toEqual({ "tool:box": "n" });
    hydrateKeymap(store);
    off();
    expect(seen).toHaveBeenCalledTimes(1);
  });

  it("does not drop this page's rebindings because storage reads empty", () => {
    // Private mode: the write was refused, so storage is empty on the next mount. The editor remounts on
    // every frame change, and a rebinding that lasted until the next `]` would be worse than none.
    const refusing = { getItem: () => null, setItem: () => { throw new Error("QuotaExceededError"); }, removeItem: () => {} };
    expect(setBinding("tool:box", "n", refusing).ok).toBe(true);
    hydrateKeymap(refusing);
    expect(getKeymap()).toEqual({ "tool:box": "n" });
  });

  it("works against the browser's own storage, and without one", () => {
    // Node has no localStorage. The default arguments must cope rather than throw.
    expect(() => hydrateKeymap()).not.toThrow();
    expect(setBinding("tool:box", "n").ok).toBe(true);
    expect(() => resetKeymap()).not.toThrow();
    expect(() => saveOverrides({ "tool:box": "n" })).not.toThrow();
    expect(loadOverrides()).toEqual(expect.any(Object));
  });
});

describe("registry lookups over a rebound registry", () => {
  it("answer for the list they are given, and for MODES when given none", () => {
    const modes = effectiveModes({ "tool:box": "n" });
    expect(modeByKey("objects", modes)?.groups).not.toEqual(modeByKey("objects")?.groups);
    expect(toolsForMode("objects", modes)).toEqual(toolsForMode("objects"));
    expect(groupsForMode("no-such-mode", modes)).toEqual(groupsForMode("objects", modes));
    expect(toolForHotkey("objects", "b")).toBe("box");
  });
});

// The shortcut overlay's own coupling test (components/shell/ShortcutOverlay.test.tsx) checks the printed
// list against the letters the frame page binds, and finds those letters by scraping the page for the
// `k === "x"` comparisons the old handler was written as. The handler is data now, so the page lists its
// default letters in that form in a comment, and that comment is only honest if something holds it to the
// data. This does: with it, the overlay's test still means "the printed list matches the keymap".
describe("the page's letter list is the keymap's, not a second copy", () => {
  const PAGE = readFileSync(join(__dirname, "../../app/frame/[id]/page.tsx"), "utf8");
  const pageLetters = [...new Set([...PAGE.matchAll(/k === "([a-z])"/g)].map((m) => m[1]))].sort();
  const actionLetters = [...new Set(ACTIONS.map((a) => a.hotkey.toLowerCase()))].sort();

  it("lists exactly the letters the actions ship with", () => {
    expect(pageLetters).toEqual(actionLetters);
    expect(actionLetters.length).toBeGreaterThan(3);
  });

  it("every single-letter row the overlay prints is a default binding, and every default is printed", () => {
    // The same two directions as the overlay's test, read from the keymap directly rather than through
    // the page, so this holds whatever the page's comment says.
    const printed = [...GLOBAL, ...TOOLS].map((r) => r.keys).filter((k) => /^[A-Za-z]$/.test(k)).map((k) => k.toLowerCase());
    const bound = new Set(bindings().map((b) => b.default));
    expect(printed.filter((k) => !bound.has(k))).toEqual([]);
    expect([...bound].filter((k) => !printed.includes(k))).toEqual([]);
  });
});
