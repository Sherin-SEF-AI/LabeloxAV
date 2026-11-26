// The editor's rebindable keys: which single key selects each tool and fires each action, and what an
// annotator has changed about that.
//
// The registry (./registry.ts) says which tools exist and what key each ships with. This file adds the
// two things it did not have. The handful of single-key actions that are not tools (fit, confirm, the two
// review verdicts, the two India attribute keys) were literals in the page's keyboard handler, so nothing
// could list them, rebind them, or notice that a tool had been given the same letter. And nothing could
// be rebound at all: a left-handed annotator, or one on a layout where `[` needs AltGr, had the keys the
// file shipped with.
//
// An override is a per-binding key, kept in localStorage. Collisions are decided by the registry's own
// `hotkeyCollisions`, run over the registry with the overrides applied and the actions added as one more
// group per mode. That is deliberate: there is one definition of "two things want this key in this mode",
// and a rebinding is refused by the same rule the shipped registry is tested against.

import { MODES, hotkeyCollisions, type EditorMode } from "./registry";

export type ActionDef = {
  id: string;
  label: string;
  hotkey: string;
  /** Active only in these modes. */
  onlyModes?: string[];
  /** Active everywhere but these modes. */
  exceptModes?: string[];
};

// The single-key actions, with the keys the page's handler used to hardcode. `a` means two things, and
// which one is a matter of mode, which is why confirmFrame steps aside in Review rather than the two
// sharing a row: in Review the key settles the selected object, everywhere else it confirms the frame.
export const ACTIONS: ActionDef[] = [
  { id: "confirmFrame", label: "confirm the objects you reviewed", hotkey: "A", exceptModes: ["review"] },
  { id: "reviewAccept", label: "accept selected (Review mode)", hotkey: "A", onlyModes: ["review"] },
  { id: "reviewReject", label: "reject selected (Review mode)", hotkey: "X", onlyModes: ["review"] },
  { id: "toggleHelmets", label: "helmets on or off for every occupant (two/three-wheeler)", hotkey: "H" },
  { id: "bumpOccupants", label: "cycle occupant count 1 to 6 (two/three-wheeler)", hotkey: "O" },
  { id: "fit", label: "fit the frame to the window", hotkey: "F" },
  // Agent A's tools. Describe opens the phrase box (YOLO-World + SAM proposals for this frame). Sweep walks
  // this frame's machine labels one at a time in Review mode, where A, X and 1 to 9 decide each one and N
  // passes over it.
  { id: "describe", label: "describe: label what a phrase matches on this frame", hotkey: "/", onlyModes: ["objects"] },
  { id: "sweep", label: "sweep this frame's machine labels, one key per decision", hotkey: "Y",
    onlyModes: ["objects", "review"] },
  { id: "sweepSkip", label: "pass over this one for now (Sweep)", hotkey: "N", onlyModes: ["review"] },
];

/** Binding id -> key. Only bindings that differ from their default appear. */
export type Overrides = Readonly<Record<string, string>>;

export type Binding = {
  /** `tool:<key>` or `action:<id>`. A tool is one binding however many modes list it. */
  id: string;
  kind: "tool" | "action";
  label: string;
  /** The key it ships with, lower case. */
  default: string;
  /** The key in force, lower case. */
  key: string;
  /** Labels of the modes it is active in, in rail order. */
  modes: string[];
};

const toolId = (key: string) => `tool:${key}`;
const actionId = (id: string) => `action:${id}`;

const activeIn = (a: ActionDef, modeKey: string) =>
  (a.onlyModes ? a.onlyModes.includes(modeKey) : true) && !(a.exceptModes ?? []).includes(modeKey);

/** Every rebindable binding, tools in registry order and then actions, with the overrides applied. */
export function bindings(overrides: Overrides = {}): Binding[] {
  const out = new Map<string, Binding>();
  for (const m of MODES) {
    for (const g of m.groups) {
      for (const t of g.tools) {
        const id = toolId(t.key);
        const seen = out.get(id);
        if (seen) { seen.modes.push(m.label); continue; }
        const def = t.hotkey.toLowerCase();
        out.set(id, { id, kind: "tool", label: t.label, default: def, key: overrides[id] ?? def, modes: [m.label] });
      }
    }
  }
  for (const a of ACTIONS) {
    const id = actionId(a.id);
    const def = a.hotkey.toLowerCase();
    out.set(id, {
      id, kind: "action", label: a.label, default: def, key: overrides[id] ?? def,
      modes: MODES.filter((m) => activeIn(a, m.key)).map((m) => m.label),
    });
  }
  return Array.from(out.values());
}

/** The key a binding answers to right now, lower case, or null for an id that does not exist. */
export function bindingKey(overrides: Overrides, id: string): string | null {
  return bindings(overrides).find((b) => b.id === id)?.key ?? null;
}

/** The registry with rebound tool keys applied. What the tool strip shows and the handler resolves from. */
export function effectiveModes(overrides: Overrides = {}): EditorMode[] {
  return MODES.map((m) => ({
    ...m,
    groups: m.groups.map((g) => ({
      ...g,
      tools: g.tools.map((t) => ({ ...t, hotkey: (overrides[toolId(t.key)] ?? t.hotkey).toUpperCase() })),
    })),
  }));
}

/** The action a single keystroke fires in a mode, or null when that key is no action here. */
export function actionForHotkey(modeKey: string, key: string, overrides: Overrides = {}): string | null {
  const want = key.toLowerCase();
  for (const a of ACTIONS) {
    if (!activeIn(a, modeKey)) continue;
    if ((overrides[actionId(a.id)] ?? a.hotkey).toLowerCase() === want) return a.id;
  }
  return null;
}

// Everything that wants a single key in each mode: the mode's tools plus the actions active there, the
// actions dressed as one more tool group so the registry's collision check can read them unchanged.
function keyspace(overrides: Overrides): EditorMode[] {
  return effectiveModes(overrides).map((m) => ({
    ...m,
    groups: [...m.groups, {
      key: "actions", label: "Actions",
      tools: ACTIONS.filter((a) => activeIn(a, m.key)).map((a) => ({
        key: actionId(a.id), label: a.label, hotkey: overrides[actionId(a.id)] ?? a.hotkey,
      })),
    }],
  }));
}

/** Keys two bindings both claim inside one mode. Empty for the shipped registry; the test asserts it. */
export function keymapCollisions(overrides: Overrides = {}): { mode: string; hotkey: string; tools: string[] }[] {
  return hotkeyCollisions(keyspace(overrides));
}

// Keys the handler gives a fixed meaning to. Letting one be bound would produce a binding that is listed
// and never fires, because the fixed branch answers first: 1 to 9 relabel, the brackets step frames, the
// arithmetic keys zoom, backslash is the console and ? is the overlay this is edited from.
const BINDABLE = /^[a-z0;',./`]$/;

export function isBindableKey(key: string): boolean {
  return BINDABLE.test(key.toLowerCase());
}

export type RebindResult = { ok: true; overrides: Overrides } | { ok: false; reason: string };

/**
 * Rebind one binding, or say why not. Never returns a keymap with a collision in it.
 *
 * Refused rather than swapped. A swap changes a second key the annotator did not ask about, and the
 * first they would hear of it is a tool that stopped answering.
 */
export function rebind(overrides: Overrides, id: string, key: string): RebindResult {
  const all = bindings(overrides);
  const target = all.find((b) => b.id === id);
  if (!target) return { ok: false, reason: "that shortcut cannot be rebound" };
  const k = key.toLowerCase();
  if (!isBindableKey(k)) {
    return { ok: false, reason: `${printable(key)} is reserved: use a letter, 0, or one of ; ' , . / \`` };
  }
  const next: Record<string, string> = { ...overrides };
  if (k === target.default) delete next[id];
  else next[id] = k;
  // Tools carry their own key as the collision's name and actions carry their binding id, so both are
  // looked up the same way once a tool key is given its prefix back.
  const asId = (name: string) => (name.startsWith("action:") ? name : toolId(name));
  const clash = keymapCollisions(next).find((c) => c.tools.map(asId).includes(id));
  if (clash) {
    const other = clash.tools.map(asId).find((x) => x !== id);
    const label = all.find((b) => b.id === other)?.label ?? "another shortcut";
    const mode = MODES.find((m) => m.key === clash.mode)?.label ?? clash.mode;
    return { ok: false, reason: `${k.toUpperCase()} is already "${label}" in ${mode}` };
  }
  return { ok: true, overrides: next };
}

const printable = (key: string) => (key === " " ? "Space" : key.length === 1 ? key.toUpperCase() : key);

/**
 * Whatever was in storage, reduced to overrides that are still legal.
 *
 * Stored bindings outlive the registry they were made against. A tool that was renamed, or a default
 * that moved onto a key somebody had bound, must not come back as a keymap with two tools on one key,
 * so each stored entry is replayed through `rebind` and the ones it refuses are dropped.
 */
export function sanitizeOverrides(raw: unknown): Overrides {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
  let out: Overrides = {};
  for (const [id, key] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof key !== "string") continue;
    const r = rebind(out, id, key);
    if (r.ok) out = r.overrides;
  }
  return out;
}

// ---- persistence and the live store -------------------------------------------------------------------

export const KEYMAP_STORAGE_KEY = "lbx.editor.keymap";

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

const browserStorage = (): StorageLike | undefined => {
  // Safari in private mode throws on the property access itself, not only on the calls.
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
};

export function loadOverrides(storage: StorageLike | undefined = browserStorage()): Overrides {
  try {
    const raw = storage?.getItem(KEYMAP_STORAGE_KEY);
    return raw ? sanitizeOverrides(JSON.parse(raw)) : {};
  } catch {
    // Unreadable or not JSON. The shipped keys are a complete keymap, so that is what this falls back to.
    return {};
  }
}

export function saveOverrides(overrides: Overrides, storage: StorageLike | undefined = browserStorage()): void {
  try {
    if (Object.keys(overrides).length) storage?.setItem(KEYMAP_STORAGE_KEY, JSON.stringify(overrides));
    else storage?.removeItem(KEYMAP_STORAGE_KEY);
  } catch {
    /* private mode, or the quota is full. The rebinding still holds for this page. */
  }
}

// One keymap for the page: the keyboard handler, the overlay that edits it and the tool strip that prints
// the keys all read this, so a rebinding shows up in all three at once. A store rather than React state
// for the reason lib/editor/cursorStore.ts gives: the page is large, and the readers are three small
// components that can subscribe on their own.

type Listener = () => void;

const EMPTY: Overrides = Object.freeze({});
let current: Overrides = EMPTY;
const listeners = new Set<Listener>();

function publish(next: Overrides): void {
  current = next;
  for (const fn of listeners) fn();
}

export function getKeymap(): Overrides {
  return current;
}

// The server render has no localStorage. A stable empty object keeps useSyncExternalStore from looping.
export function getKeymapServerSnapshot(): Overrides {
  return EMPTY;
}

export function subscribeKeymap(fn: Listener): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

/** Read the stored keymap into the store. Called from an effect, never during render. */
export function hydrateKeymap(storage?: StorageLike): void {
  const stored = storage ? loadOverrides(storage) : loadOverrides();
  // Nothing stored is not news. The page remounts on every frame change and hydrates each time; where
  // storage cannot be written (private mode) it always reads empty, and treating that as "back to the
  // defaults" would undo a rebinding at the next press of `]`.
  if (!Object.keys(stored).length) return;
  if (JSON.stringify(stored) !== JSON.stringify(current)) publish(stored);
}

/** Rebind, persist and announce. The result says why when it was refused, and nothing changes then. */
export function setBinding(id: string, key: string, storage?: StorageLike): RebindResult {
  const r = rebind(current, id, key);
  if (!r.ok) return r;
  publish(r.overrides);
  if (storage) saveOverrides(r.overrides, storage);
  else saveOverrides(r.overrides);
  return r;
}

export function resetKeymap(storage?: StorageLike): void {
  publish(EMPTY);
  if (storage) saveOverrides(EMPTY, storage);
  else saveOverrides(EMPTY);
}
