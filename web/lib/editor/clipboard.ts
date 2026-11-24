// The editor's object clipboard.
//
// It was a `useRef` on the frame page holding one object. Two things followed from that. Copy took the
// primary selection only, so six selected boxes copied one. And the page is a route: `[` and `]` remount
// it, so the ref was empty on the next frame, which is the one place a copied box is most wanted (the
// same parked truck, one frame later).
//
// So it lives at module level, which survives the remount, and is mirrored to sessionStorage, which
// survives a reload of the tab. What is kept is geometry, class and attributes and nothing that says
// which object this was: no id, no track, no version, no review state. A paste makes a new object a
// person put there, on whatever frame it lands, and must not arrive claiming to be part of the source
// frame's track.

import type { EdObject } from "@/components/editor/useEditor";
import { translateObject } from "./geometry";

export type ClipItem = Pick<EdObject, "class_id" | "class_name" | "bbox" | "mask" | "attrs"
  | "rot" | "bbox_amodal" | "keypoints" | "polyline" | "cuboid_3d">;

export type Clipboard = {
  /** The frame the objects were copied from. */
  frameId: string;
  items: ClipItem[];
  /** Where the last paste went, and how many have gone there, so repeated pastes do not stack. */
  pastedOn: string | null;
  pastes: number;
};

/** How far each successive paste is offset, in image pixels. */
export const PASTE_STEP = 14;

/** Strip identity and copy every array, so a later edit to the source cannot reach the clipboard. */
export function toClipItems(objects: EdObject[]): ClipItem[] {
  return objects.map((o) => {
    const c = translateObject(o, 0, 0);
    const item: ClipItem = {
      class_id: c.class_id, class_name: c.class_name, bbox: c.bbox, mask: c.mask, attrs: c.attrs,
    };
    if (c.rot !== undefined) item.rot = c.rot;
    if (c.bbox_amodal?.length === 4) item.bbox_amodal = c.bbox_amodal;
    if (c.keypoints) item.keypoints = c.keypoints;
    if (c.polyline) item.polyline = c.polyline;
    if (c.cuboid_3d) item.cuboid_3d = c.cuboid_3d;
    return item;
  });
}

/**
 * How far to offset the next paste on `frameId`.
 *
 * On the frame it was copied from, the first paste is one step away, or it would land exactly on its
 * source and look like nothing happened. On any other frame the first paste is in place: the point of
 * carrying a box to the next frame is that it is already where the object was. Later pastes step on from
 * there either way.
 */
export function pasteOffset(clip: Clipboard, frameId: string): number {
  const already = clip.pastedOn === frameId ? clip.pastes : 0;
  return PASTE_STEP * (already + (clip.frameId === frameId ? 1 : 0));
}

/**
 * New objects from clipboard items: fresh ids, human-authored, every geometry kind carried.
 *
 * A cuboid is carried as it is. It sits in the ego frame in metres, where an offset in image pixels has
 * no meaning, so its 2D box moves with the paste and the 3D box is left for the annotator to place.
 */
export function materialize(items: ClipItem[], offset: number, newId: () => string): EdObject[] {
  return items.map((item) => translateObject({
    ...item, id: newId(), conf: 1, state: "accepted", visible: true, isNew: true,
  }, offset, offset));
}

const isNumbers = (v: unknown, n?: number): v is number[] =>
  Array.isArray(v) && (n === undefined || v.length === n) && v.every((x) => typeof x === "number" && Number.isFinite(x));

const isPoints = (v: unknown): v is number[][] => Array.isArray(v) && v.every((p) => isNumbers(p));

function parseItem(raw: unknown): ClipItem | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.class_id !== "number" || typeof r.class_name !== "string") return null;
  if (!isNumbers(r.bbox, 4) || !isPoints(r.mask)) return null;
  const item: ClipItem = {
    class_id: r.class_id, class_name: r.class_name, bbox: r.bbox, mask: r.mask,
    attrs: r.attrs && typeof r.attrs === "object" && !Array.isArray(r.attrs) ? r.attrs as Record<string, unknown> : {},
  };
  if (typeof r.rot === "number") item.rot = r.rot;
  if (isNumbers(r.bbox_amodal, 4)) item.bbox_amodal = r.bbox_amodal;
  if (isPoints(r.polyline)) item.polyline = r.polyline;
  const kp = r.keypoints as { skeleton?: unknown; points?: unknown } | null | undefined;
  if (kp && typeof kp.skeleton === "string" && isPoints(kp.points)) {
    item.keypoints = { skeleton: kp.skeleton, points: kp.points };
  }
  const cu = r.cuboid_3d as { center?: unknown; size?: unknown; yaw?: unknown; yaw_source?: unknown } | null | undefined;
  if (cu && isNumbers(cu.center) && isNumbers(cu.size) && typeof cu.yaw === "number") {
    item.cuboid_3d = { center: cu.center, size: cu.size, yaw: cu.yaw,
                       ...(typeof cu.yaw_source === "string" ? { yaw_source: cu.yaw_source } : {}) };
  }
  return item;
}

/**
 * A clipboard from what was stored, or null when it is not one.
 *
 * sessionStorage is a string somebody else can have written, and what comes out of it is about to become
 * annotation geometry. An item that is not the right shape is dropped rather than pasted as an object
 * with a three-number box.
 */
export function parseClipboard(raw: string | null | undefined): Clipboard | null {
  if (!raw) return null;
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!v || typeof v !== "object") return null;
  const r = v as Record<string, unknown>;
  if (typeof r.frameId !== "string" || !Array.isArray(r.items)) return null;
  const items = r.items.map(parseItem).filter((x): x is ClipItem => x !== null);
  if (!items.length) return null;
  return {
    frameId: r.frameId, items,
    pastedOn: typeof r.pastedOn === "string" ? r.pastedOn : null,
    pastes: typeof r.pastes === "number" && r.pastes > 0 ? Math.floor(r.pastes) : 0,
  };
}

// ---- the store ----------------------------------------------------------------------------------------

export const CLIPBOARD_STORAGE_KEY = "lbx.editor.clipboard";

type StorageLike = Pick<Storage, "getItem" | "setItem">;

const sessionStore = (): StorageLike | undefined => {
  // Safari in private mode throws on the property access itself, not only on the calls.
  try {
    return globalThis.sessionStorage;
  } catch {
    return undefined;
  }
};

let current: Clipboard | null = null;
let hydrated = false;

function persist(storage: StorageLike | undefined): void {
  try {
    storage?.setItem(CLIPBOARD_STORAGE_KEY, JSON.stringify(current));
  } catch {
    /* Quota: a few hundred SAM masks can exceed it. The copy still holds in memory, which is what
       frame-to-frame navigation uses; only surviving a tab reload is lost. */
  }
}

/** Put objects on the clipboard. Returns how many went. Copying nothing leaves the clipboard alone. */
export function copyObjects(frameId: string, objects: EdObject[],
                            storage: StorageLike | undefined = sessionStore()): number {
  if (!objects.length) return 0;
  current = { frameId, items: toClipItems(objects), pastedOn: null, pastes: 0 };
  hydrated = true;
  persist(storage);
  return objects.length;
}

/** The clipboard, read back from sessionStorage the first time if this page load has not copied anything. */
export function readClipboard(storage: StorageLike | undefined = sessionStore()): Clipboard | null {
  if (!hydrated) {
    hydrated = true;
    try {
      current = parseClipboard(storage?.getItem(CLIPBOARD_STORAGE_KEY));
    } catch {
      current = null;
    }
  }
  return current;
}

/**
 * The objects a paste on `frameId` creates, or an empty list when there is nothing to paste.
 * Counts the paste, so the next one lands a step further on.
 */
export function pasteObjects(frameId: string, newId: () => string,
                             storage: StorageLike | undefined = sessionStore()): EdObject[] {
  const clip = readClipboard(storage);
  if (!clip) return [];
  const out = materialize(clip.items, pasteOffset(clip, frameId), newId);
  current = { ...clip, pastedOn: frameId, pastes: (clip.pastedOn === frameId ? clip.pastes : 0) + 1 };
  persist(storage);
  return out;
}

/** Test seam: forget everything, including that storage was ever read. */
export function resetClipboardForTests(): void {
  current = null;
  hydrated = false;
}
