"use client";

// The editor's local state: a working copy of the frame's objects with undo/redo, plus viewport and
// tool. Edits mutate local state; the page's Save diffs against the server and syncs. Continuous drags
// are committed once on drag-end (the canvas moves konva nodes live), so history is one entry per edit.

import { useReducer } from "react";

export type Tool = "select" | "box" | "polygon" | "polyline" | "adverse" | "cuboid" | "keypoint" | "measure" | "sam-point" | "sam-box" | "magic-wand" | "brush" | "eraser" | "superpixel";

export type EdObject = {
  id: string; // server object_id, or "tmp-N" for locally-created
  track_id?: string | null;
  class_id: number;
  class_name: string;
  bbox: number[]; // xyxy image coords
  mask: number[][]; // polygons, flattened [x,y,...] image coords
  attrs: Record<string, unknown>;
  conf: number;
  quality_score?: number | null; // M-F.1 composite label-quality signal [0,1]
  state: string;
  visible: boolean;
  // A locked object is visible but not editable: it cannot be selected by a marquee, moved, or deleted.
  // Reviewers use it to pin work they have already checked while cleaning up around it.
  locked?: boolean;
  isNew?: boolean;
  dirty?: boolean;
  version?: number; // optimistic-lock version from the server; sent back on save to detect stale writes
  rot?: number; // oriented-box rotation in degrees about the box centre (0 = axis-aligned)
  keypoints?: { skeleton: string; points: number[][] }; // COCO-style pose [[x,y,v],...]
  polyline?: number[][]; // open linear feature (curb/road_edge/barrier) as ordered [[x,y],...]
  cuboid_3d?: { center: number[]; size: number[]; yaw: number } | null; // ego-frame 3D box (size [w,l,h])
};

export type Viewport = { scale: number; ox: number; oy: number };

type Snapshot = { objects: EdObject[]; deleted: string[] };

// A history entry is a snapshot plus what the edit was and a stable identity to jump to. Kept apart from
// Snapshot so EditorState, which spreads Snapshot, does not sprout a label of its own.
export type HistoryEntry = Snapshot & { label: string; at: number };

// Ways to pick a set of objects that are not "drag a box round them". A dense frame holds forty vehicles
// and the useful selections are almost never contiguous: every autorickshaw, everything the model was
// unsure about, everything nobody has looked at yet.
export type SelectHow =
  | "all"          // every visible, unlocked object
  | "none"
  | "invert"
  | "class"        // by class name
  | "sameClass"    // the same class as the primary selection
  | "state"        // review / accepted / rejected
  | "lowConf"      // conf below a threshold
  | "unreviewed"   // still in review, which is the queue an annotator is working
  | "new";         // drawn in this session and not yet saved

export type EditorState = Snapshot & {
  selectedId: string | null;
  // Additional ids selected alongside selectedId. Kept separate from selectedId rather than replacing it so
  // every existing single-object surface (the properties panel, the class picker, the geometry handles) keeps
  // working unchanged and operates on the primary selection, while bulk actions read selectedIds.
  selectedIds: string[];
  tool: Tool;
  viewport: Viewport;
  candidate: number[][] | null; // SAM candidate polygons (image coords)
  touched: string[]; // ids the annotator actually selected/edited; "confirm frame" accepts only these
  past: HistoryEntry[];
  future: HistoryEntry[];
};

export type Action =
  | { t: "load"; objects: EdObject[]; viewport: Viewport; selectedId?: string | null }
  | { t: "tool"; tool: Tool }
  | { t: "viewport"; viewport: Viewport }
  | { t: "select"; id: string | null }
  | { t: "selectMany"; ids: string[]; additive?: boolean }
  | { t: "toggleSelect"; id: string }
  | { t: "selectBy"; how: SelectHow; value?: string | number }
  | { t: "setVisible"; ids: string[]; visible: boolean }
  | { t: "setLocked"; ids: string[]; locked: boolean }
  | { t: "candidate"; polys: number[][] | null }
  | { t: "add"; obj: EdObject }
  | { t: "update"; id: string; patch: Partial<EdObject> }
  | { t: "acceptAll" }
  | { t: "reviewed"; id: string; state: string; version?: number }
  | { t: "delete"; id: string }
  | { t: "undo" }
  | { t: "redo" }
  | { t: "jump"; at: number }
  | { t: "saved"; remap: Record<string, string>; versions?: Record<string, number> };

let _tmp = 0;
export function tmpId(): string {
  return `tmp-${++_tmp}`;
}

const snap = (s: EditorState, label = "edit"): HistoryEntry =>
  ({ objects: s.objects, deleted: s.deleted, label, at: ++_seq });

// A monotonic counter rather than a clock: history entries need a stable identity to be jumped to and
// rendered as a list, and Date.now() collides when two edits land in the same millisecond, which dragging
// a box does constantly.
let _seq = 0;

// An object id is unique by construction, so editor state must never hold two objects with the same
// id. A collision can arise when an idempotent server create returns an id already in the list (a temp
// id remapped onto an existing object after undo/redo): collapse to the most recent so the Konva layer
// never renders two children with the same key.
function uniqById(objs: EdObject[]): EdObject[] {
  const byId = new Map<string, EdObject>();
  for (const o of objs) byId.set(o.id, o);
  return Array.from(byId.values());
}

// What an update actually changed, in the words a person would use. A history that says "edit" fourteen
// times is a list nobody can navigate, which is the same as not having one.
function describeUpdate(s: EditorState, id: string, patch: Partial<EdObject>): string {
  const o = s.objects.find((x) => x.id === id);
  const name = o?.class_name ?? "object";
  if (patch.class_name && patch.class_name !== o?.class_name) {
    return `relabelled ${name} to ${patch.class_name}`;
  }
  if (patch.mask) return `edited ${name} mask`;
  if (patch.keypoints) return `posed ${name}`;
  if (patch.polyline) return `edited ${name} line`;
  if (patch.attrs) return `set ${name} attributes`;
  if (patch.state) return `marked ${name} ${patch.state}`;
  if (patch.bbox || patch.rot !== undefined) return `moved ${name}`;
  return `edited ${name}`;
}

const HISTORY_CAP = 100;

// wrap a mutating result: push current snapshot to history (capped), clear redo.
// The label describes the edit that is about to happen, not the state being stored, because that is what a
// person reading the list is looking for: "deleted 3 objects", not "12 objects".
function mutate(s: EditorState, next: Snapshot, label = "edit"): EditorState {
  return { ...s, ...next, past: [...s.past, snap(s, label)].slice(-HISTORY_CAP), future: [] };
}

// two objects differ if any human-syncable field changed (geometry / class / mask / attrs / state / rot)
function differs(a: EdObject, b: EdObject): boolean {
  return JSON.stringify([a.bbox, a.class_id, a.mask, a.attrs, a.state, a.rot ?? 0, a.keypoints ?? null])
    !== JSON.stringify([b.bbox, b.class_id, b.mask, b.attrs, b.state, b.rot ?? 0, b.keypoints ?? null]);
}

// Apply a history snapshot as the new working copy. Only objects that actually changed versus the current
// copy are marked dirty (so the revert re-syncs without rubber-stamping the rest); an object that
// reappears (undo of a delete) is re-created on save, and one that vanished (undo of a create) is queued
// for deletion. This is what lets undo/redo survive autosave instead of being wiped by it.
function restore(s: EditorState, target: Snapshot): Snapshot {
  const curById = new Map(s.objects.map((o) => [o.id, o]));
  const targetIds = new Set(target.objects.map((o) => o.id));
  const objects = target.objects.map((o) => {
    if (o.isNew) return o;
    const cur = curById.get(o.id);
    if (!cur) return { ...o, isNew: true, dirty: false };
    return differs(cur, o) ? { ...o, dirty: true } : o;
  });
  const gone = s.objects.filter((o) => !o.isNew && !targetIds.has(o.id)).map((o) => o.id);
  return { objects: uniqById(objects), deleted: Array.from(new Set([...target.deleted, ...gone])) };
}

export function isDirty(s: EditorState): boolean {
  return s.deleted.length > 0 || s.objects.some((o) => o.isNew || o.dirty);
}

// Exported for tests: this is the module where a bug corrupts annotations silently rather than throwing,
// so it is worth being able to drive it directly.
export function editorReducer(s: EditorState, a: Action): EditorState {
  switch (a.t) {
    case "load":
      return {
        objects: uniqById(a.objects),
        deleted: [],
        selectedId: a.selectedId ?? null,
        selectedIds: a.selectedId ? [a.selectedId] : [],
        tool: "select",
        viewport: a.viewport,
        candidate: null,
        touched: [],
        past: [],
        future: [],
      };
    case "tool":
      return { ...s, tool: a.tool, candidate: a.tool === "select" ? s.candidate : s.candidate };
    case "viewport":
      return { ...s, viewport: a.viewport };
    case "select":
      return { ...s, selectedId: a.id, selectedIds: a.id ? [a.id] : [],
               touched: a.id && !s.touched.includes(a.id) ? [...s.touched, a.id] : s.touched };
    case "selectMany": {
      // A locked object is deliberately excluded here rather than filtered at each action site: a marquee
      // that swept one up would otherwise let a bulk delete remove the very thing the lock was protecting.
      const lockable = new Set(s.objects.filter((o) => !o.locked).map((o) => o.id));
      const incoming = a.ids.filter((id) => lockable.has(id));
      const ids = a.additive ? Array.from(new Set([...s.selectedIds, ...incoming])) : incoming;
      return { ...s, selectedIds: ids, selectedId: ids[0] ?? null,
               touched: Array.from(new Set([...s.touched, ...ids])) };
    }
    case "toggleSelect": {
      const has = s.selectedIds.includes(a.id);
      const ids = has ? s.selectedIds.filter((x) => x !== a.id) : [...s.selectedIds, a.id];
      return { ...s, selectedIds: ids, selectedId: ids[ids.length - 1] ?? null,
               touched: s.touched.includes(a.id) ? s.touched : [...s.touched, a.id] };
    }
    case "selectBy": {
      // Locked objects are never selected. The lock exists so a bulk action cannot reach them, and a
      // select-all that ignored it would hand every bulk action the thing the lock was protecting.
      const pool = s.objects.filter((o) => !o.locked && o.visible !== false);
      const primary = s.objects.find((o) => o.id === s.selectedId);
      let ids: string[];
      switch (a.how) {
        case "all": ids = pool.map((o) => o.id); break;
        case "none": return { ...s, selectedIds: [], selectedId: null };
        case "invert": ids = pool.filter((o) => !s.selectedIds.includes(o.id)).map((o) => o.id); break;
        case "class": ids = pool.filter((o) => o.class_name === a.value).map((o) => o.id); break;
        case "sameClass":
          if (!primary) return s;
          ids = pool.filter((o) => o.class_name === primary.class_name).map((o) => o.id);
          break;
        case "state": ids = pool.filter((o) => o.state === a.value).map((o) => o.id); break;
        case "lowConf": {
          const t = typeof a.value === "number" ? a.value : 0.5;
          ids = pool.filter((o) => (o.conf ?? 1) < t).map((o) => o.id);
          break;
        }
        case "unreviewed": ids = pool.filter((o) => o.state === "review").map((o) => o.id); break;
        case "new": ids = pool.filter((o) => o.isNew).map((o) => o.id); break;
        default: return s;
      }
      // The primary selection stays whatever it was if it survived the filter, so the properties panel does
      // not jump to a different object underneath the annotator's hands.
      const keepPrimary = s.selectedId && ids.includes(s.selectedId) ? s.selectedId : (ids[0] ?? null);
      return { ...s, selectedIds: ids, selectedId: keepPrimary };
    }
    case "setVisible":
      // Visibility is a view concern, not an edit: it must not mark the object dirty or enter undo history,
      // or hiding a box to see behind it would queue a pointless save and consume an undo step.
      return { ...s, objects: s.objects.map((o) =>
        a.ids.includes(o.id) ? { ...o, visible: a.visible } : o) };
    case "setLocked":
      return { ...s, objects: s.objects.map((o) => (a.ids.includes(o.id) ? { ...o, locked: a.locked } : o)),
               // Deselect anything just locked, so the next keystroke cannot edit it.
               selectedIds: a.locked ? s.selectedIds.filter((id) => !a.ids.includes(id)) : s.selectedIds,
               selectedId: a.locked && a.ids.includes(s.selectedId ?? "") ? null : s.selectedId };
    case "candidate":
      return { ...s, candidate: a.polys };
    case "add":
      return { ...mutate(s, { objects: uniqById([...s.objects, a.obj]), deleted: s.deleted },
                         `added ${a.obj.class_name}`), selectedId: a.obj.id };
    case "update":
      return {
        ...mutate(s, {
          objects: s.objects.map((o) => (o.id === a.id ? { ...o, ...a.patch, dirty: !o.isNew } : o)),
          deleted: s.deleted,
        }, describeUpdate(s, a.id, a.patch)),
        touched: s.touched.includes(a.id) ? s.touched : [...s.touched, a.id],
      };
    case "acceptAll":
      // Confirm only objects the annotator actually looked at (selected or edited). New objects are
      // human-on-create; untouched auto-labels are left as-is rather than rubber-stamped into gold.
      return mutate(s, {
        objects: s.objects.map((o) =>
          o.isNew || !s.touched.includes(o.id) ? o : { ...o, state: "accepted", dirty: true }),
        deleted: s.deleted,
      }, `confirmed ${s.touched.length} reviewed`);
    case "reviewed":
      // A review (accept/reject) already persisted to the server via api.review with an explicit state, so
      // set the object's state and new version WITHOUT marking it dirty (dirty would make autosave re-write
      // it with the role's default accept-state, clobbering an explicit reject). Not an undoable edit.
      return {
        ...s,
        objects: s.objects.map((o) => (o.id === a.id ? { ...o, state: a.state, version: a.version ?? o.version, dirty: false } : o)),
      };
    case "delete": {
      const obj = s.objects.find((o) => o.id === a.id);
      const deleted = obj && !obj.isNew ? [...s.deleted, a.id] : s.deleted;
      return {
        ...mutate(s, { objects: s.objects.filter((o) => o.id !== a.id), deleted },
                  `deleted ${obj?.class_name ?? "object"}`),
        selectedId: s.selectedId === a.id ? null : s.selectedId,
      };
    }
    case "undo": {
      if (!s.past.length) return s;
      const prev = s.past[s.past.length - 1];
      const r = restore(s, prev);
      return { ...s, objects: r.objects, deleted: r.deleted,
        selectedId: r.objects.some((o) => o.id === s.selectedId) ? s.selectedId : null,
        past: s.past.slice(0, -1), future: [snap(s, prev.label), ...s.future] };
    }
    case "redo": {
      if (!s.future.length) return s;
      const nxt = s.future[0];
      const r = restore(s, nxt);
      return { ...s, objects: r.objects, deleted: r.deleted,
        selectedId: r.objects.some((o) => o.id === s.selectedId) ? s.selectedId : null,
        past: [...s.past, snap(s, nxt.label)], future: s.future.slice(1) };
    }
    case "jump": {
      // Move straight to a point in the history rather than stepping. Undo is fine for the last thing you
      // did and useless for "put it back to how it was before I started this junction", which is the case
      // a visible history exists to serve.
      const iPast = s.past.findIndex((h) => h.at === a.at);
      if (iPast >= 0) {
        const target = s.past[iPast];
        const r = restore(s, target);
        // Everything after the target moves to the redo stack, newest last, so stepping forward again
        // replays the same edits in the order they happened.
        const undone = [...s.past.slice(iPast + 1), snap(s, target.label)].reverse();
        return { ...s, objects: r.objects, deleted: r.deleted,
          selectedId: r.objects.some((o) => o.id === s.selectedId) ? s.selectedId : null,
          past: s.past.slice(0, iPast), future: [...undone, ...s.future] };
      }
      const iFut = s.future.findIndex((h) => h.at === a.at);
      if (iFut < 0) return s;
      const target = s.future[iFut];
      const r = restore(s, target);
      const redone = [snap(s, target.label), ...s.future.slice(0, iFut).reverse()];
      return { ...s, objects: r.objects, deleted: r.deleted,
        selectedId: r.objects.some((o) => o.id === s.selectedId) ? s.selectedId : null,
        past: [...s.past, ...redone], future: s.future.slice(iFut + 1) };
    }
    case "saved": {
      const remap = (o: EdObject): EdObject => ({ ...o, id: a.remap[o.id] ?? o.id, isNew: false });
      return {
        ...s,
        deleted: [],
        objects: uniqById(s.objects.map((o) => ({
          ...o, id: a.remap[o.id] ?? o.id, isNew: false, dirty: false,
          version: a.versions?.[o.id] ?? o.version,
        }))),
        // Keep the undo history across autosaves: remap temp ids, mark snapshot objects saved, and drop
        // pending deletes (restore() recomputes them). Without this, autosave wipes undo/redo entirely.
        // The label and the entry's identity carry through the remap. They describe the edit, which the
        // save did not change, and losing them would leave a history panel full of unnamed steps after the
        // first autosave.
        past: s.past.map((sn) => ({ ...sn, objects: sn.objects.map(remap), deleted: [] })),
        future: s.future.map((sn) => ({ ...sn, objects: sn.objects.map(remap), deleted: [] })),
      };
    }
    default:
      return s;
  }
}

const INITIAL: EditorState = {
  objects: [],
  deleted: [],
  selectedId: null,
  selectedIds: [],
  tool: "select",
  viewport: { scale: 1, ox: 0, oy: 0 },
  candidate: null,
  touched: [],
  past: [],
  future: [],
};

export function useEditor() {
  return useReducer(editorReducer, INITIAL);
}
