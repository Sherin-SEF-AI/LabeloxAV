"use client";

// The editor's local state: a working copy of the frame's objects with undo/redo, plus viewport and
// tool. Edits mutate local state; the page's Save diffs against the server and syncs. Continuous drags
// are committed once on drag-end (the canvas moves konva nodes live), so history is one entry per edit.

import { useReducer } from "react";

export type Tool = "select" | "box" | "polygon" | "keypoint" | "measure" | "sam-point" | "sam-box";

export type EdObject = {
  id: string; // server object_id, or "tmp-N" for locally-created
  track_id?: string | null;
  class_id: number;
  class_name: string;
  bbox: number[]; // xyxy image coords
  mask: number[][]; // polygons, flattened [x,y,...] image coords
  attrs: Record<string, unknown>;
  conf: number;
  state: string;
  visible: boolean;
  isNew?: boolean;
  dirty?: boolean;
  version?: number; // optimistic-lock version from the server; sent back on save to detect stale writes
  rot?: number; // oriented-box rotation in degrees about the box centre (0 = axis-aligned)
  keypoints?: { skeleton: string; points: number[][] }; // COCO-style pose [[x,y,v],...]
};

export type Viewport = { scale: number; ox: number; oy: number };

type Snapshot = { objects: EdObject[]; deleted: string[] };

export type EditorState = Snapshot & {
  selectedId: string | null;
  tool: Tool;
  viewport: Viewport;
  candidate: number[][] | null; // SAM candidate polygons (image coords)
  touched: string[]; // ids the annotator actually selected/edited; "confirm frame" accepts only these
  past: Snapshot[];
  future: Snapshot[];
};

export type Action =
  | { t: "load"; objects: EdObject[]; viewport: Viewport; selectedId?: string | null }
  | { t: "tool"; tool: Tool }
  | { t: "viewport"; viewport: Viewport }
  | { t: "select"; id: string | null }
  | { t: "candidate"; polys: number[][] | null }
  | { t: "add"; obj: EdObject }
  | { t: "update"; id: string; patch: Partial<EdObject> }
  | { t: "acceptAll" }
  | { t: "delete"; id: string }
  | { t: "undo" }
  | { t: "redo" }
  | { t: "saved"; remap: Record<string, string>; versions?: Record<string, number> };

let _tmp = 0;
export function tmpId(): string {
  return `tmp-${++_tmp}`;
}

const snap = (s: EditorState): Snapshot => ({ objects: s.objects, deleted: s.deleted });

// An object id is unique by construction, so editor state must never hold two objects with the same
// id. A collision can arise when an idempotent server create returns an id already in the list (a temp
// id remapped onto an existing object after undo/redo): collapse to the most recent so the Konva layer
// never renders two children with the same key.
function uniqById(objs: EdObject[]): EdObject[] {
  const byId = new Map<string, EdObject>();
  for (const o of objs) byId.set(o.id, o);
  return Array.from(byId.values());
}

const HISTORY_CAP = 100;

// wrap a mutating result: push current snapshot to history (capped), clear redo
function mutate(s: EditorState, next: Snapshot): EditorState {
  return { ...s, ...next, past: [...s.past, snap(s)].slice(-HISTORY_CAP), future: [] };
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

function reducer(s: EditorState, a: Action): EditorState {
  switch (a.t) {
    case "load":
      return {
        objects: uniqById(a.objects),
        deleted: [],
        selectedId: a.selectedId ?? null,
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
      return { ...s, selectedId: a.id, touched: a.id && !s.touched.includes(a.id) ? [...s.touched, a.id] : s.touched };
    case "candidate":
      return { ...s, candidate: a.polys };
    case "add":
      return { ...mutate(s, { objects: uniqById([...s.objects, a.obj]), deleted: s.deleted }), selectedId: a.obj.id };
    case "update":
      return {
        ...mutate(s, {
          objects: s.objects.map((o) => (o.id === a.id ? { ...o, ...a.patch, dirty: !o.isNew } : o)),
          deleted: s.deleted,
        }),
        touched: s.touched.includes(a.id) ? s.touched : [...s.touched, a.id],
      };
    case "acceptAll":
      // Confirm only objects the annotator actually looked at (selected or edited). New objects are
      // human-on-create; untouched auto-labels are left as-is rather than rubber-stamped into gold.
      return mutate(s, {
        objects: s.objects.map((o) =>
          o.isNew || !s.touched.includes(o.id) ? o : { ...o, state: "accepted", dirty: true }),
        deleted: s.deleted,
      });
    case "delete": {
      const obj = s.objects.find((o) => o.id === a.id);
      const deleted = obj && !obj.isNew ? [...s.deleted, a.id] : s.deleted;
      return {
        ...mutate(s, { objects: s.objects.filter((o) => o.id !== a.id), deleted }),
        selectedId: s.selectedId === a.id ? null : s.selectedId,
      };
    }
    case "undo": {
      if (!s.past.length) return s;
      const prev = s.past[s.past.length - 1];
      const r = restore(s, prev);
      return { ...s, objects: r.objects, deleted: r.deleted,
        selectedId: r.objects.some((o) => o.id === s.selectedId) ? s.selectedId : null,
        past: s.past.slice(0, -1), future: [snap(s), ...s.future] };
    }
    case "redo": {
      if (!s.future.length) return s;
      const nxt = s.future[0];
      const r = restore(s, nxt);
      return { ...s, objects: r.objects, deleted: r.deleted,
        selectedId: r.objects.some((o) => o.id === s.selectedId) ? s.selectedId : null,
        past: [...s.past, snap(s)], future: s.future.slice(1) };
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
        past: s.past.map((sn) => ({ objects: sn.objects.map(remap), deleted: [] })),
        future: s.future.map((sn) => ({ objects: sn.objects.map(remap), deleted: [] })),
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
  tool: "select",
  viewport: { scale: 1, ox: 0, oy: 0 },
  candidate: null,
  touched: [],
  past: [],
  future: [],
};

export function useEditor() {
  return useReducer(reducer, INITIAL);
}
