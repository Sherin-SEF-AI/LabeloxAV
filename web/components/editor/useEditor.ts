"use client";

// The editor's local state: a working copy of the frame's objects with undo/redo, plus viewport and
// tool. Edits mutate local state; the page's Save diffs against the server and syncs. Continuous drags
// are committed once on drag-end (the canvas moves konva nodes live), so history is one entry per edit.

import { useReducer } from "react";

export type Tool = "select" | "box" | "sam-point" | "sam-box";

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
};

export type Viewport = { scale: number; ox: number; oy: number };

type Snapshot = { objects: EdObject[]; deleted: string[] };

export type EditorState = Snapshot & {
  selectedId: string | null;
  tool: Tool;
  viewport: Viewport;
  candidate: number[][] | null; // SAM candidate polygons (image coords)
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
  | { t: "saved"; remap: Record<string, string> };

let _tmp = 0;
export function tmpId(): string {
  return `tmp-${++_tmp}`;
}

const snap = (s: EditorState): Snapshot => ({ objects: s.objects, deleted: s.deleted });

// wrap a mutating result: push current snapshot to history, clear redo
function mutate(s: EditorState, next: Snapshot): EditorState {
  return { ...s, ...next, past: [...s.past, snap(s)], future: [] };
}

export function isDirty(s: EditorState): boolean {
  return s.deleted.length > 0 || s.objects.some((o) => o.isNew || o.dirty);
}

function reducer(s: EditorState, a: Action): EditorState {
  switch (a.t) {
    case "load":
      return {
        objects: a.objects,
        deleted: [],
        selectedId: a.selectedId ?? null,
        tool: "select",
        viewport: a.viewport,
        candidate: null,
        past: [],
        future: [],
      };
    case "tool":
      return { ...s, tool: a.tool, candidate: a.tool === "select" ? s.candidate : s.candidate };
    case "viewport":
      return { ...s, viewport: a.viewport };
    case "select":
      return { ...s, selectedId: a.id };
    case "candidate":
      return { ...s, candidate: a.polys };
    case "add":
      return { ...mutate(s, { objects: [...s.objects, a.obj], deleted: s.deleted }), selectedId: a.obj.id };
    case "update":
      return mutate(s, {
        objects: s.objects.map((o) => (o.id === a.id ? { ...o, ...a.patch, dirty: !o.isNew } : o)),
        deleted: s.deleted,
      });
    case "acceptAll":
      // Confirm every object as human-verified gold. New objects are already human-on-create; existing
      // ones get state=accepted + dirty so Save sends a review (source=human, state=accepted).
      return mutate(s, {
        objects: s.objects.map((o) => (o.isNew ? o : { ...o, state: "accepted", dirty: true })),
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
      return { ...s, ...prev, past: s.past.slice(0, -1), future: [snap(s), ...s.future] };
    }
    case "redo": {
      if (!s.future.length) return s;
      const nxt = s.future[0];
      return { ...s, ...nxt, past: [...s.past, snap(s)], future: s.future.slice(1) };
    }
    case "saved":
      return {
        ...s,
        deleted: [],
        objects: s.objects.map((o) => ({ ...o, id: a.remap[o.id] ?? o.id, isNew: false, dirty: false })),
        past: [],
        future: [],
      };
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
  past: [],
  future: [],
};

export function useEditor() {
  return useReducer(reducer, INITIAL);
}
