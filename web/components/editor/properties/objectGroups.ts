// Filtering and grouping the object list, lifted out of the inline IIFE it used to live in so the two
// rules it encodes can be tested without rendering the editor.
//
// Both rules are easy to get subtly wrong in a rewrite and invisible when you do:
//
//   - The search matches the class name OR the object id. An annotator pasting an id from a review queue
//     is the reason the id is in there at all, and a rewrite that only matches the class name still looks
//     like a working search.
//   - Groups sort by localeCompare, rows keep their incoming order. The incoming order is the server's,
//     which is drawing order, so re-sorting rows inside a group silently breaks the correspondence
//     between the list and the canvas stacking.

import type { EdObject } from "../useEditor";

export type ObjectGroup = {
  /** class_name, which is also the collapse key. */
  name: string;
  /** Taken from the first member, only so the swatch has a colour to draw. */
  classId: number;
  objects: EdObject[];
};

export function matchesQuery(o: EdObject, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return o.class_name.toLowerCase().includes(q) || o.id.toLowerCase().includes(q);
}

export function groupObjects(objects: EdObject[], query: string): ObjectGroup[] {
  const groups = new Map<string, EdObject[]>();
  for (const o of objects) {
    if (!matchesQuery(o, query)) continue;
    // A Map rather than a plain object so a class named "constructor" or "__proto__" cannot collide with
    // Object.prototype. The ontology takes custom classes from a free-text box, so those names are user
    // input.
    const bucket = groups.get(o.class_name);
    if (bucket) bucket.push(o);
    else groups.set(o.class_name, [o]);
  }
  // Filtering before grouping is what keeps a class whose every member was filtered out from producing an
  // empty group header, which would read as "this class is here but collapsed".
  return [...groups.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([name, objs]) => ({ name, classId: objs[0].class_id, objects: objs }));
}
