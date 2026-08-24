import { describe, expect, it } from "vitest";

import type { EdObject } from "../useEditor";
import { groupObjects, matchesQuery } from "./objectGroups";

// Two of these assertions guard behaviour that looks fine when broken.
//
// Matching on the object id as well as the class name is why an annotator can paste an id out of a review
// queue and land on the row. Drop it and the search still works, just not for the case it was added for.
//
// Row order inside a group is the server's order, which is drawing order. Sorting it would silently break
// the correspondence between the list and the canvas stack, and nothing on screen would say so.

function obj(id: string, class_name: string, over: Partial<EdObject> = {}): EdObject {
  return {
    id, class_id: 1, class_name, bbox: [0, 0, 10, 10], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, ...over,
  };
}

describe("matchesQuery", () => {
  it("matches the class name case-insensitively", () => {
    expect(matchesQuery(obj("a", "autorickshaw"), "AUTO")).toBe(true);
    expect(matchesQuery(obj("a", "autorickshaw"), "sedan")).toBe(false);
  });

  it("also matches the object id, which is the reason ids are searchable at all", () => {
    expect(matchesQuery(obj("c11881d9", "sedan"), "c1188")).toBe(true);
    expect(matchesQuery(obj("C11881D9", "sedan"), "c1188")).toBe(true);
  });

  it("an empty or whitespace query matches everything", () => {
    expect(matchesQuery(obj("a", "sedan"), "")).toBe(true);
    expect(matchesQuery(obj("a", "sedan"), "   ")).toBe(true);
  });
});

describe("groupObjects", () => {
  it("groups by class and sorts the groups by name", () => {
    const groups = groupObjects([
      obj("1", "motorcycle"), obj("2", "autorickshaw"), obj("3", "motorcycle"), obj("4", "hoarding"),
    ], "");
    expect(groups.map((g) => g.name)).toEqual(["autorickshaw", "hoarding", "motorcycle"]);
    expect(groups.map((g) => g.objects.length)).toEqual([1, 1, 2]);
  });

  it("keeps row order inside a group as it came in", () => {
    const groups = groupObjects([obj("z", "sedan"), obj("a", "sedan"), obj("m", "sedan")], "");
    expect(groups[0].objects.map((o) => o.id)).toEqual(["z", "a", "m"]);
  });

  it("takes the swatch colour from the first member of each group", () => {
    const groups = groupObjects([obj("1", "sedan", { class_id: 7 }), obj("2", "sedan", { class_id: 9 })], "");
    expect(groups[0].classId).toBe(7);
  });

  it("does not emit a group header for a class whose every member was filtered out", () => {
    // An empty group reads as "this class is present but collapsed", which is a different fact.
    const groups = groupObjects([obj("1", "sedan"), obj("2", "motorcycle")], "motor");
    expect(groups.map((g) => g.name)).toEqual(["motorcycle"]);
  });

  it("returns nothing when the query matches nothing", () => {
    expect(groupObjects([obj("1", "sedan")], "tanker")).toEqual([]);
  });

  it("survives a class named like an Object prototype key", () => {
    // Custom classes come from a free-text box, so these are reachable user input rather than paranoia.
    const groups = groupObjects([obj("1", "constructor"), obj("2", "__proto__"), obj("3", "sedan")], "");
    expect(groups.map((g) => g.name).sort()).toEqual(["__proto__", "constructor", "sedan"]);
    expect(groups.every((g) => g.objects.length === 1)).toBe(true);
  });
});
