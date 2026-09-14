import { describe, expect, it } from "vitest";

import { attrKeymap, noKeymapReason } from "./attrKeys";

// Two things are being pinned here, and only one of them is the happy path.
//
// The first is that the bindings come from the ontology in ontology order, so the badge printed next to a
// value is a true promise about which key sets it. A drifted table would bind 3 to the wrong load type
// with no visible symptom: the annotator presses 3, something is recorded, and it is wrong.
//
// The second is that some attributes get no keyboard at all. `truncation` is a measured float and
// `helmet` is one boolean per rider; a mode that offered a key for either would record a guess that reads
// exactly like an answer. null is the correct output and the caller has to handle it.

describe("attrKeymap", () => {
  it("numbers enum values in ontology order", () => {
    // The real load_type values, in the order the YAML declares them.
    const m = attrKeymap({ type: "enum", values: ["none", "goods", "construction_material", "agricultural"] });
    expect(m).not.toBeNull();
    expect(m!.multi).toBe(false);
    expect(m!.keys.map((k) => [k.key, k.value])).toEqual([
      ["1", "none"], ["2", "goods"], ["3", "construction_material"], ["4", "agricultural"],
    ]);
  });

  it("gives a binary question letters rather than an ordering it does not have", () => {
    const m = attrKeymap({ type: "bool" });
    expect(m!.keys.map((k) => k.key)).toEqual(["y", "n"]);
    expect(m!.keys.map((k) => k.value)).toEqual([true, false]);
  });

  it("binds an integer range to the digits that are the answer", () => {
    // occupant_count is 0..6, so pressing 4 means four occupants and not "the fourth option".
    const m = attrKeymap({ type: "int", range: [0, 6] });
    expect(m!.keys.map((k) => k.key)).toEqual(["0", "1", "2", "3", "4", "5", "6"]);
    expect(m!.keys.map((k) => k.value)).toEqual([0, 1, 2, 3, 4, 5, 6]);
  });

  it("refuses an integer range too wide for the number row", () => {
    // group_size runs 1..50. Nobody counts a herd with one keystroke.
    expect(attrKeymap({ type: "int", range: [1, 50] })).toBeNull();
    expect(noKeymapReason({ type: "int", range: [1, 50] })).toContain("too wide");
  });

  it("refuses a float and says why", () => {
    expect(attrKeymap({ type: "float", range: [0, 1] })).toBeNull();
    expect(noKeymapReason({ type: "float" })).toContain("measured value");
  });

  it("refuses a per-occupant boolean array and says why", () => {
    // helmet is one value per rider, so the answer depends on how many riders there are; defaulting it
    // to all-false would record a claim nobody made about a scooter carrying three.
    expect(attrKeymap({ type: "bool_array" })).toBeNull();
    expect(noKeymapReason({ type: "bool_array" })).toContain("per occupant");
  });

  it("marks multi_select as needing a commit", () => {
    // A Bengaluru signboard routinely carries Kannada and English, so one press cannot be the answer.
    const m = attrKeymap({ type: "multi_select", values: ["kannada", "latin", "devanagari"] });
    expect(m!.multi).toBe(true);
    expect(m!.keys).toHaveLength(3);
  });

  it("leaves values past the ninth without a key rather than wrapping onto letters", () => {
    const values = Array.from({ length: 12 }, (_, i) => `v${i}`);
    const m = attrKeymap({ type: "enum", values });
    expect(m!.keys).toHaveLength(9);
    expect(m!.keys.at(-1)!.value).toBe("v8");
  });

  it("returns null for an enum with no values rather than an empty keyboard", () => {
    expect(attrKeymap({ type: "enum", values: [] })).toBeNull();
    expect(attrKeymap({ type: "enum", values: null })).toBeNull();
  });
});
