// The console was a page, so reading it meant leaving whatever raised the question.

import { describe, expect, it } from "vitest";

import { CONSOLE_SECTIONS, filterSections, groupSections, resolveSelection } from "./consoleSections";

describe("finding a section by typing", () => {
  it("matches the words people actually reach for, not the words on the row", () => {
    // Neither "vram" nor "disk full" appears in a section's name, and both are how the two questions this
    // console exists for get phrased.
    expect(filterSections("vram").map((s) => s.id)).toEqual(["gpu"]);
    expect(filterSections("disk full").map((s) => s.id)).toEqual(["machine"]);
  });

  it("narrows on a second word rather than widening", () => {
    const one = filterSections("memory");
    const two = filterSections("memory disk");
    expect(two.length).toBeLessThan(one.length);
  });

  it("ignores case and surrounding space", () => {
    expect(filterSections("  GPU  ").map((s) => s.id)).toEqual(["gpu"]);
  });

  it("returns everything for an empty query, in declared order", () => {
    expect(filterSections("")).toHaveLength(CONSOLE_SECTIONS.length);
    expect(filterSections("").map((s) => s.id)).toEqual(CONSOLE_SECTIONS.map((s) => s.id));
  });

  it("returns nothing rather than everything when nothing matches", () => {
    expect(filterSections("kubernetes")).toEqual([]);
  });
});

describe("what stays selected while typing", () => {
  it("keeps the current section when it still matches", () => {
    // Typing must feel like filtering. Jumping to the first result would move the reader off the panel
    // they are reading as soon as they typed a letter.
    expect(resolveSelection("memory", "machine")).toBe("machine");
  });

  it("moves to the first match when the current one is filtered out", () => {
    expect(resolveSelection("vram", "machine")).toBe("gpu");
  });

  it("says there is nothing to show rather than picking something arbitrary", () => {
    expect(resolveSelection("kubernetes", "gpu")).toBeNull();
  });
});

describe("the sidebar grouping", () => {
  it("keeps declared order and does not repeat a heading", () => {
    const groups = groupSections(CONSOLE_SECTIONS);
    expect(groups.map((g) => g.group)).toEqual(["Activity", "Machine"]);
    expect(groups[0].items.map((i) => i.id)).toEqual(["overview", "jobs", "background", "canvas"]);
  });

  it("groups whatever it is given, so a filtered list keeps its headings", () => {
    const groups = groupSections(filterSections("memory"));
    expect(groups.flatMap((g) => g.items).length).toBe(filterSections("memory").length);
  });

  it("is empty for an empty list rather than emitting a heading with nothing under it", () => {
    expect(groupSections([])).toEqual([]);
  });
});
