// Cmd+K, type "jobs", press Enter, arrive at Projects.
//
// The palette matched label plus hint as one substring and showed the results in menu order. Projects sits
// above Jobs in the Label menu and its hint reads "assign jobs, stages, scorecards", so the page whose name
// is literally the word typed came second. Reported from the running app.

import { describe, expect, it } from "vitest";

import { MENU_DESTINATIONS } from "./menus";
import { rankDestinations } from "./paletteRank";

const items = [
  { href: "/projects", label: "Projects", hint: "assign jobs, stages, scorecards" },
  { href: "/review/queue", label: "Review queue", hint: "active learning and error candidates" },
  { href: "/jobs", label: "Jobs", hint: "import, training and autolabel runs" },
  { href: "/curation", label: "Curation", hint: "frame-level active learning" },
];

describe("the reported case", () => {
  it("puts the page actually called Jobs first", () => {
    expect(rankDestinations(items, "jobs")[0].href).toBe("/jobs");
  });

  it("holds against the real menu, not just a fixture", () => {
    const first = rankDestinations(MENU_DESTINATIONS, "jobs")[0];
    expect(first.href).toBe("/jobs");
  });
});

describe("what beats what", () => {
  it("an exact name beats a name that merely starts with it", () => {
    const list = [{ href: "/a", label: "Jobsboard" }, { href: "/b", label: "Jobs" }];
    expect(rankDestinations(list, "jobs")[0].href).toBe("/b");
  });

  it("a name beats a hint that mentions the same word, without hiding the hint match", () => {
    // Both of these match "jobs": one is called it, one describes itself with it. Order matters and so does
    // keeping the second, since somebody may well have meant Projects.
    expect(rankDestinations(items, "jobs").map((d) => d.href)).toEqual(["/jobs", "/projects"]);
  });

  it("a word inside the name beats a hint match", () => {
    // "Review queue" contains the word; nothing else does except by description.
    expect(rankDestinations(items, "review")[0].href).toBe("/review/queue");
  });

  it("still finds a page by what it does, which is the point of hints", () => {
    // Somebody thinking "active learning" and not "Curation" must still get there.
    const hrefs = rankDestinations(items, "active learning").map((d) => d.href);
    expect(hrefs).toContain("/curation");
  });
});

describe("the edges", () => {
  it("an empty query is everything, in menu order", () => {
    expect(rankDestinations(items, "").map((d) => d.href)).toEqual(items.map((d) => d.href));
    expect(rankDestinations(items, "   ").map((d) => d.href)).toEqual(items.map((d) => d.href));
  });

  it("ignores case and surrounding space, since people type both", () => {
    expect(rankDestinations(items, "  JOBS ")[0].href).toBe("/jobs");
  });

  it("no match is an empty list rather than everything", () => {
    expect(rankDestinations(items, "zzzzz")).toEqual([]);
  });

  it("regex metacharacters are a search for those characters, not a pattern", () => {
    // The word-boundary test builds a RegExp from the query. Unescaped, "c++" throws and takes the palette
    // down on a keystroke.
    const list = [{ href: "/a", label: "C++ export" }, { href: "/b", label: "Other" }];
    expect(() => rankDestinations(list, "c++")).not.toThrow();
    expect(rankDestinations(list, "c++")[0].href).toBe("/a");
  });

  it("does not drop a destination that has no hint at all", () => {
    const list = [{ href: "/a", label: "Jobs" }];
    expect(rankDestinations(list, "jobs")).toHaveLength(1);
  });

  it("keeps ties in menu order so the list does not jump around", () => {
    const list = [{ href: "/a", label: "Alpha", hint: "shared" },
                  { href: "/b", label: "Beta", hint: "shared" }];
    expect(rankDestinations(list, "shared").map((d) => d.href)).toEqual(["/a", "/b"]);
  });
});
