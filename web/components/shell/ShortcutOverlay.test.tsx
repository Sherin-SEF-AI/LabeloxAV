import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { GLOBAL, TOOLS } from "./ShortcutOverlay";

// The coupling test this reference never had.
//
// The overlay is a hand-written list and the bindings live in a 2,000-line keyboard handler in the frame
// editor. Nothing connected them, so the two could drift in either direction and neither would fail: a key
// removed from the handler still appears here, and an annotator learns a shortcut that does nothing.
//
// This reads the handler's source rather than simulating keystrokes. Simulating would need the whole editor
// page mounted with an ontology, a frame and a canvas, which is a different and much heavier test; what is
// actually in question is whether the two lists agree about which letters exist.

const PAGE = readFileSync(join(__dirname, "../../app/frame/[id]/page.tsx"), "utf8");

/** Single letters the handler binds, from its `k === "x"` chain. */
function boundLetters(src: string): Set<string> {
  return new Set([...src.matchAll(/k === "([a-z])"/g)].map((m) => m[1]));
}

/** Single letters the overlay claims, ignoring the chorded and named rows. */
function claimedLetters(rows: { keys: string }[]): string[] {
  return rows.map((r) => r.keys).filter((k) => /^[A-Za-z]$/.test(k)).map((k) => k.toLowerCase());
}

describe("the shortcut overlay matches the editor keymap", () => {
  it("claims no letter the handler does not bind", () => {
    const bound = boundLetters(PAGE);
    const unbound = [...claimedLetters(GLOBAL), ...claimedLetters(TOOLS)].filter((k) => !bound.has(k));
    expect(unbound).toEqual([]);
  });

  it("documents every letter the handler binds", () => {
    const claimed = new Set([...claimedLetters(GLOBAL), ...claimedLetters(TOOLS)]);
    const undocumented = [...boundLetters(PAGE)].filter((k) => !claimed.has(k));
    expect(undocumented).toEqual([]);
  });

  it("reads the handler at all", () => {
    // If the path ever moves, both assertions above pass vacuously on an empty set.
    expect(boundLetters(PAGE).size).toBeGreaterThan(10);
  });

  it("documents the two India attribute keys", () => {
    const labels = GLOBAL.map((r) => `${r.keys} ${r.label}`).join("\n");
    expect(labels).toMatch(/^H .*helmet/m);
    expect(labels).toMatch(/^O .*occupant/m);
  });
});
