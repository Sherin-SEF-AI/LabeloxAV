import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { MODES } from "@/lib/editor/registry";

import { GLOBAL, TOOLS } from "./ShortcutOverlay";

// The coupling test this reference never had.
//
// The overlay is a hand-written list and the bindings used to live in a 2,000-line keyboard handler in the
// frame editor. Nothing connected them, so the two could drift in either direction and neither would fail:
// a key removed from the handler still appeared here, and an annotator learned a shortcut that did nothing.
//
// Tool bindings now come from lib/editor/registry.ts, so this reads them from there rather than scraping a
// regex over the page, which is both stronger and no longer dependent on how the handler happens to be
// written. What remains in the handler is the handful of non-tool letters, and those are still scraped,
// because they genuinely are written there.

const PAGE = readFileSync(join(__dirname, "../../app/frame/[id]/page.tsx"), "utf8");

/** Single letters the handler binds directly, from its `k === "x"` chain. */
function handlerLetters(src: string): Set<string> {
  return new Set([...src.matchAll(/k === "([a-z])"/g)].map((m) => m[1]));
}

/** Single letters any mode's tools bind, from the registry. */
function registryLetters(): Set<string> {
  const out = new Set<string>();
  for (const m of MODES) {
    for (const g of m.groups) {
      for (const t of g.tools) out.add(t.hotkey.toLowerCase());
    }
  }
  return out;
}

function boundLetters(src: string): Set<string> {
  return new Set([...handlerLetters(src), ...registryLetters()]);
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

  it("reads both sources at all", () => {
    // If either source moves or empties, both assertions above pass vacuously.
    expect(registryLetters().size).toBeGreaterThan(5);
    expect(handlerLetters(PAGE).size).toBeGreaterThan(0);
    expect(boundLetters(PAGE).size).toBeGreaterThan(10);
  });

  it("documents the two India attribute keys", () => {
    const labels = GLOBAL.map((r) => `${r.keys} ${r.label}`).join("\n");
    expect(labels).toMatch(/^H .*helmet/m);
    expect(labels).toMatch(/^O .*occupant/m);
  });
});
