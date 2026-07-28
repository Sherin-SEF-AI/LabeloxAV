import { describe, expect, it } from "vitest";

import { MENUS, MENU_DESTINATIONS } from "./menus";

describe("navigation menus", () => {
  it("every menu has a key, a label, and at least one item", () => {
    expect(MENUS.length).toBeGreaterThan(0);
    for (const m of MENUS) {
      expect(m.key).toBeTruthy();
      expect(m.label).toBeTruthy();
      expect(m.items.length).toBeGreaterThan(0);
    }
  });

  it("every destination has an app-relative href", () => {
    expect(MENU_DESTINATIONS.length).toBeGreaterThan(0);
    for (const d of MENU_DESTINATIONS) {
      expect(d.href.startsWith("/")).toBe(true);
      expect(d.label).toBeTruthy();
    }
  });

  // Note: the same href legitimately appears in several menus (a page reachable from more than one menu), so
  // hrefs are deliberately not unique. Keys, however, must be: a duplicate key is a real React reconciliation
  // bug. Menu keys are unique across the bar, and item keys are unique within each menu.
  it("menu keys are unique", () => {
    const keys = MENUS.map((m) => m.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  // The command palette renders one row per destination keyed by href+label. Since hrefs repeat by design,
  // that pair is what must be unique or React reconciliation breaks on duplicate keys.
  it("href and label together are unique across destinations", () => {
    const pairs = MENU_DESTINATIONS.map((d) => `${d.href}::${d.label}`);
    expect(new Set(pairs).size).toBe(pairs.length);
  });

  // Deep-linked menu entries carry a query string the destination page must read. These went inert once
  // (all fourteen import-format entries landed on the default), so the contract is pinned here: any
  // destination with a query string is listed, and lib/useQueryParam is how a page honours it.
  it("query-string destinations are the known deep links", () => {
    const withQuery = MENU_DESTINATIONS.filter((d) => d.href.includes("?"));
    const params = new Set(withQuery.map((d) => new URL(d.href, "http://x").searchParams.keys().next().value));
    // hits: the LabeloxSec console opens with its watchlist-hit filter already applied.
    expect([...params].sort()).toEqual(["format", "hits", "mine", "panel", "rig", "tab"]);
  });

  it("item keys are unique within each menu", () => {
    for (const m of MENUS) {
      const keys = m.items.map((i) => i.key);
      expect(new Set(keys).size, `duplicate item key in menu ${m.key}`).toBe(keys.length);
    }
  });
});
