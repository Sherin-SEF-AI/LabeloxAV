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

  it("item keys are unique within each menu", () => {
    for (const m of MENUS) {
      const keys = m.items.map((i) => i.key);
      expect(new Set(keys).size, `duplicate item key in menu ${m.key}`).toBe(keys.length);
    }
  });
});
