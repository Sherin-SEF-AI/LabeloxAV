import { afterEach, describe, expect, it, vi } from "vitest";

import { PREF_TOOL_TAB, prefSection, readPref, writePref } from "./panelPrefs";

// The failure this file has to survive is Safari in private mode, where localStorage exists as a property
// but THROWS on access rather than returning null. A layout preference is never worth taking the panel
// down for, so both accessors swallow, and these tests are the only thing that proves it: the happy path
// passes identically whether or not the try/catch is there.
//
// The other guarded case is a stored value that is no longer a legal option. The tool tabs are named in
// one place; a leftover value selecting a tab that no longer exists renders a panel with no tab
// highlighted and no group visible, which reads as a load failure rather than a stale preference.

function withStorage(impl: Partial<Storage>) {
  vi.stubGlobal("localStorage", impl as Storage);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("readPref / writePref", () => {
  it("round-trips a value", () => {
    const store = new Map<string, string>();
    withStorage({
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    });
    writePref(PREF_TOOL_TAB, "bulk");
    expect(readPref(PREF_TOOL_TAB, "agent")).toBe("bulk");
  });

  it("returns the fallback when nothing is stored", () => {
    withStorage({ getItem: () => null });
    expect(readPref(PREF_TOOL_TAB, "agent")).toBe("agent");
  });

  it("returns the fallback when localStorage throws instead of returning null", () => {
    withStorage({ getItem: () => { throw new DOMException("denied", "SecurityError"); } });
    expect(readPref(PREF_TOOL_TAB, "agent")).toBe("agent");
  });

  it("does not throw when a write is refused", () => {
    withStorage({ setItem: () => { throw new DOMException("quota", "QuotaExceededError"); } });
    expect(() => writePref(PREF_TOOL_TAB, "bulk")).not.toThrow();
  });

  it("survives an environment with no localStorage at all", () => {
    vi.stubGlobal("localStorage", undefined);
    expect(readPref(PREF_TOOL_TAB, "agent")).toBe("agent");
    expect(() => writePref(PREF_TOOL_TAB, "bulk")).not.toThrow();
  });

  it("rejects a stored value that is no longer a legal option", () => {
    // "tools" was the old tab name. Restoring it would select a tab that does not exist.
    withStorage({ getItem: () => "tools" });
    expect(readPref(PREF_TOOL_TAB, "agent", ["agent", "bulk", "frame"])).toBe("agent");
  });

  it("accepts a stored value that is still legal", () => {
    withStorage({ getItem: () => "frame" });
    expect(readPref(PREF_TOOL_TAB, "agent", ["agent", "bulk", "frame"])).toBe("frame");
  });
});

describe("key namespacing", () => {
  it("namespaces every key so a preference cannot collide with another feature's", () => {
    expect(PREF_TOOL_TAB.startsWith("lbx.props.")).toBe(true);
    expect(prefSection("history-and-saves")).toBe("lbx.props.section.history-and-saves");
  });
});
