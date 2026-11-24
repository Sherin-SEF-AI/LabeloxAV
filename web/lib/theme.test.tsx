import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { BOOT_SCRIPT, LOCALE_CODES, THEME_PREF } from "./bootScript";
import { LOCALES } from "./i18n";
import { applyTheme, prefersLight, readThemePref, resolveTheme, setThemePref, useThemePref, watchTheme } from "./theme";

type Listener = () => void;

/** A matchMedia whose answer the test controls, and whose `change` listeners it can fire. */
function fakeMedia(light: boolean) {
  const listeners = new Set<Listener>();
  const mql = {
    matches: light,
    addEventListener: (_: string, cb: Listener) => listeners.add(cb),
    removeEventListener: (_: string, cb: Listener) => listeners.delete(cb),
  };
  window.matchMedia = vi.fn(() => mql) as unknown as typeof window.matchMedia;
  return {
    set(next: boolean) { mql.matches = next; listeners.forEach((cb) => cb()); },
    listeners,
  };
}

const original = window.matchMedia;

beforeEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
  document.documentElement.lang = "en";
});
afterEach(() => { window.matchMedia = original; });

describe("resolveTheme", () => {
  it("follows the device only when asked to", () => {
    expect(resolveTheme("system", true)).toBe("light");
    expect(resolveTheme("system", false)).toBe("dark");
    expect(resolveTheme("dark", true)).toBe("dark");
    expect(resolveTheme("light", false)).toBe("light");
  });
});

describe("the stored preference", () => {
  it("is dark until somebody chooses, and ignores a value it does not know", () => {
    expect(readThemePref()).toBe("dark");
    localStorage.setItem(THEME_PREF, "sepia");
    expect(readThemePref()).toBe("dark");
    localStorage.setItem(THEME_PREF, "system");
    expect(readThemePref()).toBe("system");
    localStorage.setItem(THEME_PREF, "light");
    expect(readThemePref()).toBe("light");
  });

  it("is written, applied and announced by setThemePref", () => {
    fakeMedia(false);
    const heard = vi.fn();
    window.addEventListener("lbx:theme", heard);
    setThemePref("light");
    expect(localStorage.getItem(THEME_PREF)).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(heard).toHaveBeenCalledTimes(1);
    window.removeEventListener("lbx:theme", heard);
  });

  it("re-renders a component that reads it", () => {
    fakeMedia(false);
    const { result } = renderHook(() => useThemePref());
    expect(result.current).toBe("dark");
    act(() => setThemePref("dark"));
    expect(result.current).toBe("dark");
  });

  it("hears another tab change it", () => {
    fakeMedia(false);
    const { result } = renderHook(() => useThemePref());
    act(() => {
      localStorage.setItem(THEME_PREF, "light");
      window.dispatchEvent(new StorageEvent("storage", { key: THEME_PREF }));
    });
    expect(result.current).toBe("light");
    // A different key is not this module's business.
    act(() => {
      localStorage.setItem(THEME_PREF, "dark");
      window.dispatchEvent(new StorageEvent("storage", { key: "something.else" }));
    });
    expect(result.current).toBe("light");
  });
});

describe("applyTheme", () => {
  it("resolves system against the device", () => {
    fakeMedia(true);
    expect(prefersLight()).toBe(true);
    expect(applyTheme("system")).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    fakeMedia(false);
    expect(applyTheme("system")).toBe("dark");
  });

  it("stays dark on a light device until somebody chooses", () => {
    fakeMedia(true);
    expect(applyTheme()).toBe("dark");
  });

  it("is dark where the device cannot be asked", () => {
    // @ts-expect-error a browser without matchMedia
    window.matchMedia = undefined;
    expect(prefersLight()).toBe(false);
    expect(applyTheme()).toBe("dark");
  });
});

describe("watchTheme", () => {
  it("follows the device while the choice is system, and stops when told", () => {
    const media = fakeMedia(false);
    localStorage.setItem(THEME_PREF, "system");
    const stop = watchTheme();
    expect(document.documentElement.dataset.theme).toBe("dark");
    media.set(true);
    expect(document.documentElement.dataset.theme).toBe("light");
    stop();
    expect(media.listeners.size).toBe(0);
    media.set(false);
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("does not follow the device once a theme is chosen", () => {
    const media = fakeMedia(false);
    localStorage.setItem(THEME_PREF, "dark");
    const stop = watchTheme();
    media.set(true);
    expect(document.documentElement.dataset.theme).toBe("dark");
    stop();
  });

  it("works without matchMedia", () => {
    // @ts-expect-error a browser without matchMedia
    window.matchMedia = undefined;
    const stop = watchTheme();
    expect(document.documentElement.dataset.theme).toBe("dark");
    stop();
  });
});

describe("the boot script", () => {
  // It is a string, so nothing type-checks it. Running it is the only check there is.
  const boot = () => { new Function(BOOT_SCRIPT)(); };

  it("applies the stored theme before anything renders", () => {
    fakeMedia(false);
    localStorage.setItem(THEME_PREF, "light");
    boot();
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("is dark when nothing was chosen, whatever the device prefers", () => {
    fakeMedia(true);
    boot();
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("follows the device only when system was chosen", () => {
    localStorage.setItem(THEME_PREF, "system");
    fakeMedia(true);
    boot();
    expect(document.documentElement.dataset.theme).toBe("light");
    fakeMedia(false);
    boot();
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it("sets <html lang> from the stored language", () => {
    fakeMedia(false);
    localStorage.setItem("lbx_locale", "ta");
    boot();
    expect(document.documentElement.lang).toBe("ta");
  });

  it("leaves lang alone for a language the app does not have", () => {
    fakeMedia(false);
    localStorage.setItem("lbx_locale", "fr");
    boot();
    // jsdom's navigator.language is en-US, which the app does have.
    expect(document.documentElement.lang).toBe("en");
  });

  it("knows every locale the app ships", () => {
    // The script cannot import LOCALES (that module is a client module), so its list is its own. This is
    // what notices a fifth language.
    expect([...LOCALE_CODES].sort()).toEqual(LOCALES.map((l) => l.code).sort());
    for (const { code } of LOCALES) expect(BOOT_SCRIPT).toContain(`"${code}":1`);
  });

  it("survives storage that throws", () => {
    fakeMedia(true);
    const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
    expect(boot).not.toThrow();
    // A preference that cannot be read is the default, and the default is dark.
    expect(document.documentElement.dataset.theme).toBe("dark");
    get.mockRestore();
  });
});
