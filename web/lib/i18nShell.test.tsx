import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

import { act, render, renderHook, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import {
  LOCALES, T, UNTRANSLATED_LABELS, dictFor, getLocale, labelKey, translate, translateLabel, useLabel,
  useLocale, useT,
} from "./i18n";
import { EN } from "./locales/en";
import { MENUS, type MenuItem } from "./menus";
import { NAV, NAV_FOOTER, PACK_SECTIONS } from "./sidebar";

// The navigation is data with English labels, and the label is the translation key (labelKey). So the
// failure this guards is quiet: somebody adds a sidebar entry, it renders, every test passes, and a Tamil
// sidebar has one row in English that nobody who reviews the change can see.

/** Every label the sidebar can show: sections, entries, sub-pages, the footer and optional packs. */
function sidebarLabels(): string[] {
  const out: string[] = [];
  for (const section of [...NAV, ...PACK_SECTIONS]) {
    out.push(section.label);
    for (const item of section.items) out.push(item.label, ...(item.children ?? []).map((c) => c.label));
  }
  for (const item of NAV_FOOTER) out.push(item.label, ...(item.children ?? []).map((c) => c.label));
  return out;
}

/**
 * Every label the menus can show. The import and export submenus are left out on purpose: their entries
 * are format names ("COCO", "nuScenes (3D)", "Pascal VOC"), which are data, and data is never translated.
 */
function menuLabels(): string[] {
  const out: string[] = [];
  const isFormat = (it: MenuItem) => /^(import|export)-/.test(it.key);
  const walk = (items: MenuItem[]) => {
    for (const it of items) {
      if (isFormat(it)) continue;
      out.push(it.label);
      if (it.items) walk(it.items);
    }
  };
  for (const menu of MENUS) { out.push(menu.label); walk(menu.items); }
  return out;
}

/** Titles passed as string literals to <PageShell title="...">, across every page. */
function pageTitles(): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) walk(full);
      else if (entry === "page.tsx") {
        for (const m of readFileSync(full, "utf8").matchAll(/<PageShell\b[^>]*?\btitle="([^"]+)"/gs)) out.push(m[1]);
      }
    }
  };
  walk(join(process.cwd(), "app"));
  return out;
}

const missing = (labels: string[]) =>
  [...new Set(labels)].filter((l) => !UNTRANSLATED_LABELS.has(l) && !EN[labelKey(l)]);

describe("navigation labels have translation keys", () => {
  it("finds the navigation, so this cannot pass by checking nothing", () => {
    expect(sidebarLabels().length).toBeGreaterThan(60);
    expect(menuLabels().length).toBeGreaterThan(50);
  });

  it("every sidebar label has a key", () => {
    expect(missing(sidebarLabels()), "add these to lib/locales/*.ts under shell.<slug>").toEqual([]);
  });

  it("every menu label has a key", () => {
    expect(missing(menuLabels()), "add these to lib/locales/*.ts under shell.<slug>").toEqual([]);
  });

  it("the key's English is the label, whatever its capitalisation", () => {
    // labelKey is lossy ("Review queue" and "Review Queue" share a key), which is fine for a translation
    // and would be a bug if two different labels collided.
    for (const label of [...sidebarLabels(), ...menuLabels()]) {
      if (UNTRANSLATED_LABELS.has(label)) continue;
      expect(EN[labelKey(label)].toLowerCase(), label).toBe(label.toLowerCase());
    }
  });

  it("every one of those keys is translated in every other language", () => {
    for (const { code } of LOCALES) {
      if (code === "en") continue;
      const dict = dictFor(code);
      const untranslated = [...new Set([...sidebarLabels(), ...menuLabels()])]
        .filter((l) => !UNTRANSLATED_LABELS.has(l))
        .filter((l) => !dict[labelKey(l)] || dict[labelKey(l)] === l);
      expect(untranslated, `${code} shows these in English`).toEqual([]);
    }
  });

  it("a name kept in English has no key", () => {
    // Otherwise the exemption and a translation would both exist and the exemption would silently win.
    for (const label of UNTRANSLATED_LABELS) expect(EN[labelKey(label)], label).toBeUndefined();
  });

  it("the page titles a header can show have keys", () => {
    // A page in the sidebar is titled by its sidebar entry (PageShell), so only the pages outside it show
    // the title they pass. Those are the ones that need a key of their own.
    const shown = ["Activity", "Console", "Integrations", "Platforms", "Your account", "About", "API keys",
                   "Projects", "Team"];
    const passed = new Set(pageTitles());
    expect(passed.size).toBeGreaterThan(40);
    for (const title of shown) expect(passed.has(title), `${title} is no longer a page title`).toBe(true);
    expect(missing(shown)).toEqual([]);
  });
});

describe("translateLabel", () => {
  it("returns English labels as written, capitalisation included", () => {
    expect(translateLabel("en", "Session Inspector")).toBe("Session Inspector");
  });

  it("translates a label that has a key", () => {
    expect(translateLabel("hi", "My queue")).toBe(dictFor("hi")["shell.my_queue"]);
    expect(translateLabel("ta", "Jobs")).not.toBe("Jobs");
  });

  it("leaves product names and data alone", () => {
    expect(translateLabel("kn", "LiDAR")).toBe("LiDAR");
    expect(translateLabel("kn", "DASHCAM-01 2026-07-03")).toBe("DASHCAM-01 2026-07-03");
  });
});

describe("translate", () => {
  it("falls back to English, then the fallback, then the key", () => {
    expect(translate("hi", "common.cancel")).toBe(dictFor("hi")["common.cancel"]);
    expect(translate("hi", "no.such.key", "plain")).toBe("plain");
    expect(translate("hi", "no.such.key")).toBe("no.such.key");
    // A locale code that is not one of ours (a stale value in storage) reads as English.
    expect(translate("xx" as "en", "common.cancel")).toBe("Cancel");
  });

  it("interpolates in the translated word order", () => {
    expect(translate("hi", "home.setup.progress", undefined, { done: 1, total: 5 })).toContain("5");
    expect(translate("en", "pager.range", undefined, { from: 1, to: 50, total: 2070 })).toBe("1-50 of 2070");
  });
});

describe("reading the locale during render", () => {
  afterEach(() => {
    localStorage.clear();
    // Drop the module's cached locale the way another tab changing it would.
    act(() => { window.dispatchEvent(new StorageEvent("storage", { key: "lbx_locale" })); });
  });

  /** What a second tab does: write the key, and the storage event tells this one. */
  const switchTo = (code: string) => act(() => {
    localStorage.setItem("lbx_locale", code);
    window.dispatchEvent(new StorageEvent("storage", { key: "lbx_locale" }));
  });

  it("useLocale follows a change made in another tab", () => {
    const { result } = renderHook(() => useLocale());
    expect(result.current).toBe("en");
    switchTo("ta");
    expect(result.current).toBe("ta");
    // A different key is not a language change.
    act(() => { window.dispatchEvent(new StorageEvent("storage", { key: "lbx.theme" })); });
    expect(result.current).toBe("ta");
  });

  it("useT, useLabel and <T> render in the stored language", () => {
    function Probe() {
      const t = useT();
      const label = useLabel();
      return (
        <>
          <span data-testid="t">{t("common.cancel", "Cancel")}</span>
          <span data-testid="label">{label("Settings")}</span>
          <span data-testid="el"><T k="common.retry">Retry</T></span>
          <span data-testid="vars"><T k="load.route" vars={{ what: "LiDAR" }}>{"loading {what}"}</T></span>
        </>
      );
    }
    render(<Probe />);
    expect(screen.getByTestId("t")).toHaveTextContent("Cancel");
    expect(screen.getByTestId("label")).toHaveTextContent("Settings");
    expect(screen.getByTestId("el")).toHaveTextContent("Retry");
    expect(screen.getByTestId("vars")).toHaveTextContent("loading LiDAR");

    switchTo("kn");
    const kn = dictFor("kn");
    expect(screen.getByTestId("t")).toHaveTextContent(kn["common.cancel"]);
    expect(screen.getByTestId("label")).toHaveTextContent(kn["shell.settings"]);
    expect(screen.getByTestId("el")).toHaveTextContent(kn["common.retry"]);
    expect(screen.getByTestId("vars")).toHaveTextContent("LiDAR");
  });

  it("ignores a stored value that is not a language and falls back to the browser's", () => {
    switchTo("klingon");
    expect(getLocale()).toBe("en");
  });

  it("reads as the browser's language when storage throws", () => {
    act(() => { window.dispatchEvent(new StorageEvent("storage", { key: "lbx_locale" })); });
    const original = Storage.prototype.getItem;
    Storage.prototype.getItem = () => { throw new Error("blocked"); };
    try { expect(getLocale()).toBe("en"); } finally { Storage.prototype.getItem = original; }
  });
});
