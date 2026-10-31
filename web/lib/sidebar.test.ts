import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { describe, expect, it } from "vitest";

import { NAV, NAV_DESTINATIONS, NAV_FOOTER, NAV_HREFS, PACK_SECTIONS, activeKey, isLabsPath, packOf, sectionsFor,
  titleFor } from "./sidebar";

const APP = join(__dirname, "..", "app");

// Settings' own pages, linked from the settings page instead of the pinned sidebar footer.
const SETTINGS_PAGES = ["/profile", "/integrations", "/activity", "/console", "/settings/keys", "/settings/team"];

// Every static route the app serves: app/**/page.tsx without a dynamic [segment] anywhere in its path.
function staticRoutes(dir = APP): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      if (name.startsWith("[") || name.startsWith("(") || name.startsWith("_")) continue;
      out.push(...staticRoutes(full));
    } else if (name === "page.tsx") {
      const rel = relative(APP, dir).split(sep).join("/");
      out.push(rel ? `/${rel}` : "/");
    }
  }
  return out;
}

describe("sidebar navigation", () => {
  it("every entry points at a page that exists", () => {
    for (const d of NAV_DESTINATIONS) {
      const path = d.href.split("?")[0];
      const file = path === "/" ? join(APP, "page.tsx") : join(APP, ...path.slice(1).split("/"), "page.tsx");
      expect(existsSync(file), `${d.label} -> ${d.href} has no page`).toBe(true);
    }
  });

  it("keys are unique, so React and the active state can tell entries apart", () => {
    const keys = NAV_DESTINATIONS.map((d) => d.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("every entry says what it is for", () => {
    for (const d of NAV_DESTINATIONS) expect(d.hint.length, d.label).toBeGreaterThan(10);
  });

  // The point of the rework: a new person reads what a page does, not an invented product name.
  it("labels are plain words, not internal code names", () => {
    for (const d of NAV_DESTINATIONS) {
      expect(d.label, d.label).not.toMatch(/yx\b|^[A-Z]{4,}$/);
    }
  });

  it("only the research tools are under Labs", () => {
    const labs = NAV.filter((s) => s.labs).flatMap((s) => s.items.map((i) => i.href));
    for (const daily of ["/", "/review/queue", "/annotations", "/annotate/new", "/import", "/datasets", "/training"]) {
      expect(labs).not.toContain(daily);
    }
  });

  // Removing the menu bar and the platform switcher must not orphan a page: every static page is in the
  // sidebar (as an entry or a sub-page). Parent pages do not link to their own sub-pages, so "reachable from
  // the parent" is not a substitute, and the command palette is a shortcut, not a route anyone discovers.
  it("every page is in the sidebar", () => {
    const inSidebar = new Set(NAV_HREFS.map((h) => h.split("?")[0]));
    // Sign-in; the old platform launcher, kept so links into it still land; and the security pack, which is
    // a separate optional package with its own entry when it is installed.
    const exempt = new Set(["/login", "/platforms", ...SETTINGS_PAGES]);
    const missing = staticRoutes().filter((r) => !inSidebar.has(r) && !exempt.has(r));
    expect(missing).toEqual([]);
  });

  it("the settings page links every settings page the sidebar leaves out", () => {
    const src = readFileSync(join(APP, "settings", "page.tsx"), "utf8");
    for (const href of SETTINGS_PAGES) expect(src, href).toContain(`href: "${href}"`);
  });

  it("sub-pages point at pages that exist", () => {
    for (const d of NAV_DESTINATIONS) {
      for (const c of d.children ?? []) {
        const file = join(APP, ...c.href.slice(1).split("/"), "page.tsx");
        expect(existsSync(file), `${d.label} > ${c.label} -> ${c.href}`).toBe(true);
      }
    }
  });
});

describe("activeKey", () => {
  it.each([
    ["/", "home"],
    ["/review/queue", "review"],
    ["/review/rapid", "review"],
    ["/annotations", "sessions"],
    ["/annotate/new", "upload"],
    ["/annotate/asset/abc", "sessions"],
    ["/sanyx/predictive", "ingest-health"],
    ["/sievyx/longtail", "curation"],
    ["/calibration", "calibration"],
    ["/govern/compliance", "govern"],
    ["/settings", "settings"],
    ["/integrations", "settings"],
  ])("%s is %s", (path, key) => {
    expect(activeKey(path)).toBe(key);
  });

  it("the home entry does not claim every page", () => {
    expect(activeKey("/frame/123")).toBeNull();
  });
});

describe("Labs visibility", () => {
  it("a Labs page shows its section even with Labs switched off", () => {
    expect(isLabsPath("/calyx/recovery")).toBe(true);
    expect(isLabsPath("/agent")).toBe(true);
  });

  it("the everyday pages are not Labs", () => {
    expect(isLabsPath("/training")).toBe(false);
    expect(isLabsPath("/")).toBe(false);
  });

  it("settings sits in the footer, not in a section", () => {
    expect(NAV_FOOTER.map((f) => f.href)).toContain("/settings");
    expect(NAV.flatMap((s) => s.items).map((i) => i.href)).not.toContain("/settings");
  });
});

describe("titleFor", () => {
  it("names a page the way the sidebar does", () => {
    expect(titleFor("/verdyx")).toEqual({ title: "Slice matrix", parent: "Slice evaluation" });
    expect(titleFor("/lidar/annotate")).toEqual({ title: "3D cuboids", parent: "LiDAR" });
    expect(titleFor("/training")).toEqual({ title: "Training" });
    expect(titleFor("/settings")).toEqual({ title: "Settings" });
  });

  it("a sub-page named like its entry is not repeated", () => {
    expect(titleFor("/datasets")).toEqual({ title: "Datasets" });
  });

  it("a page outside the sidebar keeps its own title", () => {
    expect(titleFor("/object/abc")).toBeNull();
  });
});

describe("optional packs", () => {
  it("the security section appears only when the security pack is installed", () => {
    expect(sectionsFor(() => false).map((x) => x.key)).not.toContain("security");
    expect(sectionsFor((p) => p === "sec").map((x) => x.key)).toContain("security");
  });

  it("security pages belong to the security pack and product pages to none", () => {
    expect(packOf("/labeloxsec")).toBe("sec");
    expect(packOf("/labeloxsec/incidents")).toBe("sec");
    expect(packOf("/labeloxsec?hits=1")).toBe("sec");
    expect(packOf("/training")).toBeNull();
  });

  it("every optional section names its pack", () => {
    for (const s of PACK_SECTIONS) expect(s.pack, s.key).toBeTruthy();
  });
});
