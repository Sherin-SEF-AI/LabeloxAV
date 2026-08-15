// Reported from the console on /annotate/new:
//
//   :3000/annotate?_rsc=1jxsw   404 (Not Found)
//
// The breadcrumb links every path segment but the last, assuming a page exists at each level. Eleven do
// not, and `<Link>` prefetches, so every page under one of them fires an RSC request that 404s on render.
// `/annotate` is linked from 5 pages and `/review` from 3, so this was not one broken link, it was a
// steady drip of 404s describing the app's own navigation.
//
// The hand-maintained list is the small half of the problem: 69 page routes against 11 gaps. This test is
// what keeps it honest, by walking web/app and asserting the two agree in both directions. Adding
// `app/review/page.tsx` later fails this test and says to delete the entry.

import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { NON_PAGE_ANCESTORS, isNavigable } from "./routeGaps";

/** Every route that has a page.tsx, as a URL path. */
function pageRoutes(root: string): Set<string> {
  const out = new Set<string>();
  const walk = (dir: string, url: string) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) {
        walk(full, `${url}/${entry}`);
      } else if (entry === "page.tsx") {
        out.add(url === "" ? "/" : url);
      }
    }
  };
  walk(root, "");
  return out;
}

/** The ancestors a breadcrumb would link, across every real page route. */
function linkedAncestors(routes: Set<string>): Set<string> {
  const out = new Set<string>();
  for (const r of routes) {
    const segs = r.split("/").filter(Boolean);
    let acc = "";
    // The last segment is the current page and never a link, so it is excluded.
    for (const s of segs.slice(0, -1)) {
      acc += `/${s}`;
      // A dynamic segment has no fixed URL to prefetch, so it is not part of this question.
      if (!acc.includes("[")) out.add(acc);
    }
  }
  return out;
}

describe("breadcrumb ancestors", () => {
  const routes = pageRoutes(join(process.cwd(), "app"));
  const ancestors = linkedAncestors(routes);
  const gaps = new Set([...ancestors].filter((a) => !routes.has(a)));

  it("finds the app, so this cannot pass by walking nothing", () => {
    expect(routes.size).toBeGreaterThan(50);
    expect(ancestors.size).toBeGreaterThan(5);
  });

  it("lists every ancestor that has no page", () => {
    const missing = [...gaps].filter((g) => !NON_PAGE_ANCESTORS.has(g)).sort();
    expect(missing, `these breadcrumb links would 404: ${missing.join(", ")}`).toEqual([]);
  });

  it("lists no ancestor that does have a page", () => {
    // The other direction. A stale entry silently un-links a breadcrumb that works, which is a quieter bug
    // than a 404 and harder to notice.
    const stale = [...NON_PAGE_ANCESTORS].filter((g) => routes.has(g)).sort();
    expect(stale, `these have a page and should be linked: ${stale.join(", ")}`).toEqual([]);
  });

  it("marks a real route navigable", () => {
    expect(isNavigable("/analytics")).toBe(true);
    expect(isNavigable("/")).toBe(true);
  });

  it("marks the reported one not navigable", () => {
    expect(isNavigable("/annotate")).toBe(false);
  });
});
