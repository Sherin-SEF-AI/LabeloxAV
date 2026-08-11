// Every API path in the shared client has to start with /api, and two of them did not.
//
// `opPrecisionLatest` and `opPrecisionAll` were written as `/eval/operations/...`, so they resolved against
// the Next.js page router and 404ed there without ever reaching the backend. The console showed:
//
//   :3000/eval/operations/cuboid/latest   404 (Not Found)
//   :3000/eval/operations/attribute/latest 404 (Not Found)
//   :3000/eval/operations/relabel/latest   404 (Not Found)
//
// What makes this worth a guard rather than a one-line fix is that the client could not tell. A 404 on that
// endpoint is a documented state meaning "this operation has no measurement yet", and the client responds by
// making the dry run mandatory and routing everything to review. So a route that had never existed produced
// exactly the same behaviour as a measurement nobody had taken, indefinitely, and looked correct while doing
// it. A missing prefix is not a typo here; it is a silent downgrade of a product decision.
//
// This is the same shape of guard as nofetch.test.ts: a rule about the shared client that is cheap to state
// and expensive to notice being broken.

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

// Paths the client navigates to rather than fetches. They are page routes and must not carry /api.
const NAVIGATION = new Set(["/", "/login", "/login?next="]);

function pathLiterals(src: string): string[] {
  // Every single-slash-rooted path in a string or template literal, stopping before an interpolation.
  return [...new Set(src.match(/[`"](\/(?!\/)[^`"$\s)]*)/g) ?? [])].map((m) => m.slice(1));
}

describe("api client paths", () => {
  const src = readFileSync(join(process.cwd(), "lib/api.ts"), "utf8");

  it("routes every backend call through /api", () => {
    const stray = pathLiterals(src)
      .filter((p) => !p.startsWith("/api"))
      .filter((p) => !NAVIGATION.has(p));
    expect(stray, `these resolve against the page router, not the backend: ${stray.join(", ")}`).toEqual([]);
  });

  it("finds the paths at all, so the check cannot pass by matching nothing", () => {
    // A regex that quietly stops matching would make this file a green test that checks nothing, which is
    // worse than not having it.
    expect(pathLiterals(src).length).toBeGreaterThan(200);
  });

  it("still allows the page routes the client navigates to", () => {
    expect(pathLiterals(src)).toContain("/login");
  });
});
