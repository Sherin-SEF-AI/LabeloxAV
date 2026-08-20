import { readdirSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

// `.catch(() => {})` turns a failed request into a state indistinguishable from having no data. That is
// the defect that let the triage page render "Queue is clear" when a fetch dropped, and the analytics page
// render "no objects yet" over a 570k-object corpus - both of which read as a finished shift rather than a
// broken one.
//
// This does not demand that every failure be surfaced. Some genuinely are optional: a panel whose endpoint
// may legitimately 404, a best-effort prefetch. What it demands is that the number stop growing on its own
// - a new page cannot quietly add another, and clearing one removes it from the list.
//
// Shaped after lib/nofetch.test.ts, which guards the same class of problem for the auth header.

const WEB_ROOT = resolve(__dirname, "..");
const SCAN_DIRS = ["app", "components", "lib", "platforms"];

// The pattern, in the two spellings that appear in this tree.
const SWALLOW = /\.catch\(\(\)\s*=>\s*\{\s*\}\)/g;

// The baseline as it stands, NOT a per-file endorsement. Being honest about that matters: these counts
// were measured, not individually adjudicated, and some of them are probably wrong in the same way the
// triage page was wrong. What this test buys is that the number cannot grow without somebody choosing to
// raise it, and that a page whose swallow is fixed drops out of the list rather than lingering as a
// permission nobody uses.
//
// Each entry is a count rather than a filename, so adding a second swallow to an already-listed file
// still fails.
const ALLOWED: Record<string, number> = {
  // Optional analytics panels. The page has a real error state for its five REQUIRED calls; these five are
  // supplementary charts that render their own "no data" and whose absence is not the page failing.
  "app/analytics/page.tsx": 5,
  // Secondary reads behind an already-guarded primary: the linked view reports its own load failure.
  "app/lidar/linked/page.tsx": 4,
  // Best-effort side panels; the board's own loader reports failures.
  "app/forgyx/page.tsx": 3,
  "app/agent/page.tsx": 2,
  "app/explore/page.tsx": 2,
  "app/oraclyx/page.tsx": 2,
  "app/projects/page.tsx": 0,   // fixed: the project list is this page's primary data
  "app/import/page.tsx": 1,
  "app/lidar/annotate/page.tsx": 1,
  "app/sievyx/page.tsx": 1,
  "app/training/page.tsx": 1,
  "app/verdyx/page.tsx": 1,
  "components/UserPicker.tsx": 1,
  "components/training/RunsPanel.tsx": 1,
  // The shared client's own best-effort paths (token refresh, the 401 announce).
  "lib/api.ts": 1,
  "components/explore/EvalDrilldown.tsx": 2,
  "components/console/Panels.tsx": 2,
  "components/agent/AgentPanel.tsx": 2,
  "components/CorrectionModal.tsx": 2,
  "app/frame/[id]/page.tsx": 2,
  "app/page.tsx": 3,
  "app/annotate/lane/[frameId]/page.tsx": 1,
  "app/flywheel/adaptive/page.tsx": 1,
  "app/ops/page.tsx": 1,
  "app/review/rapid/page.tsx": 1,
  "app/sanyx/predictive/page.tsx": 1,
};

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry === ".next") continue;
    const full = `${dir}/${entry}`;
    if (statSync(full).isDirectory()) walk(full, out);
    // Test files are skipped: several of them contain this pattern as fixture data or, in the case of the
    // source scanners, as the literal thing they detect.
    else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) out.push(full);
  }
  return out;
}

function counts(): Map<string, number> {
  const found = new Map<string, number>();
  for (const dir of SCAN_DIRS) {
    let files: string[];
    try {
      files = walk(resolve(WEB_ROOT, dir));
    } catch {
      continue;
    }
    for (const file of files) {
      const rel = file.slice(WEB_ROOT.length + 1);
      const n = (readFileSync(file, "utf8").match(SWALLOW) ?? []).length;
      if (n > 0) found.set(rel, n);
    }
  }
  return found;
}

describe("swallowed failures are a decision, not a habit", () => {
  it("scans a non-trivial number of files", () => {
    // Without this, a broken glob turns every assertion below into a statement about the empty set.
    let n = 0;
    for (const dir of SCAN_DIRS) {
      try {
        n += walk(resolve(WEB_ROOT, dir)).length;
      } catch {
        /* directory moved */
      }
    }
    expect(n, "the source scan found almost nothing; SCAN_DIRS has drifted").toBeGreaterThan(150);
  });

  it("no file swallows more failures than it is allowed", () => {
    const offenders: string[] = [];
    for (const [file, n] of counts()) {
      const allowed = ALLOWED[file] ?? 0;
      if (n > allowed) offenders.push(`${file}: ${n} swallowed, ${allowed} allowed`);
    }
    expect(
      offenders,
      "a swallowed failure renders as an empty state, which is what a finished shift looks like. " +
        "Surface it, or add the file to ALLOWED with a reason.",
    ).toEqual([]);
  });

  it("the allowlist has not gone stale", () => {
    // An exemption for a file that no longer swallows anything is a permission nobody is using, and the
    // next reader has to work out whether it still matters.
    const live = counts();
    const stale = Object.entries(ALLOWED)
      .filter(([f, n]) => n > 0 && (live.get(f) ?? 0) === 0)
      .map(([f]) => f);
    expect(stale, "these files no longer swallow anything and can leave ALLOWED").toEqual([]);
  });
});
