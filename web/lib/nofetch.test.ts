import { readFileSync, readdirSync, statSync } from "fs";
import { join, resolve } from "path";

import { describe, expect, it } from "vitest";

// Guard against the bug class where a page hand-rolls fetch("/api/...") instead of going through the shared
// client. A bare fetch sends no Authorization header, so under the deny-by-default read gate it 401s, and
// because these call sites typically end in .catch(() => {}) the page renders silently blank instead of
// redirecting to sign-in. Every data call must use apiGet/apiPost/api.* from lib/api.ts.
//
// The allowlist is the credential-bootstrap paths, which by definition cannot carry a token yet.
const ALLOWED = [
  "lib/api.ts", // the shared client itself
  "lib/user.ts", // POST /auth/refresh, sent with the expiring token explicitly
  "app/login/page.tsx", // dev-login and the paste-a-token verification
  "components/AuthBootstrap.tsx", // first-token acquisition on a fresh tab
  "lib/nofetch.test.ts", // this guard: its own detection pattern is a literal match
];

const WEB_ROOT = resolve(__dirname, "..");
const SCAN_DIRS = ["app", "lib", "components", "platforms"];

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry === ".next") continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (/\.(ts|tsx)$/.test(full)) out.push(full);
  }
  return out;
}

describe("no unauthenticated API fetches", () => {
  it("every /api call goes through the shared authed client", () => {
    const offenders: string[] = [];
    for (const dir of SCAN_DIRS) {
      let files: string[];
      try {
        files = walk(join(WEB_ROOT, dir));
      } catch {
        continue; // directory may not exist
      }
      for (const file of files) {
        const rel = file.slice(WEB_ROOT.length + 1);
        if (ALLOWED.includes(rel)) continue;
        const src = readFileSync(file, "utf8");
        // a raw fetch whose URL literal starts with /api
        if (/fetch\(\s*[`"']\/api/.test(src)) offenders.push(rel);
      }
    }
    expect(offenders, `raw fetch("/api/...") found; use apiGet/apiPost from lib/api.ts`).toEqual([]);
  });
});
