import { defineConfig, devices } from "@playwright/test";

// End-to-end smoke over the golden path.
//
// There was none. Playwright was installed in the Python venv and invoked by nothing, so 71 pages and the
// whole sign-in → work → review journey had no automated coverage at all. Every other tier here tests a
// piece: the unit tier tests pure logic, the jsdom tier renders one component, the Python suite exercises
// services and the route table. None of them can tell you that a person can sign in and reach their queue,
// because that requires the API, the database, the web server and the browser to agree - which is exactly
// the class of breakage the audits kept finding (a page that rendered blank because a fetch sent no auth
// header, a route reachable only by typing its URL).
//
// Deliberately a smoke, not a regression suite. It asserts the journey is walkable and that the specific
// failures this remediation fixed stay fixed. A broad e2e suite over a UI this large would be slow and
// flaky, and a flaky gate gets switched off.
export default defineConfig({
  testDir: "./e2e",
  // The app is under test, not the network: a retry would hide exactly the flakiness worth knowing about.
  retries: 0,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 10_000 },
  reporter: process.env.CI ? "list" : "line",
  use: {
    baseURL: process.env.LBX_E2E_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // Started by the caller (make e2e) rather than here, because the app needs a migrated database and a
  // running API, and a webServer block that only starts Next would produce a suite that fails on the API
  // being absent while looking like a frontend failure.
});
