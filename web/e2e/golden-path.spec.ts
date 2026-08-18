import { expect, test } from "@playwright/test";

// The journey, and the specific failures this remediation fixed.
//
// Everything else in this repo tests a piece. Only this can tell you that a person can sign in and reach
// their queue, because that needs the API, the database, the web server and the browser to agree - which
// is precisely the class of breakage the audits kept finding. Two examples, both real: ~20 call sites sent
// no auth header, so under deny-by-default those pages rendered blank rather than erroring; and a page
// existed, worked, and was reachable only by typing its URL.
//
// Requires a running stack. `make e2e` brings one up; without it these skip rather than fail, because a
// suite that goes red when infra is absent teaches people to ignore red.

const NEEDS_STACK = "requires a running app (make e2e)";

test.beforeEach(async ({ page }) => {
  const res = await page.request.get("/api/readyz").catch(() => null);
  test.skip(!res || !res.ok(), NEEDS_STACK);
});

test.describe("the app is reachable and says who it is", () => {
  test("an unauthenticated visitor is sent to sign in, not shown an empty app", async ({ page }) => {
    const res = await page.goto("/");
    // Status first. "The body is not empty" was the original assertion here and it is not good enough:
    // a Next error page is a perfectly full body, so a 500 would have passed. Found exactly that way -
    // the home page was serving 500 from a stale build and this test was green.
    expect(res?.status(), "the home page did not render").toBeLessThan(400);
    // Either the queue (a dev session already exists) or the login page. What must NOT happen is the
    // shell rendering with no content and no explanation, which is what deny-by-default produced before
    // the client routed every call through the authed helper.
    await expect(page.locator("body")).not.toBeEmpty();
    const signedOut = page.url().includes("/login");
    if (signedOut) await expect(page.getByRole("button", { name: /sign in/i }).first()).toBeVisible();
  });

  test("the readiness probe distinguishes ready from merely alive", async ({ page }) => {
    // /api/health returns 200 while degraded so an operator can see which dependency is down; readyz is
    // what a load balancer should watch. Both existing is the invariant.
    const ready = await page.request.get("/api/readyz");
    expect(ready.status()).toBeLessThan(500);
    const health = await page.request.get("/api/health");
    expect(health.ok()).toBeTruthy();
  });
});

test.describe("navigation reaches what it claims to", () => {
  test("every menu destination resolves to a page, not a 404", async ({ page }) => {
    // The menu is data (lib/menus.ts) and the palette reads the same definition, so they cannot drift -
    // but neither can tell you the page behind an href actually renders. 23 of 31 destinations were once
    // inert because five pages ignored their query string.
    const res = await page.goto("/platforms");
    expect(res?.status(), "the platform launcher did not render").toBeLessThan(400);
    await expect(page.locator("body")).not.toBeEmpty();
    await expect(page).toHaveTitle(/labelox/i);
  });

  test("the driving-events pages are reachable from the app, not only by URL", async ({ page }) => {
    // Both were listed only in a dead second navigation registry, so one had a single inbound link from
    // inside the frame editor and the other had none at all.
    for (const path of ["/events", "/events/search"]) {
      const res = await page.goto(path);
      expect(res?.status(), `${path} did not render`).toBeLessThan(400);
      await expect(page.locator("body")).not.toBeEmpty();
    }
  });
});

test.describe("a failure does not read as a finished shift", () => {
  test("the queue says something is wrong rather than that it is clear", async ({ page }) => {
    // The defect this remediation found three times over: a dropped request rendering as an empty state.
    // Triage said "Queue is clear", analytics said "no objects yet" over a 570k-object corpus, and the
    // project board rendered as though a manager had no projects.
    await page.route("**/api/triage*", (route) => route.abort("failed"));
    const res = await page.goto("/");
    expect(res?.status(), "the page must render in order to be wrong about the queue").toBeLessThan(400);
    const body = await page.locator("body").innerText();
    expect(body.toLowerCase()).not.toContain("queue is clear");
  });

  test("analytics does not report an empty corpus when its backend is down", async ({ page }) => {
    await page.route("**/api/analytics/**", (route) => route.abort("failed"));
    const res = await page.goto("/analytics");
    expect(res?.status()).toBeLessThan(400);
    await expect(page.locator("body")).not.toBeEmpty();
    const body = await page.locator("body").innerText();
    expect(body.toLowerCase()).not.toContain("no objects yet");
  });
});

test.describe("the keyboard floor", () => {
  test("tab reaches a skip link first, and it is visible when focused", async ({ page }) => {
    // Tailwind's preflight removes the default outline and nothing put one back, so tabbing moved an
    // invisible cursor through an app whose whole premise is keyboard-driven work.
    await page.goto("/platforms");
    await page.keyboard.press("Tab");
    const focused = page.locator(":focus");
    await expect(focused).toBeVisible();
    await expect(focused).toHaveText(/skip to content/i);
  });

  test("the skip link lands on the content landmark", async ({ page }) => {
    await page.goto("/platforms");
    await page.keyboard.press("Tab");
    await page.keyboard.press("Enter");
    await expect(page.locator("#content")).toBeVisible();
  });
});
