import { expect, test, type Page } from "@playwright/test";

// A new team's first hour, on a fresh install: the sign-in link the installer prints, an empty home page,
// a sample drive, auto-labelling it, accepting a label, and exporting a dataset, with the home page's
// "Get started" checklist ticking itself off as each step lands.
//
// Every other test here starts from a database that already has data in it. This one starts from nothing,
// which is the only way to know a newcomer can get from the installer to a dataset. It needs a throwaway
// install (scripts/e2e_first_run.sh makes one, runs this, and removes it) and skips without one:
//   LBX_E2E_BASE_URL   the fresh install's web address
//   LBX_E2E_SIGNIN_URL the sign-in link its installer printed
// It runs for minutes: the sample is fetched and imported, and auto-labelling waits for the GPU worker.

const SIGNIN = process.env.LBX_E2E_SIGNIN_URL;
// Random per run: the install is thrown away afterwards, and the password policy rightly refuses anything that
// reads as a phrase (an earlier fixed one containing the word "password" was refused, which is the policy working).
const PASSWORD = `lbx-${crypto.randomUUID()}`;

test.describe.configure({ mode: "serial", timeout: 30 * 60_000 });
test.skip(!SIGNIN, "set LBX_E2E_SIGNIN_URL (scripts/e2e_first_run.sh does) to run the first-run journey");

async function stepsDone(page: Page): Promise<number> {
  await page.goto("/");
  const label = page.getByText(/\d of 5 done/);
  await expect(label).toBeVisible({ timeout: 30_000 });
  return Number((await label.innerText()).split(" ")[0]);
}

async function token(page: Page): Promise<string> {
  const raw = await page.evaluate(() => window.localStorage.getItem("lbx_user"));
  const t = raw ? (JSON.parse(raw) as { token?: string }).token : undefined;
  if (!t) throw new Error("signed in, but no token in storage");
  return t;
}

async function waitFor<T>(what: string, fn: () => Promise<T | null>, timeoutMs: number): Promise<T> {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    const v = await fn();
    if (v !== null) return v;
    await new Promise((r) => setTimeout(r, 5_000));
  }
  throw new Error(`timed out waiting for ${what}`);
}

test("from the installer's sign-in link to an exported dataset", async ({ page }) => {
  await page.addInitScript(() => { try { window.localStorage.setItem("lbx_onboarded", "1"); } catch { /* none */ } });

  // 1. The link the installer printed: choose a password, which signs in.
  await page.goto(SIGNIN!);
  await page.getByLabel("new password").fill(PASSWORD);
  await page.getByRole("button", { name: "Set password and sign in" }).click();
  await page.waitForURL((u) => !u.pathname.startsWith("/login"), { timeout: 30_000 });

  // 2. An empty install: nothing done, and the first step offers a sample drive.
  expect(await stepsDone(page)).toBe(0);
  await expect(page.getByText("Bring in data")).toBeVisible();

  // 3. Load a sample drive and wait for its import to finish: the sample has a session and nothing in flight.
  const auth = { Authorization: `Bearer ${await token(page)}` };
  await page.getByRole("button", { name: "Load sample" }).first().click();
  await waitFor("the sample import to finish", async () => {
    const r = await page.request.get("/api/samples", { headers: auth });
    if (!r.ok()) return null;
    const samples = (await r.json()).samples as { session_id: string | null; job_id: string | null }[];
    return samples.some((x) => x.session_id && !x.job_id) ? true : null;
  }, 15 * 60_000);
  expect(await stepsDone(page)).toBe(1);

  // 4. Auto-label it from the home page, and wait for the GPU worker to finish the job.
  await page.getByRole("button", { name: /^Auto-label / }).click();
  await waitFor("auto-labelling to finish", async () => {
    const home = await (await page.request.get("/api/home", { headers: auth })).json();
    const failed = (home.jobs.failed as { kind: string; error: string }[]).find((f) => f.kind === "autolabel");
    if (failed) throw new Error(`auto-label failed: ${failed.error}`);
    return home.steps.autolabel && home.corpus.objects > 0 ? true : null;
  }, 20 * 60_000);
  expect(await stepsDone(page)).toBeGreaterThanOrEqual(2);

  // 5. Accept a label in My queue, the way a reviewer would.
  await page.goto("/queue");
  const firstRow = page.locator("tbody input[type=checkbox]").first();
  await expect(firstRow).toBeVisible({ timeout: 60_000 });
  await firstRow.check();
  await page.getByRole("button", { name: "approve" }).click();
  await waitFor("the accepted label to count as reviewed", async () => {
    const home = await (await page.request.get("/api/home", { headers: auth })).json();
    return home.steps.review ? true : null;
  }, 60_000);

  // 6. Export a dataset of the session, the way the Datasets page does, and wait for the commit.
  const sessions = await (await page.request.get("/api/sessions?limit=5", { headers: auth })).json();
  const r = await page.request.post("/api/datasets/export", {
    headers: auth, data: { name: "first-run", session_id: sessions[0].session_id, formats: ["coco"] } });
  expect(r.ok(), await r.text()).toBeTruthy();
  await waitFor("the export to land", async () => {
    const home = await (await page.request.get("/api/home", { headers: auth })).json();
    return home.steps.export ? true : null;
  }, 10 * 60_000);

  // Data, auto-label, review and export done; training a model is the one step left.
  expect(await stepsDone(page)).toBe(4);
  await expect(page.getByRole("listitem").filter({ hasText: "Train a model" })).toBeVisible();
});
