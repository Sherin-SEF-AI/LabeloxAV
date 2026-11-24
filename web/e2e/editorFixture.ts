import fs from "node:fs";

import { type Browser, type BrowserContext, type Page, test } from "@playwright/test";

// What scripts/e2e_seed.py wrote for this run: ten frames in a throwaway database and one user per role.
//
// These specs need exactly that state (known frames, known objects, three identities), which is why they
// do not run against whatever stack happens to be up: `make e2e-editor` builds the database, starts the
// app and tears it down. Without the fixture they skip rather than fail, for the reason the other specs
// here do: a suite that goes red when its infrastructure is absent teaches people to ignore red.

export type Role = "admin" | "reviewer" | "annotator";
export type Fixture = {
  session_id: string;
  frames: string[];
  users: Record<Role, { user_id: string; name: string; role: Role; token: string }>;
};
export type ApiObject = { object_id: string; bbox: number[]; state: string; track_id: string | null };

const FIXTURE_PATH = process.env.LBX_E2E_FIXTURE;
export const API = process.env.LBX_E2E_API_URL ?? "http://127.0.0.1:8000";

export function loadFixture(): Fixture | null {
  if (!FIXTURE_PATH || !fs.existsSync(FIXTURE_PATH)) return null;
  return JSON.parse(fs.readFileSync(FIXTURE_PATH, "utf8")) as Fixture;
}

export function requireFixture(): Fixture {
  const fx = loadFixture();
  test.skip(!fx, "requires the seeded editor stack (make e2e-editor)");
  return fx as Fixture;
}

/** A signed-in browser context for one of the seeded users, with the onboarding tour already seen. */
export async function signedIn(browser: Browser, fx: Fixture, role: Role): Promise<BrowserContext> {
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 950 } });
  await ctx.addInitScript((u) => {
    window.localStorage.setItem("lbx_user", JSON.stringify(u));
    window.localStorage.setItem("lbx_onboarded", new Date().toISOString());
  }, fx.users[role]);
  return ctx;
}

/** Open a frame in the editor and wait until its canvas is up. Not "networkidle": the editor polls. */
export async function openFrame(ctx: BrowserContext, frameId: string): Promise<Page> {
  const page = await ctx.newPage();
  await page.goto(`/frame/${frameId}`, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("canvas", { timeout: 60_000 });
  // The objects arrive after the canvas mounts; the count in the header is the sign they have.
  await page.waitForTimeout(1500);
  return page;
}

/** The frame's objects as the server has them: the only honest way to tell whether an edit was saved. */
export async function serverObjects(fx: Fixture, frameId: string, role: Role = "admin"): Promise<ApiObject[]> {
  const r = await fetch(`${API}/api/frames/${frameId}/objects`, {
    headers: { Authorization: `Bearer ${fx.users[role].token}` },
  });
  if (!r.ok) throw new Error(`GET objects ${r.status}`);
  return ((await r.json()) as ApiObject[]).filter((o) => o.state !== "rejected");
}

export const signature = (objs: ApiObject[]) =>
  objs.map((o) => `${o.object_id}:${o.bbox.map((v) => Math.round(v * 10) / 10).join(",")}`).sort().join("|");

/** Autosave fires 700 ms after the last change; this is comfortably past it and the request. */
export const AUTOSAVE_MS = 2500;
