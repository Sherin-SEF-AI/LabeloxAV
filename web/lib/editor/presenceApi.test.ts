import { beforeEach, describe, expect, it, vi } from "vitest";

// The wire shape of presence, and the one request that is made while the page is closing. Lives beside
// the hook that uses it; the module under test is lib/presenceApi.ts.

const api = vi.hoisted(() => ({
  apiPost: vi.fn(async (_path: string, _body: unknown) => ({})),
  getBinary: vi.fn(async (_path: string, _init?: RequestInit) => new Response(null)),
}));
vi.mock("@/lib/api", () => api);

import { presenceApi } from "@/lib/presenceApi";

const TAB = "tab-0123456789abcdef";

beforeEach(() => { api.apiPost.mockClear(); api.getBinary.mockClear(); });

describe("presenceApi", () => {
  it("sends only the tab token: who the caller is comes from their credential", async () => {
    await presenceApi.heartbeat("f1", TAB);
    await presenceApi.heartbeat("f1", TAB, false);
    await presenceApi.acquire("f1", TAB);
    await presenceApi.release("f1", TAB);
    await presenceApi.takeOver("f1", TAB);
    expect(api.apiPost.mock.calls).toEqual([
      ["/api/presence/frames/f1/heartbeat", { tab: TAB, acquire: true }],
      ["/api/presence/frames/f1/heartbeat", { tab: TAB, acquire: false }],
      ["/api/presence/frames/f1/lock/acquire", { tab: TAB }],
      ["/api/presence/frames/f1/lock/release", { tab: TAB, leave: true }],
      ["/api/presence/frames/f1/lock/takeover", { tab: TAB }],
    ]);
    for (const [, body] of api.apiPost.mock.calls) {
      expect(Object.keys(body as object).some((k) => /name|user|role/.test(k))).toBe(false);
    }
  });

  it("escapes the frame id it puts in the path", async () => {
    await presenceApi.heartbeat("a/b?c", TAB);
    expect(api.apiPost.mock.calls[0][0]).toBe("/api/presence/frames/a%2Fb%3Fc/heartbeat");
  });

  it("releases on unload through the shared client, as a keepalive POST", () => {
    presenceApi.releaseOnUnload("f1", TAB);
    expect(api.getBinary).toHaveBeenCalledTimes(1);
    const [path, init] = api.getBinary.mock.calls[0];
    expect(path).toBe("/api/presence/frames/f1/lock/release");
    expect(init).toMatchObject({ method: "POST", keepalive: true, headers: { "Content-Type": "application/json" } });
    expect(JSON.parse(String(init?.body))).toEqual({ tab: TAB, leave: true });
  });

  it("does not throw, or leave a rejection unhandled, when that last request fails", async () => {
    api.getBinary.mockRejectedValueOnce(new Error("page is gone"));
    expect(() => presenceApi.releaseOnUnload("f1", TAB)).not.toThrow();
    await Promise.resolve();
    await Promise.resolve();
  });
});
