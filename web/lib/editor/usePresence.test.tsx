import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LockHolder, PresenceState } from "@/lib/presenceApi";

import { usePresence, type PresenceTransport } from "./usePresence";

// The part of presence that needs a clock and a document: that it beats on a schedule, stops while the
// tab is hidden, starts again when it is shown, and lets go when the frame changes or the page closes.
// What an answer MEANS is tested in presence.test.ts; here the answers are canned.

const state = (frameId: string, over: Partial<PresenceState> = {}): PresenceState => ({
  frame_id: frameId, degraded: false, others: [], lock: { mine: false, holder: null },
  can_take_over: false, ttl_s: 15, heartbeat_s: 5, ...over,
});
const me: LockHolder = { user_id: "u-me", name: "Me", role: "annotator", since_ms: 1000, same_user: true };
const asha: LockHolder = { user_id: "u-asha", name: "Asha", role: "annotator", since_ms: 500, same_user: false };
const mine = (f: string) => state(f, { lock: { mine: true, holder: me } });
const heldByAsha = (f: string, canTakeOver = false) => state(f, {
  lock: { mine: false, holder: asha }, can_take_over: canTakeOver,
  others: [{ user_id: asha.user_id, name: asha.name, role: asha.role, editing: true }],
});

function transport(answer: (frameId: string) => PresenceState = mine) {
  const t = {
    acquire: vi.fn(async (frameId: string, _tab: string) => answer(frameId)),
    heartbeat: vi.fn(async (frameId: string, _tab: string, _acquire?: boolean) => answer(frameId)),
    release: vi.fn(async (frameId: string, _tab: string) => ({ frame_id: frameId, released: true, degraded: false })),
    takeOver: vi.fn(async (frameId: string, _tab: string) => mine(frameId)),
    releaseOnUnload: vi.fn((_frameId: string, _tab: string) => undefined),
  };
  return t satisfies PresenceTransport;
}

let hidden = false;
function setHidden(v: boolean) {
  hidden = v;
  document.dispatchEvent(new Event("visibilitychange"));
}

beforeEach(() => {
  vi.useFakeTimers();
  hidden = false;
  Object.defineProperty(document, "hidden", { configurable: true, get: () => hidden });
});
afterEach(() => { vi.useRealTimers(); });

const flush = () => act(async () => { await vi.advanceTimersByTimeAsync(0); });
const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

const mount = (t: PresenceTransport, frameId = "f1", hasUser = true) =>
  renderHook(({ id }) => usePresence(id, { transport: t, hasUser: () => hasUser }), { initialProps: { id: frameId } });

describe("opening a frame", () => {
  it("is view-only until the first answer, then editable when the lock is this tab's", async () => {
    const t = transport();
    const { result } = mount(t);
    expect(result.current.status).toBe("connecting");
    expect(result.current.readOnly).toBe(true);

    await flush();
    expect(t.acquire).toHaveBeenCalledTimes(1);
    expect(t.acquire.mock.calls[0][0]).toBe("f1");
    expect(result.current.status).toBe("live");
    expect(result.current.lock.mine).toBe(true);
    expect(result.current.readOnly).toBe(false);
    expect(result.current.degraded).toBe(false);
  });

  it("is view-only, with the holder named and listed, when somebody else has it", async () => {
    const { result } = mount(transport(heldByAsha));
    await flush();
    expect(result.current.readOnly).toBe(true);
    expect(result.current.lock).toEqual({ mine: false, holder: asha });
    expect(result.current.others.map((p) => p.name)).toEqual(["Asha"]);
    expect(result.current.canTakeOver).toBe(false);
  });

  it("presents one token, in the alphabet the server accepts, on every call", async () => {
    const t = transport();
    mount(t);
    await flush();
    await tick(5000);
    const tab = t.acquire.mock.calls[0][1];
    expect(tab).toMatch(/^[A-Za-z0-9_-]{16,64}$/);
    expect(t.heartbeat.mock.calls[0][1]).toBe(tab);
  });

  it("with nobody signed in asks nothing and gates nothing", async () => {
    const t = transport();
    const { result } = mount(t, "f1", false);
    await flush();
    expect(t.acquire).not.toHaveBeenCalled();
    expect(result.current.degraded).toBe(true);
    expect(result.current.readOnly).toBe(false);
  });
});

describe("the heartbeat", () => {
  it("beats every five seconds and asks for the lock each time", async () => {
    const t = transport();
    mount(t);
    await flush();
    expect(t.heartbeat).not.toHaveBeenCalled();
    await tick(4999);
    expect(t.heartbeat).not.toHaveBeenCalled();
    await tick(1);
    expect(t.heartbeat).toHaveBeenCalledTimes(1);
    expect(t.heartbeat.mock.calls[0]).toEqual(["f1", t.acquire.mock.calls[0][1], true]);
    await tick(15000);
    expect(t.heartbeat).toHaveBeenCalledTimes(4);
    expect(t.acquire).toHaveBeenCalledTimes(1);
  });

  it("a viewer becomes the editor on the beat after the holder leaves", async () => {
    let held = true;
    const t = transport((f) => (held ? heldByAsha(f) : mine(f)));
    const { result } = mount(t);
    await flush();
    expect(result.current.readOnly).toBe(true);
    held = false;
    await tick(5000);
    expect(result.current.readOnly).toBe(false);
    expect(result.current.acquisitions).toBe(1);
    // Asha was editing while this tab watched, so the page is told whose edits it has not seen.
    expect(result.current.regainedFrom).toEqual(asha);
  });

  it("an editor whose frame is taken over goes view-only on the next beat and is told by whom", async () => {
    let taken = false;
    const t = transport((f) => (taken ? heldByAsha(f) : mine(f)));
    const { result } = mount(t);
    await flush();
    expect(result.current.readOnly).toBe(false);
    taken = true;
    await tick(5000);
    expect(result.current.readOnly).toBe(true);
    expect(result.current.lostTo).toEqual(asha);
  });

  it("one failed beat changes nothing; three in a row stop gating", async () => {
    const t = transport(heldByAsha);
    const { result } = mount(t);
    await flush();
    t.heartbeat.mockRejectedValue(new Error("network"));
    await tick(5000);
    expect(result.current.readOnly).toBe(true);
    expect(result.current.degraded).toBe(false);
    await tick(10000);
    expect(result.current.degraded).toBe(true);
    expect(result.current.readOnly).toBe(false);
    expect(result.current.others).toEqual([]);
  });

  it("a first call that fails leaves the editor usable, and keeps trying to acquire", async () => {
    const t = transport();
    t.acquire.mockRejectedValueOnce(new Error("api restarting"));
    const { result } = mount(t);
    await flush();
    expect(result.current.degraded).toBe(true);
    expect(result.current.readOnly).toBe(false);
    await tick(5000);
    expect(t.acquire).toHaveBeenCalledTimes(2);
    expect(result.current.status).toBe("live");
    expect(result.current.lock.mine).toBe(true);
  });

  it("does not stack a second request behind a slow one", async () => {
    const t = transport();
    let release: (s: PresenceState) => void = () => {};
    mount(t);
    await flush();
    t.heartbeat.mockImplementationOnce(() => new Promise<PresenceState>((r) => { release = r; }));
    await tick(5000);
    expect(t.heartbeat).toHaveBeenCalledTimes(1);
    await tick(20000);                      // the answer has not come; nothing else is sent
    expect(t.heartbeat).toHaveBeenCalledTimes(1);
    await act(async () => { release(mine("f1")); await vi.advanceTimersByTimeAsync(0); });
    await tick(5000);
    expect(t.heartbeat).toHaveBeenCalledTimes(2);
  });
});

describe("a tab in the background", () => {
  it("stops beating while hidden and beats at once on return", async () => {
    const t = transport();
    mount(t);
    await flush();
    await tick(5000);
    expect(t.heartbeat).toHaveBeenCalledTimes(1);

    act(() => setHidden(true));
    await tick(60000);
    expect(t.heartbeat).toHaveBeenCalledTimes(1);

    await act(async () => { setHidden(false); await vi.advanceTimersByTimeAsync(0); });
    expect(t.heartbeat).toHaveBeenCalledTimes(2);
    await tick(5000);
    expect(t.heartbeat).toHaveBeenCalledTimes(3);
  });

  it("does not release when hidden: the lock is left to lapse", async () => {
    const t = transport();
    mount(t);
    await flush();
    act(() => setHidden(true));
    await tick(30000);
    expect(t.release).not.toHaveBeenCalled();
    expect(t.releaseOnUnload).not.toHaveBeenCalled();
  });

  it("opened in the background, asks nothing until it is first shown", async () => {
    hidden = true;
    const t = transport();
    const { result } = mount(t);
    await tick(20000);
    expect(t.acquire).not.toHaveBeenCalled();
    expect(result.current.readOnly).toBe(true);
    await act(async () => { setHidden(false); await vi.advanceTimersByTimeAsync(0); });
    expect(t.acquire).toHaveBeenCalledTimes(1);
    expect(result.current.readOnly).toBe(false);
  });

  it("back to find the frame taken: view-only, and told who has it", async () => {
    let taken = false;
    const t = transport((f) => (taken ? heldByAsha(f) : mine(f)));
    const { result } = mount(t);
    await flush();
    act(() => setHidden(true));
    taken = true;
    await tick(60000);
    expect(result.current.readOnly).toBe(false);       // nothing has been heard yet
    await act(async () => { setHidden(false); await vi.advanceTimersByTimeAsync(0); });
    expect(result.current.readOnly).toBe(true);
    expect(result.current.lostTo).toEqual(asha);
  });
});

describe("leaving", () => {
  it("releases on unmount and stops beating", async () => {
    const t = transport();
    const { unmount } = mount(t);
    await flush();
    const tab = t.acquire.mock.calls[0][1];
    unmount();
    expect(t.release).toHaveBeenCalledWith("f1", tab);
    await tick(30000);
    expect(t.heartbeat).not.toHaveBeenCalled();
  });

  it("changing frame releases the old one, takes the new one, and is view-only in between", async () => {
    const t = transport();
    const { result, rerender } = mount(t);
    await flush();
    expect(result.current.readOnly).toBe(false);

    rerender({ id: "f2" });
    // The very render in which the frame changed: the old frame's "mine" must not carry over.
    expect(result.current.readOnly).toBe(true);
    expect(result.current.status).toBe("connecting");
    expect(result.current.model.frameId).toBe("f2");
    expect(t.release).toHaveBeenCalledWith("f1", expect.any(String));

    await flush();
    expect(t.acquire.mock.calls.map((c) => c[0])).toEqual(["f1", "f2"]);
    expect(result.current.readOnly).toBe(false);
    await tick(5000);
    expect(t.heartbeat.mock.calls.map((c) => c[0])).toEqual(["f2"]);
  });

  it("ignores an answer for the previous frame that arrives late", async () => {
    const t = transport();
    let late: (s: PresenceState) => void = () => {};
    t.acquire.mockImplementationOnce(() => new Promise<PresenceState>((r) => { late = r; }));
    t.acquire.mockImplementationOnce(async (f: string) => heldByAsha(f));
    const { result, rerender } = mount(t);
    rerender({ id: "f2" });
    await flush();
    expect(result.current.lock.holder).toEqual(asha);
    await act(async () => { late(mine("f1")); await vi.advanceTimersByTimeAsync(0); });
    expect(result.current.lock.holder).toEqual(asha);
    expect(result.current.readOnly).toBe(true);
  });

  it("releases with a keepalive request when the page is closed", async () => {
    const t = transport();
    mount(t);
    await flush();
    window.dispatchEvent(new Event("pagehide"));
    expect(t.releaseOnUnload).toHaveBeenCalledWith("f1", t.acquire.mock.calls[0][1]);
  });

  it("a release that fails is not an unhandled rejection", async () => {
    const t = transport();
    t.release.mockRejectedValue(new Error("gone"));
    const { unmount } = mount(t);
    await flush();
    unmount();
    await tick(0);
    expect(t.release).toHaveBeenCalledTimes(1);
  });
});

describe("taking over", () => {
  it("applies the server's answer at once, without waiting for the next beat", async () => {
    const t = transport((f) => heldByAsha(f, true));
    const { result } = mount(t);
    await flush();
    expect(result.current.canTakeOver).toBe(true);
    await act(async () => { await result.current.takeOver(); });
    expect(t.takeOver).toHaveBeenCalledWith("f1", t.acquire.mock.calls[0][1]);
    expect(result.current.lock.mine).toBe(true);
    expect(result.current.readOnly).toBe(false);
    expect(result.current.regainedFrom).toEqual(asha);
  });

  it("rejects when the server refuses, and stays view-only", async () => {
    const t = transport(heldByAsha);
    t.takeOver.mockRejectedValue(new Error("requires reviewer"));
    const { result } = mount(t);
    await flush();
    await expect(act(async () => { await result.current.takeOver(); })).rejects.toThrow("requires reviewer");
    expect(result.current.readOnly).toBe(true);
  });
});
