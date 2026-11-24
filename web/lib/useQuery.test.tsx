import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import LoadState from "@/components/shell/LoadState";

import { clearQueries, configureQueries, invalidate, keyMatches, setQueryData, useQuery } from "./useQuery";

// The contract the migrated pages rely on. Each of these was a hand-written flag or timer in a page before,
// and each was slightly different from the next one.

let n = 0;
/** A fresh namespace per test, so nothing here depends on another test's cache. */
const ns = () => `t${++n}`;

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => state });
  document.dispatchEvent(new Event("visibilitychange"));
}

afterEach(() => {
  setVisibility("visible");
  vi.useRealTimers();
  configureQueries({ dedupeMs: 0 });
});

describe("keyMatches", () => {
  it("matches a key under its prefix and not a longer name", () => {
    expect(keyMatches(["jobs", "running"], "jobs")).toBe(true);
    expect(keyMatches(["jobs"], ["jobs"])).toBe(true);
    expect(keyMatches("jobs", "jobs")).toBe(true);
    expect(keyMatches(["jobsx"], "jobs")).toBe(false);
    expect(keyMatches(["jobs"], ["jobs", "running"])).toBe(false);
  });

  it("compares object parts by value", () => {
    expect(keyMatches(["explore", { state: "auto" }, 0], ["explore", { state: "auto" }])).toBe(true);
    expect(keyMatches(["explore", { state: "auto" }], ["explore", { state: "human" }])).toBe(false);
  });

  it("ignores keys that are not ours", () => {
    expect(keyMatches(undefined, "jobs")).toBe(false);
    expect(keyMatches({ a: 1 }, "jobs")).toBe(false);
  });
});

describe("useQuery", () => {
  it("loads, then holds the data", async () => {
    const key = ns();
    const { result } = renderHook(() => useQuery([key], () => Promise.resolve(7)));
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeUndefined();
    await waitFor(() => expect(result.current.data).toBe(7));
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeUndefined();
  });

  it("treats a bare string as a one-part key", async () => {
    const key = ns();
    const fetcher = vi.fn(() => Promise.resolve("a"));
    const one = renderHook(() => useQuery(key, fetcher));
    await waitFor(() => expect(one.result.current.data).toBe("a"));
    const two = renderHook(() => useQuery([key], fetcher));
    // Already cached under the same key: there is something to show on the first render.
    expect(two.result.current.data).toBe("a");
  });

  it("does not request without a key, or when disabled", async () => {
    const fetcher = vi.fn(() => Promise.resolve(1));
    const a = renderHook(() => useQuery(null, fetcher));
    const key = ns();
    const b = renderHook(() => useQuery([key], fetcher, { enabled: false }));
    await act(async () => { await Promise.resolve(); });
    expect(fetcher).not.toHaveBeenCalled();
    expect(a.result.current.loading).toBe(false);
    expect(b.result.current.loading).toBe(false);
  });

  it("shares one request between two readers of the same key", async () => {
    configureQueries({ dedupeMs: 2000 });
    const key = ns();
    const fetcher = vi.fn(() => Promise.resolve("shared"));
    const a = renderHook(() => useQuery([key], fetcher));
    const b = renderHook(() => useQuery([key], fetcher));
    await waitFor(() => expect(a.result.current.data).toBe("shared"));
    await waitFor(() => expect(b.result.current.data).toBe("shared"));
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("serves the cached value to a later mount and refreshes it behind", async () => {
    const key = ns();
    let value = "first";
    const first = renderHook(() => useQuery([key], () => Promise.resolve(value)));
    await waitFor(() => expect(first.result.current.data).toBe("first"));
    first.unmount();
    // Let the dedupe window (0 in tests) close.
    await act(async () => { await new Promise((r) => setTimeout(r, 5)); });

    value = "second";
    const again = renderHook(() => useQuery([key], () => Promise.resolve(value)));
    // Stale-while-revalidate: the list is there at once, not a spinner...
    expect(again.result.current.data).toBe("first");
    expect(again.result.current.loading).toBe(false);
    // ...and the fresh answer replaces it.
    await waitFor(() => expect(again.result.current.data).toBe("second"));
  });

  it("reports a failure and recovers on reload", async () => {
    const key = ns();
    let fail = true;
    const { result } = renderHook(() =>
      useQuery([key], () => (fail ? Promise.reject(new Error("down")) : Promise.resolve("up"))));
    await waitFor(() => expect(result.current.error).toBeInstanceOf(Error));
    expect(result.current.loading).toBe(false);
    expect(result.current.loadState.error).toBeInstanceOf(Error);

    fail = false;
    await act(async () => { await result.current.reload(); });
    expect(result.current.data).toBe("up");
    expect(result.current.error).toBeUndefined();
  });

  it("does not retry a failure on its own", async () => {
    const key = ns();
    const fetcher = vi.fn(() => Promise.reject(new Error("401")));
    const { result } = renderHook(() => useQuery([key], fetcher));
    await waitFor(() => expect(result.current.error).toBeTruthy());
    await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("ignores a response that lands after unmount", async () => {
    const key = ns();
    let release: (v: string) => void = () => {};
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    const { unmount } = renderHook(() =>
      useQuery([key], () => new Promise<string>((r) => { release = r; })));
    unmount();
    await act(async () => { release("late"); await Promise.resolve(); });
    // No state update on an unmounted component, which React reports through console.error.
    expect(errors).not.toHaveBeenCalled();
    errors.mockRestore();
  });

  it("keeps the last key's data while the next key loads, when asked", async () => {
    const base = ns();
    const { result, rerender } = renderHook(
      ({ page }: { page: number }) =>
        useQuery([base, page], () => Promise.resolve(`page ${page}`), { keepPrevious: true }),
      { initialProps: { page: 0 } });
    await waitFor(() => expect(result.current.data).toBe("page 0"));
    rerender({ page: 1 });
    expect(result.current.data).toBe("page 0");
    await waitFor(() => expect(result.current.data).toBe("page 1"));
  });

  it("setData replaces the value without a request", async () => {
    const key = ns();
    const fetcher = vi.fn(() => Promise.resolve([1, 2, 3]));
    const { result } = renderHook(() => useQuery([key], fetcher));
    await waitFor(() => expect(result.current.data).toEqual([1, 2, 3]));
    act(() => result.current.setData((cur) => (cur ?? []).filter((x) => x !== 2)));
    await waitFor(() => expect(result.current.data).toEqual([1, 3]));
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});

describe("invalidate", () => {
  it("refetches every mounted query under the prefix and leaves the rest", async () => {
    const base = ns();
    const other = ns();
    const list = vi.fn(() => Promise.resolve("list"));
    const detail = vi.fn(() => Promise.resolve("detail"));
    const unrelated = vi.fn(() => Promise.resolve("other"));
    const a = renderHook(() => useQuery([base], list));
    const b = renderHook(() => useQuery([base, "id-1"], detail));
    const c = renderHook(() => useQuery([other], unrelated));
    await waitFor(() => expect(a.result.current.data).toBe("list"));
    await waitFor(() => expect(b.result.current.data).toBe("detail"));
    await waitFor(() => expect(c.result.current.data).toBe("other"));

    await act(async () => { await invalidate(base); });
    expect(list).toHaveBeenCalledTimes(2);
    expect(detail).toHaveBeenCalledTimes(2);
    expect(unrelated).toHaveBeenCalledTimes(1);
  });

  it("bypasses the dedupe window, because the data is known to have changed", async () => {
    configureQueries({ dedupeMs: 60_000 });
    const key = ns();
    let value = 1;
    const { result } = renderHook(() => useQuery([key], () => Promise.resolve(value)));
    await waitFor(() => expect(result.current.data).toBe(1));
    value = 2;
    await act(async () => { await invalidate([key]); });
    expect(result.current.data).toBe(2);
  });

  it("setQueryData seeds a key, and clearQueries empties the cache", async () => {
    const key = ns();
    await act(async () => { await setQueryData([key], "seeded"); });
    const never = () => new Promise<string>(() => {});
    const seeded = renderHook(() => useQuery([key], never));
    expect(seeded.result.current.data).toBe("seeded");
    seeded.unmount();

    await act(async () => { await clearQueries(); });
    const after = renderHook(() => useQuery([key], never));
    expect(after.result.current.data).toBeUndefined();
  });
});

describe("polling", () => {
  // setImmediate stays real: React's act() waits on it, and faking it hangs every awaited act.
  beforeEach(() => { vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] }); });

  /** Let the pending request, React's render and the zero-delay reschedule all run. */
  const settle = async () => { await act(async () => { await vi.advanceTimersByTimeAsync(1); }); };

  it("refetches on the interval", async () => {
    const fetcher = vi.fn(() => Promise.resolve("x"));
    const key = ns();
    renderHook(() => useQuery([key], fetcher, { pollMs: 1000 }));
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it("keeps polling after a failure, as the interval it replaces did", async () => {
    let calls = 0;
    const fetcher = vi.fn(() => (++calls === 2 ? Promise.reject(new Error("blip")) : Promise.resolve(calls)));
    const key = ns();
    const { result } = renderHook(() => useQuery([key], fetcher, { pollMs: 1000 }));
    await settle();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    await settle();
    expect(result.current.error).toBeInstanceOf(Error);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(result.current.error).toBeUndefined();
    expect(result.current.data).toBe(3);
  });

  it("pauses while the tab is hidden and catches up when it is shown", async () => {
    const fetcher = vi.fn(() => Promise.resolve("x"));
    const key = ns();
    renderHook(() => useQuery([key], fetcher, { pollMs: 1000 }));
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(1);

    setVisibility("hidden");
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(fetcher).toHaveBeenCalledTimes(1);

    await act(async () => { setVisibility("visible"); await vi.advanceTimersByTimeAsync(1); });
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("does not refetch on becoming visible when no beat was missed", async () => {
    const fetcher = vi.fn(() => Promise.resolve("x"));
    const key = ns();
    renderHook(() => useQuery([key], fetcher, { pollMs: 5000 }));
    await settle();
    setVisibility("hidden");
    setVisibility("visible");
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("lets the cadence follow the latest result", async () => {
    let calls = 0;
    const fetcher = vi.fn(() => (++calls === 1 ? Promise.resolve("running") : Promise.reject(new Error("down"))));
    const cadence = vi.fn((data: string | undefined, error: unknown) =>
      (error ? 30_000 : data === "running" ? 3000 : 20_000));
    const key = ns();
    renderHook(() => useQuery([key], fetcher, { pollMs: cadence }));
    await settle();
    // The first answer says something is running, so the next beat is the live cadence...
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(2);
    // ...and that one failed, so the beat after it is the error backoff, not the live one.
    await act(async () => { await vi.advanceTimersByTimeAsync(20_000); });
    expect(fetcher).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    await settle();
    expect(fetcher).toHaveBeenCalledTimes(3);
  });

  it("a zero cadence stops the poll", async () => {
    const fetcher = vi.fn(() => Promise.resolve("done"));
    const key = ns();
    renderHook(() => useQuery([key], fetcher, { pollMs: (d) => (d === "done" ? 0 : 1000) }));
    await settle();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    await settle();
    const after = fetcher.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(fetcher).toHaveBeenCalledTimes(after);
  });

  it("stops on unmount", async () => {
    const fetcher = vi.fn(() => Promise.resolve("x"));
    const key = ns();
    const { unmount } = renderHook(() => useQuery([key], fetcher, { pollMs: 1000 }));
    await settle();
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});

describe("with LoadState", () => {
  function Jobs({ fetcher }: { fetcher: () => Promise<string[]> }) {
    const q = useQuery(["jobs-view", fetcher.name || "f"], fetcher);
    return (
      <LoadState {...q.loadState} empty={q.data?.length === 0} emptyText="no jobs">
        <ul>{(q.data ?? []).map((j) => <li key={j}>{j}</li>)}</ul>
      </LoadState>
    );
  }

  it("spreads into the loading, failed and empty states the pages already render", async () => {
    const { unmount } = render(<Jobs fetcher={function ok() { return Promise.resolve(["a", "b"]); }} />);
    expect(screen.getByText("loading")).toBeInTheDocument();
    expect(await screen.findByText("a")).toBeInTheDocument();
    unmount();

    const empty = render(<Jobs fetcher={function none() { return Promise.resolve([]); }} />);
    expect(await screen.findByText("no jobs")).toBeInTheDocument();
    empty.unmount();

    render(<Jobs fetcher={function bad() { return Promise.reject(new Error("boom")); }} />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("shows loading, not the old error, while a retry with nothing on screen is in flight", async () => {
    const key = ns();
    let release: (v: string) => void = () => {};
    let first = true;
    const { result } = renderHook(() => useQuery([key], () => {
      if (first) { first = false; return Promise.reject(new Error("down")); }
      return new Promise<string>((r) => { release = r; });
    }));
    await waitFor(() => expect(result.current.loadState.error).toBeInstanceOf(Error));
    act(() => result.current.loadState.onRetry());
    await waitFor(() => expect(result.current.loadState.loading).toBe(true));
    expect(result.current.loadState.error).toBeUndefined();
    await act(async () => { release("ok"); await Promise.resolve(); });
    await waitFor(() => expect(result.current.data).toBe("ok"));
  });
});
