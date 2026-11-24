"use client";

// One cache for what the app reads from the API.
//
// Every page fetched in a mount effect with `cache: "no-store"`, kept the answer in its own useState and
// threw it away on unmount. So going list -> detail -> back refetched the list from nothing and showed a
// spinner where the list had been a second earlier, two components asking for the same thing on one page
// asked twice, and each poll was a hand-written setInterval that kept firing in a tab nobody was looking at.
//
// This is a thin layer over SWR, which is the part worth not writing: request dedupe, a stale-while-
// revalidate cache, and the race handling that makes a slow first response lose to a fast second one.
// What is added here is the part this app needs and SWR does not decide for it:
//
//   - keys are arrays with a namespace first (["jobs"], ["dataset", id]), so a mutation can invalidate
//     everything under a prefix without knowing which filters and pages happen to be cached;
//   - nothing refetches on focus or retries on error unless asked, because the pages being migrated did
//     neither and a retry storm against a backend that just returned 401 or 503 is not an improvement;
//   - polling keeps its cadence through a failure (SWR's own interval stops while the key holds an error
//     and leaves recovery to the retry it was told not to do) and stops while the tab is hidden;
//   - the result carries the three props LoadState takes, so a page keeps the loading / failed / empty
//     contract it already has.
//
// It does not import lib/api: the fetcher is whatever the caller already called (`() => api.jobs()`), so
// no URL or request option changes by moving a call here.

import { useCallback, useEffect, useMemo, useRef } from "react";
import useSWR, { mutate as swrMutate, unstable_serialize, useSWRConfig } from "swr";

/** A cache key. A bare string is the same key as a one-element array. */
export type QueryKey = string | readonly unknown[];

export type QueryOptions<T> = {
  /** false: do not request (the same as passing a null key). For a call that waits on an id or a role. */
  enabled?: boolean;
  /**
   * Refetch every this many milliseconds while mounted and the tab is visible. A function is given the
   * latest data and error so a poll can speed up while something is running and back off after a failure.
   * 0 or undefined: no polling.
   */
  pollMs?: number | ((data: T | undefined, error: unknown) => number);
  /** Keep showing the previous key's data while a new key (the next page, another filter) loads. */
  keepPrevious?: boolean;
  /** Requests for the same key inside this window share one response. Default 2000. */
  dedupeMs?: number;
  /** Refetch when the window regains focus. Off by default: no page did this before. */
  revalidateOnFocus?: boolean;
};

export type QueryResult<T> = {
  data: T | undefined;
  error: unknown;
  /** A first load is in flight: there is nothing to show yet. */
  loading: boolean;
  /** Any request is in flight, including a background refresh over data already shown. */
  validating: boolean;
  /** Fetch again now, bypassing the dedupe window. Resolves when the request settles. */
  reload: () => Promise<T | undefined>;
  /** Replace the cached value without a request, for a change the page already knows the result of. */
  setData: (next: T | undefined | ((current: T | undefined) => T | undefined)) => void;
  /** Spread into <LoadState>: loading, error and onRetry. `empty` stays the page's call. */
  loadState: { loading: boolean; error: unknown; onRetry: () => void };
};

const defaults = { dedupeMs: 2000 };

/** Process-wide defaults. The test setup sets dedupeMs to 0 so one test's response cannot answer the next. */
export function configureQueries(next: Partial<typeof defaults>): void {
  Object.assign(defaults, next);
}

const parts = (key: QueryKey): readonly unknown[] => (typeof key === "string" ? [key] : key);

const same = (a: unknown, b: unknown): boolean => a === b || JSON.stringify(a) === JSON.stringify(b);

/** Whether `key` is `prefix` or sits under it: ["jobs", "running"] is under "jobs", "jobsx" is not. */
export function keyMatches(key: unknown, prefix: QueryKey): boolean {
  if (typeof key !== "string" && !Array.isArray(key)) return false;
  const k = parts(key as QueryKey);
  const p = parts(prefix);
  return p.length <= k.length && p.every((v, i) => same(v, k[i]));
}

/**
 * Mark everything under the given prefixes stale. Mounted queries refetch now; cached ones refetch when
 * something next reads them. Call it after a mutation, with the keys the mutation changed.
 */
export function invalidate(...prefixes: QueryKey[]): Promise<unknown> {
  return swrMutate((key) => prefixes.some((p) => keyMatches(key, p)));
}

/** Write a value into the cache without a request. */
export function setQueryData<T>(key: QueryKey, data: T): Promise<unknown> {
  return swrMutate(parts(key), data, { revalidate: false });
}

/** Drop every cached value. For sign-out, and for tests. */
export function clearQueries(): Promise<unknown> {
  return swrMutate(() => true, undefined, { revalidate: false });
}

const hidden = (): boolean => typeof document !== "undefined" && document.visibilityState === "hidden";

export function useQuery<T>(key: QueryKey | null | false | undefined, fetcher: () => Promise<T>,
                            opts: QueryOptions<T> = {}): QueryResult<T> {
  const active = Boolean(key) && opts.enabled !== false;
  const swrKey = active ? parts(key as QueryKey) : null;
  const { data, error, isLoading, isValidating, mutate } = useSWR<T, unknown>(swrKey, () => fetcher(), {
    dedupingInterval: opts.dedupeMs ?? defaults.dedupeMs,
    revalidateOnFocus: opts.revalidateOnFocus ?? false,
    revalidateOnReconnect: false,
    shouldRetryOnError: false,
    keepPreviousData: opts.keepPrevious ?? false,
  });

  // Read through a ref so changing the cadence does not tear the timer down.
  const pollRef = useRef(opts.pollMs);
  pollRef.current = opts.pollMs;

  const { cache } = useSWRConfig();
  const polled = active && Boolean(opts.pollMs);
  const cacheKey = swrKey ? unstable_serialize(swrKey) : "";
  // Restart the timer when the first response lands, so the first beat is measured from an answer rather
  // than from mount: a cadence that speeds up while something is running should not sit out one idle
  // interval before it finds out that something is.
  const settled = data !== undefined || error !== undefined;
  useEffect(() => {
    if (!polled) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    let missed = false;
    const schedule = () => {
      if (stopped) return;
      const p = pollRef.current;
      // The cache, not this component's last render: the request has settled but React may not have
      // rendered its result yet, and a cadence that backs off on error has to see this response.
      const state = cache.get(cacheKey) as { data?: T; error?: unknown } | undefined;
      const ms = typeof p === "function" ? p(state?.data, state?.error) : p ?? 0;
      if (ms > 0) timer = setTimeout(tick, ms);
    };
    const tick = () => {
      if (stopped) return;
      // A hidden tab is not read by anyone. Note the missed beat and catch up when it is shown again,
      // rather than keeping a request going every few seconds behind somebody's other work.
      if (hidden()) { missed = true; return; }
      void mutate().catch(() => undefined).then(schedule);
    };
    const onVisibility = () => {
      if (!hidden() && missed) { missed = false; tick(); }
    };
    document.addEventListener("visibilitychange", onVisibility);
    schedule();
    return () => {
      stopped = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [polled, cacheKey, cache, mutate, settled]);

  const reload = useCallback(() => mutate(), [mutate]);
  const setData = useCallback<QueryResult<T>["setData"]>((next) => {
    void mutate(next as Parameters<typeof mutate>[0], { revalidate: false });
  }, [mutate]);

  return useMemo(() => ({
    data, error, loading: isLoading, validating: isValidating, reload, setData,
    // While a retry with nothing on screen is in flight the page is loading, not failed: the previous
    // error is about a request that has been superseded.
    loadState: { loading: isLoading, error: isLoading ? undefined : error, onRetry: () => { void reload(); } },
  }), [data, error, isLoading, isValidating, reload, setData]);
}
