"use client";

import { useCallback } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

// List-window state that lives in the URL, so a list -> detail -> back round-trip lands you exactly where you
// were: same page, same filters, because the browser restores the query string. Before this, returning to a
// list reset it to the top and the first page. The offset is a query param; setOffset rewrites it without a
// full navigation (scroll: false), and setParam does the same for any filter a page wants to persist.

export function usePager(limit = 50) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const offset = Math.max(0, parseInt(params.get("offset") || "0", 10) || 0);

  const write = useCallback((mut: (p: URLSearchParams) => void) => {
    const p = new URLSearchParams(Array.from(params.entries()));
    mut(p);
    router.replace(`${pathname}?${p.toString()}`, { scroll: false });
  }, [params, pathname, router]);

  const setOffset = useCallback((n: number) => {
    write((p) => { if (n <= 0) p.delete("offset"); else p.set("offset", String(n)); });
  }, [write]);

  // A filter change resets to the first page, since the old offset is meaningless against a new result set.
  const setParam = useCallback((key: string, value: string | null) => {
    write((p) => { if (value == null || value === "") p.delete(key); else p.set(key, value); p.delete("offset"); });
  }, [write]);

  return { offset, limit, setOffset, setParam, get: (k: string) => params.get(k) };
}
