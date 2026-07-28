"use client";

import { useSearchParams } from "next/navigation";

// Read a query-string parameter as the initial value of a page's state.
//
// The nav menus deep-link with query strings (/import?format=cvat, /projects?mine=1). Those links were inert
// because the destination pages never read the param, so fourteen import-format menu entries all landed on
// the same default and the menu was a stable map of nothing. Pages that use this must wrap their body in a
// <Suspense> boundary: useSearchParams forces a client-side-render bailout that Next requires be suspended,
// or the static prerender fails the build.
export function useQueryParam(name: string): string | null {
  return useSearchParams().get(name);
}

// A boolean flag query param: present and not "0"/"false" means on (?mine=1, ?rig=1).
export function useQueryFlag(name: string): boolean {
  const raw = useSearchParams().get(name);
  return raw != null && raw !== "0" && raw !== "false";
}
