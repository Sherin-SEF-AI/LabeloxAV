"use client";

// Panel layout choices that should survive a reload, following the one precedent in the tree
// (components/editor/CanvasConsole.tsx and its `lbx.canvasConsole.open`) rather than inventing a second
// storage convention.
//
// The frame editor is a route, so `[` and `]` remount the entire page. Without this, an annotator who
// works the agent all day re-picks the agent tab on every single frame. That is the whole reason this
// file exists.
//
// Two things are deliberately NOT persisted, because a restored value would hide objects with no visible
// cause and read as data loss:
//
//   - the object search box: a stale filter on load looks like the frame lost its objects.
//   - per-class group collapse: a group collapsed on a junction frame means nothing on the next frame,
//     which may not contain that class at all.

import { useCallback, useEffect, useState } from "react";

const PREFIX = "lbx.props.";

export const PREF_TOOL_TAB = `${PREFIX}toolTab`;
export const PREF_OBJECTS_OPEN = `${PREFIX}objectsOpen`;
/** Per-PanelSection open flag. The slug comes from the section title, which is stable and unique. */
export const prefSection = (slug: string) => `${PREFIX}section.${slug}`;

// Both accessors swallow. Safari in private mode throws on localStorage access rather than returning
// null, and a layout preference is never worth taking the panel down for.
export function readPref(key: string, fallback: string, allowed?: readonly string[]): string {
  try {
    const raw = globalThis.localStorage?.getItem(key);
    if (raw == null) return fallback;
    // A stored value that is no longer a legal option must not be trusted. The tool tabs were renamed
    // once already; a leftover "tools" selecting a tab that does not exist renders an empty panel with no
    // tab highlighted, which looks like a load failure.
    if (allowed && !allowed.includes(raw)) return fallback;
    return raw;
  } catch {
    return fallback;
  }
}

export function writePref(key: string, value: string): void {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    /* private mode, or the quota is full. A layout preference is not worth surfacing. */
  }
}

// Seeded from the fallback and corrected on mount, NOT read in the useState initialiser. These are
// "use client" components but Next still renders them on the server, where localStorage does not exist
// and where a value read on the client would not match the server's markup. CanvasConsole reads in an
// effect for exactly this reason. The cost is one frame at the default before it corrects.
// A null key opts out entirely and the hook behaves as a plain useState. That exists so a component can
// take an optional storageKey without branching on it at the call site, which would call a different
// number of hooks per render and fail react-hooks/rules-of-hooks.
export function usePanelPref<T extends string>(
  key: string | null, fallback: T, allowed?: readonly T[],
): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(fallback);

  useEffect(() => {
    if (key == null) return;
    setValue(readPref(key, fallback, allowed) as T);
    // `allowed` is a literal array at every call site, so a new identity every render would re-run this
    // effect forever. The key is what identifies the preference.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const set = useCallback((v: T) => {
    setValue(v);
    if (key != null) writePref(key, v);
  }, [key]);

  return [value, set];
}

// The boolean case, which is most of them. Stored as "1"/"0" to match lbx.canvasConsole.open.
export function usePanelFlag(key: string | null, fallback: boolean): [boolean, (v: boolean) => void] {
  const [raw, setRaw] = usePanelPref(key, fallback ? "1" : "0", ["1", "0"] as const);
  const set = useCallback((v: boolean) => setRaw(v ? "1" : "0"), [setRaw]);
  return [raw === "1", set];
}
