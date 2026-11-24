"use client";

// Light and dark.
//
// The palette is two sets of CSS variables in app/globals.css, switched by `data-theme` on <html>. This
// module owns the preference: what the person chose and what that resolves to on this device. The script
// that applies it before the first paint is lib/bootScript.ts.
//
// Three choices, not two. "System" follows the device and keeps following it, so somebody whose laptop
// goes dark at sunset gets that here too. It is a choice, not the default: until somebody picks, the
// theme is dark, which is what this product has always looked like. Defaulting to the device would have
// switched every person whose desktop is in light mode to the light palette on their next page load,
// without their asking and before anyone had looked at it.

import { useSyncExternalStore } from "react";

import { LIGHT_QUERY, THEME_PREF } from "./bootScript";
import { readPref, writePref } from "./prefs";

export type ThemePref = "system" | "dark" | "light";
export type Theme = "dark" | "light";

export const THEME_EVENT = "lbx:theme";
export const THEME_PREFS: ThemePref[] = ["system", "dark", "light"];

export function readThemePref(): ThemePref {
  const v = readPref(THEME_PREF);
  return v === "light" || v === "system" ? v : "dark";
}

export function prefersLight(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    && window.matchMedia(LIGHT_QUERY).matches;
}

/** What a preference means on a device that does or does not prefer light. */
export function resolveTheme(pref: ThemePref, deviceLight: boolean): Theme {
  if (pref === "system") return deviceLight ? "light" : "dark";
  return pref;
}

/** Put the resolved theme on <html>. Idempotent: the inline script has usually done it already. */
export function applyTheme(pref: ThemePref = readThemePref()): Theme {
  const theme = resolveTheme(pref, prefersLight());
  if (typeof document !== "undefined") document.documentElement.dataset.theme = theme;
  return theme;
}

export function setThemePref(pref: ThemePref): void {
  writePref(THEME_PREF, pref);
  applyTheme(pref);
  window.dispatchEvent(new Event(THEME_EVENT));
}

function subscribe(onChange: () => void): () => void {
  const onStorage = (e: StorageEvent) => { if (e.key === THEME_PREF) onChange(); };
  window.addEventListener(THEME_EVENT, onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(THEME_EVENT, onChange);
    window.removeEventListener("storage", onStorage);
  };
}

const serverPref = (): ThemePref => "dark";

/** The stored preference, safe to read during render (the server and hydration see "dark"). */
export function useThemePref(): ThemePref {
  return useSyncExternalStore(subscribe, readThemePref, serverPref);
}

/**
 * Keep <html> in step with the preference after load: another tab changing it, and the device switching
 * between light and dark while the choice is "system". Returns the cleanup.
 */
export function watchTheme(): () => void {
  applyTheme();
  const reapply = () => { applyTheme(); };
  const unsubscribe = subscribe(reapply);
  const media = typeof window.matchMedia === "function" ? window.matchMedia(LIGHT_QUERY) : null;
  media?.addEventListener?.("change", reapply);
  return () => {
    unsubscribe();
    media?.removeEventListener?.("change", reapply);
  };
}
