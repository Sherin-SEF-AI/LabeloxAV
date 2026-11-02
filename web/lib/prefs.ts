// Per-browser interface preferences: whether the sidebar is collapsed, whether Labs is shown.
//
// Conveniences only. Nothing here may decide what someone is allowed to see or do, because it lives in one
// browser and anyone can change it. Storage can also be missing or throw (a private window, blocked site
// data), so every access is guarded and the interface falls back to its defaults.

export const SIDEBAR_COLLAPSED_PREF = "lbx.sidebar.collapsed";
export const LABS_PREF = "lbx.labs";
// Fired when a preference that other mounted components read changes, so the sidebar reflects a toggle on
// the settings page without a reload.
export const LABS_EVENT = "lbx:labs";

export function readPref(key: string): string | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writePref(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage unavailable: the preference holds for this page view only, which is the honest fallback.
  }
}

export function setLabs(on: boolean): void {
  writePref(LABS_PREF, on ? "1" : "0");
  window.dispatchEvent(new Event(LABS_EVENT));
}
