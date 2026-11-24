// The script that runs in <head> before the body exists, and the storage keys it shares with the modules
// that write them.
//
// Deliberately not a client module. app/layout.tsx is a server component and needs this as a plain string;
// imported from a "use client" file it would arrive as a client reference instead. So the keys live here
// and lib/theme.ts and lib/i18n.ts import them, rather than the other way round.

export const THEME_PREF = "lbx.theme";
export const LOCALE_STORAGE_KEY = "lbx_locale";
export const LIGHT_QUERY = "(prefers-color-scheme: light)";
/** Every interface language. lib/i18n.test.ts holds this equal to LOCALES. */
export const LOCALE_CODES = ["en", "hi", "kn", "ta"] as const;

/**
 * Sets <html data-theme> and <html lang> from what this person chose, so the first paint is already in the
 * right palette and the document announces the right language. Without it a light-theme user sees the dark
 * page for as long as the bundle takes to load, on every full page load.
 *
 * It is a string because it must not wait for React. It is built from the constants above, so the keys
 * cannot drift from the ones the pickers write. Everything is inside try: storage can throw (a private
 * window, blocked site data), and a preference that cannot be read is the default, not a blank page.
 */
export const BOOT_SCRIPT = `(function(){try{var d=document.documentElement;var t=null;try{t=localStorage.getItem(${JSON.stringify(THEME_PREF)})}catch(e){}`
  + `var l=window.matchMedia&&window.matchMedia(${JSON.stringify(LIGHT_QUERY)}).matches;`
  + `d.dataset.theme=t==="light"?"light":(t==="system"&&l?"light":"dark");`
  + `var ok=${JSON.stringify(Object.fromEntries(LOCALE_CODES.map((c) => [c, 1])))};var g=null;`
  + `try{g=localStorage.getItem(${JSON.stringify(LOCALE_STORAGE_KEY)})}catch(e){}`
  + `if(!ok[g]){g=(navigator.language||"en").slice(0,2)}if(ok[g]){d.lang=g}}catch(e){}})();`;
