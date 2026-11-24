"use client";

import { T, useT } from "@/lib/i18n";
import { THEME_PREFS, setThemePref, useThemePref, type ThemePref } from "@/lib/theme";

// Light, dark, or whatever the device is using. Unlike the language this takes effect at once: the palette
// is CSS variables on <html>, so nothing has to re-render to change colour.
export default function ThemePicker({ compact = false }: { compact?: boolean }) {
  const pref = useThemePref();
  const t = useT();
  const fallback: Record<ThemePref, string> = { system: "System", dark: "Dark", light: "Light" };
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <label htmlFor={compact ? "theme-menu" : "theme"} className="text-[10.5px] font-medium uppercase tracking-[0.06em] text-ink-3">
        <T k="prefs.theme">theme</T>
      </label>
      <select
        id={compact ? "theme-menu" : "theme"}
        value={pref}
        onChange={(e) => setThemePref(e.target.value as ThemePref)}
        className="bg-bg border border-line px-1.5 py-0.5 text-ink text-[12.5px]"
      >
        {THEME_PREFS.map((p) => <option key={p} value={p}>{t(`prefs.theme.${p}`, fallback[p])}</option>)}
      </select>
      {!compact && (
        <span className="text-[11px] text-ink-3">
          <T k="prefs.theme_note">saved in this browser; System follows this device</T>
        </span>
      )}
    </div>
  );
}
