"use client";

import { useEffect, useState } from "react";

import { LOCALES, T, getLocale, setLocale, type Locale } from "@/lib/i18n";

// The interface language, somewhere a person can actually find it.
//
// Four locales are implemented - English, Hindi, Kannada, Tamil - chosen because this is a product built
// for Indian roads where the people doing the annotating are far more likely to read one of the latter
// three than the person who wrote the labels. The only way to change it was a picker inside the onboarding
// tour, which a user sees once and then cannot get back to without knowing the `lbx:language` menu entry
// re-opens the tour. A setting reachable only from a thing you have already dismissed is not a setting.
//
// Reads at mount rather than during render, because getLocale touches localStorage and navigator, and the
// server render has neither - a mismatch would hydrate wrong and flicker.
export default function LocalePicker({ compact = false }: { compact?: boolean }) {
  const [locale, setLocaleState] = useState<Locale>("en");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setLocaleState(getLocale());
    setReady(true);
  }, []);

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <label htmlFor={compact ? "locale-menu" : "locale"} className="text-[12.5px] font-medium text-ink-2">
        <T k="prefs.language">Interface language</T>
      </label>
      <select
        id={compact ? "locale-menu" : "locale"}
        value={ready ? locale : "en"}
        disabled={!ready}
        onChange={(e) => setLocale(e.target.value as Locale)}
        className="field"
      >
        {LOCALES.map((l) => (
          <option key={l.code} value={l.code}>
            {l.native} ({l.label})
          </option>
        ))}
      </select>
      {!compact && (
        <span className="text-[12px] text-ink-3">
          <T k="prefs.language_note">the page reloads; class names and ids are never translated</T>
        </span>
      )}
    </div>
  );
}
