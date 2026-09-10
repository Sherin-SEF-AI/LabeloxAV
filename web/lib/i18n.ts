"use client";

// Interface language.
//
// The app is English-only in a product built for India, where the people doing the annotating are far more
// likely to read Hindi, Kannada or Tamil than the person who wrote the labels. That is not a cosmetic gap:
// an annotator who has to decode "quarantine" or "promotion blocked" in a second language is slower and
// makes different mistakes, and those mistakes land in the corpus.
//
// Deliberately not react-intl or next-intl. Both bring a provider, a message-extraction step and a build
// plugin, and this app needs none of that: the strings are chrome, they are known at build time, and the
// data (class names, session ids, model versions) is never translated because renaming a class in the
// interface would make it impossible to talk about with the ontology.
//
// Missing keys fall through to English rather than rendering a key. A half-translated interface should read
// as partly English, not as `nav.review.queue`.

import { EN, type Dict } from "./locales/en";
import { HI } from "./locales/hi";
import { KN } from "./locales/kn";
import { TA } from "./locales/ta";

export type Locale = "en" | "hi" | "kn" | "ta";
export type { Dict };

export const LOCALES: { code: Locale; label: string; native: string }[] = [
  { code: "en", label: "English", native: "English" },
  { code: "hi", label: "Hindi", native: "हिन्दी" },
  { code: "kn", label: "Kannada", native: "ಕನ್ನಡ" },
  { code: "ta", label: "Tamil", native: "தமிழ்" },
];

const KEY = "lbx_locale";

// The dictionaries live in lib/locales/*.ts, one file per language. They were inline here and the file
// grew by four entries every time a single string was translated, which made every review of this module
// a review of four unrelated languages.
const DICTS: Record<Locale, Dict> = { en: EN, hi: HI, kn: KN, ta: TA };

let _locale: Locale | null = null;

export function getLocale(): Locale {
  if (_locale) return _locale;
  if (typeof window === "undefined") return "en";
  const stored = localStorage.getItem(KEY) as Locale | null;
  if (stored && stored in DICTS) { _locale = stored; return stored; }
  // The browser's own preference, which is a better first guess than English for a user who never opens
  // the language menu because they did not know it was there.
  const nav = (navigator.language || "en").slice(0, 2) as Locale;
  _locale = nav in DICTS ? nav : "en";
  return _locale;
}

export function setLocale(locale: Locale): void {
  _locale = locale;
  if (typeof window !== "undefined") {
    localStorage.setItem(KEY, locale);
    document.documentElement.lang = locale;
    // A full reload rather than a re-render. Strings are read at call time all over the tree, and a
    // context that re-renders only its consumers would leave half the interface in the previous language,
    // which is more confusing than either language alone.
    window.location.reload();
  }
}

/**
 * Translate a key, substituting `{name}` placeholders from `vars`.
 *
 * Falls through to English, then to the supplied fallback, then to the key, so nothing ever renders
 * blank and a partly translated interface reads as partly English rather than as dotted keys.
 *
 * Interpolation is here rather than at the call site because the parts of a sentence do not sit in the
 * same order in every language: "{n} objects" is "{n} वस्तुएँ" in Hindi but a template assembled by
 * concatenation in the component would force English word order onto all four. A placeholder the
 * translator can move is the whole point.
 */
export function t(key: string, fallback?: string, vars?: Record<string, string | number>): string {
  const dict = DICTS[getLocale()];
  const raw = dict[key] ?? EN[key] ?? fallback ?? key;
  if (!vars) return raw;
  return raw.replace(/\{(\w+)\}/g, (whole, name) =>
    // An unknown placeholder is left as written rather than blanked: a visible {count} in the interface
    // names the bug, where an empty gap reads as a missing value in the data.
    (name in vars ? String(vars[name]) : whole));
}

/** The placeholders a key expects, for the test that keeps translations from dropping one. */
export function placeholdersOf(text: string): string[] {
  return [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
}

/** Every key, for the test that keeps the dictionaries from drifting apart. */
export function allKeys(): string[] {
  return Object.keys(EN).sort();
}

export function dictFor(locale: Locale): Dict {
  return DICTS[locale];
}
