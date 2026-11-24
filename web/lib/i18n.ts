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

import { useCallback, useSyncExternalStore } from "react";

import { LOCALE_STORAGE_KEY } from "./bootScript";
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

// Shared with the inline script in app/layout.tsx that sets <html lang> before first paint.
const KEY = LOCALE_STORAGE_KEY;

// The dictionaries live in lib/locales/*.ts, one file per language. They were inline here and the file
// grew by four entries every time a single string was translated, which made every review of this module
// a review of four unrelated languages.
const DICTS: Record<Locale, Dict> = { en: EN, hi: HI, kn: KN, ta: TA };

let _locale: Locale | null = null;

export function getLocale(): Locale {
  if (_locale) return _locale;
  if (typeof window === "undefined") return "en";
  // Storage can be missing or throw (a private window, blocked site data); the browser's own preference
  // is then the answer, not a crash in whatever happened to render a string first.
  let stored: Locale | null = null;
  try { stored = localStorage.getItem(KEY) as Locale | null; } catch { stored = null; }
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
    try { localStorage.setItem(KEY, locale); } catch { /* holds for this page view only */ }
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
  return translate(getLocale(), key, fallback, vars);
}

/** `t` for a named locale. Pure, so a hook can bind it to the locale it rendered with. */
export function translate(locale: Locale, key: string, fallback?: string,
                          vars?: Record<string, string | number>): string {
  const dict = DICTS[locale] ?? EN;
  const raw = dict[key] ?? EN[key] ?? fallback ?? key;
  if (!vars) return raw;
  return raw.replace(/\{(\w+)\}/g, (whole, name) =>
    // An unknown placeholder is left as written rather than blanked: a visible {count} in the interface
    // names the bug, where an empty gap reads as a missing value in the data.
    (name in vars ? String(vars[name]) : whole));
}

// ---- Reading the locale during render -----------------------------------------------------------------
//
// `t()` reads localStorage, which the server does not have. Called during render it therefore returns
// English on the server and the stored language in the browser, and React hydrates a Hindi tree against
// English HTML: a mismatch on every string, on every page, for exactly the people the translations are
// for. That was survivable with nine call sites inside components that only render after mount. It is
// not survivable for a sidebar.
//
// So anything rendered goes through `useT()` (or `<T>`), which reads the locale with
// useSyncExternalStore: English for the server render and for hydration, then the stored language in the
// same commit, before the browser paints the hydrated tree. `t()` stays for what runs outside render:
// toasts, event handlers, and the editor's existing calls.

export type Translate = (key: string, fallback?: string, vars?: Record<string, string | number>) => string;

function subscribe(onChange: () => void): () => void {
  // Another tab changing the language. This tab's own change reloads the page (see setLocale).
  const onStorage = (e: StorageEvent) => {
    if (e.key !== KEY) return;
    _locale = null;
    onChange();
  };
  window.addEventListener("storage", onStorage);
  return () => window.removeEventListener("storage", onStorage);
}

const serverLocale = (): Locale => "en";

/** The interface language, safe to read during render. */
export function useLocale(): Locale {
  return useSyncExternalStore(subscribe, getLocale, serverLocale);
}

/** `t`, bound to the locale this component rendered with. */
export function useT(): Translate {
  const locale = useLocale();
  return useCallback<Translate>((key, fallback, vars) => translate(locale, key, fallback, vars), [locale]);
}

/**
 * A translated string as an element: `<T k="common.cancel">Cancel</T>`. The children are the English
 * text, kept at the call site so the JSX still reads as what it renders. For text content only; an
 * attribute (placeholder, aria-label) needs `useT()`.
 */
export function T({ k, children, vars }: { k: string; children?: string; vars?: Record<string, string | number> }) {
  return useT()(k, children, vars);
}

// ---- Navigation labels --------------------------------------------------------------------------------
//
// The sidebar and the menus are data (lib/sidebar.ts, lib/menus.ts) with English labels, and the label is
// also what the palette ranks on and what tests look up. Rather than give every entry a second field that
// can drift from the first, the key is derived from the label: "My queue" is `shell.my_queue`. A test
// walks both files and fails when a label has no key, so adding an entry without translating it is a red
// build rather than an English row in a Tamil sidebar.

/** Names that are the same in every language. They have no key, and the test checks they do not grow one. */
export const UNTRANSLATED_LABELS: ReadonlySet<string> = new Set(["LiDAR"]);

export function labelKey(label: string): string {
  return `shell.${label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "")}`;
}

/** A navigation label in the given locale. English returns the label as written, capitalisation included. */
export function translateLabel(locale: Locale, label: string): string {
  if (locale === "en" || UNTRANSLATED_LABELS.has(label)) return label;
  return translate(locale, labelKey(label), label);
}

/** `translateLabel`, bound to the locale this component rendered with. */
export function useLabel(): (label: string) => string {
  const locale = useLocale();
  return useCallback((label: string) => translateLabel(locale, label), [locale]);
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
