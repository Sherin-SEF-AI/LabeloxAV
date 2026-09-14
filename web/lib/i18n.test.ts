import { describe, expect, it } from "vitest";

import { LOCALES, allKeys, dictFor, t } from "./i18n";

// A dictionary that drifts is worse than no dictionary: the interface reads as English in exactly the
// places nobody thought to check, and the person who cannot read English is the one who finds out.

describe("interface translations", () => {
  it("every locale covers every key", () => {
    const keys = allKeys();
    expect(keys.length).toBeGreaterThan(20);
    for (const { code } of LOCALES) {
      const dict = dictFor(code);
      const missing = keys.filter((k) => !dict[k]);
      expect(missing, `${code} is missing ${missing.join(", ")}`).toEqual([]);
    }
  });

  it("no locale has a key English does not", () => {
    // English is the source. A key only present in a translation can never render, so it is dead weight
    // that looks like coverage.
    const keys = new Set(allKeys());
    for (const { code } of LOCALES) {
      const extra = Object.keys(dictFor(code)).filter((k) => !keys.has(k));
      expect(extra, `${code} has orphan keys ${extra.join(", ")}`).toEqual([]);
    }
  });

  it("a non-English locale actually differs from English", () => {
    // Guards against a copy-paste that leaves a translation file full of English, which passes a
    // completeness check and helps nobody.
    const en = dictFor("en");
    for (const code of ["hi", "kn", "ta"] as const) {
      const dict = dictFor(code);
      const same = allKeys().filter((k) => dict[k] === en[k]);
      // Product names stay untranslated on purpose, so a handful of exact matches is expected.
      expect(same.length, `${code} duplicates English on ${same.length} keys`).toBeLessThan(4);
    }
  });

  it("an unknown key falls back rather than rendering blank", () => {
    // A half-translated interface should read as partly English, never as `nav.review.queue`.
    expect(t("does.not.exist", "a fallback")).toBe("a fallback");
    expect(t("does.not.exist")).toBe("does.not.exist");
    expect(t("action.accept")).toBeTruthy();
  });
});

// ---- Interpolation and the coverage the editor now depends on (WP6) ------------------------------------

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { placeholdersOf } from "./i18n";
import { EN } from "./locales/en";

describe("interpolation", () => {
  it("substitutes a named placeholder", () => {
    expect(t("frame.objects_count", undefined, { n: 7 })).toBe("7 objects");
  });

  it("leaves an unknown placeholder visible rather than blanking it", () => {
    // A visible {total} names the bug. An empty gap reads as a missing value in the data, which sends
    // whoever sees it looking in the wrong place.
    expect(t("frame.of_session", undefined, { index: 3 })).toContain("{total}");
  });

  it("returns the string untouched when no vars are given", () => {
    expect(t("frame.objects_count")).toBe("{n} objects");
  });

  it("does not interpolate keys that have no placeholders", () => {
    expect(t("action.accept", undefined, { n: 1 })).toBe("accept");
  });
});

describe("every language carries the same message contract", () => {
  it("declares no key the English dictionary does not have", () => {
    // The other direction is allowed: a missing key falls through to English, which is a partly
    // translated interface. A key that exists only in Hindi is a typo nothing would ever surface.
    for (const { code } of LOCALES) {
      for (const key of Object.keys(dictFor(code))) {
        expect(EN[key], `${code} has an orphan key ${key}`).toBeDefined();
      }
    }
  });

  it("keeps every placeholder a translated string is given", () => {
    // A translation that drops {n} renders a sentence with the number missing and no error anywhere.
    for (const { code } of LOCALES) {
      const dict = dictFor(code);
      for (const [key, english] of Object.entries(EN)) {
        if (!(key in dict)) continue;
        expect(placeholdersOf(dict[key]), `${code} ${key}`).toEqual(placeholdersOf(english));
      }
    }
  });

  it("translates the strings an annotator reads while working", () => {
    // The tool strip, the describe tool, the tube verdict and the next-object hint are what a person
    // looks at for hours. Governance surfaces stay English on purpose and are not checked here.
    const worked = Object.keys(EN).filter((k) =>
      k.startsWith("tool.") || k.startsWith("describe.") || k.startsWith("tube.") ||
      k.startsWith("next.") || k.startsWith("action.") || k.startsWith("frame."));
    for (const { code } of LOCALES) {
      if (code === "en") continue;
      const dict = dictFor(code);
      const missing = worked.filter((k) => !(k in dict));
      expect(missing, `${code} is missing working strings`).toEqual([]);
    }
  });
});

describe("the editor does not hardcode a string it has a translation for", () => {
  // The failure this catches is silent: somebody adds a label straight into the component, it renders in
  // English for everyone, and no test or type error notices because the string is perfectly valid code.
  const FILES = [
    "app/frame/[id]/page.tsx",
    "components/shell/ToolStrip.tsx",
  ];
  // Only unambiguous, whole-string labels. A word like "class" or "next" appears in code and in prose for
  // reasons that have nothing to do with a label, and asserting on those would make this test noise.
  const CHECKED = ["unsaved changes", "pick a label first", "confirm frame", "nothing in the queue"];

  it("renders those labels through t() or not at all", () => {
    for (const rel of FILES) {
      const src = readFileSync(join(process.cwd(), rel), "utf8");
      for (const literal of CHECKED) {
        const quoted = new RegExp(`["'\`]${literal}["'\`]`);
        const translated = new RegExp(`t\\(\\s*["'][^"']+["'][^)]*${literal}`);
        if (quoted.test(src) && !translated.test(src)) {
          throw new Error(`${rel} hardcodes "${literal}"; render it through t()`);
        }
      }
    }
  });
});
