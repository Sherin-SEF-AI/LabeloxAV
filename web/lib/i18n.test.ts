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
