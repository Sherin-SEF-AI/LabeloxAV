import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

// A button whose whole content is an icon or a glyph has no name. A sighted user reads the picture; a
// screen reader says "button". `title` does not fix it: it is a tooltip, announced inconsistently and
// never shown to a keyboard. eslint's control-has-associated-label catches the empty <button />, and
// cannot see that <button><Icon name="trash" /></button> or <button>x</button> is the same problem. This
// scan can.
//
// Shaped after nofetch.test.ts and noswallow.test.ts: a source scan with an explicit scope, so the number
// cannot drift back up one "×" at a time.

const WEB_ROOT = resolve(__dirname, "..");
const SCAN_DIRS = ["app", "components", "platforms"];
// The frame editor is being rewritten and is out of this guard's reach until it lands.
const SKIP = [/^app\/frame\//, /^components\/editor\//];

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry === ".next") continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, out);
    else if (/\.tsx$/.test(entry) && !/\.test\.tsx$/.test(entry)) out.push(full);
  }
  return out;
}

/** Index of the `>` that closes the opening tag starting at `from`, skipping braces and strings. */
function openTagEnd(src: string, from: number): number {
  let depth = 0;
  let quote: string | null = null;
  for (let i = from; i < src.length; i++) {
    const c = src[i];
    if (quote) { if (c === quote) quote = null; continue; }
    if (c === '"' || c === "'" || c === "`") quote = c;
    else if (c === "{") depth++;
    else if (c === "}") depth--;
    else if (c === ">" && depth === 0 && src[i - 1] !== "=") return i;
  }
  return -1;
}

/** Buttons with no word in them and no aria-label, as `file:line: content`. */
export function unnamedButtons(rel: string, src: string): string[] {
  const out: string[] = [];
  for (const m of src.matchAll(/<button\b/g)) {
    const end = openTagEnd(src, m.index);
    if (end < 0) continue;
    const tag = src.slice(m.index, end + 1);
    if (/aria-label(ledby)?=/.test(tag)) continue;
    // A spread may carry the label; that cannot be judged from here.
    if (/\{\.\.\./.test(tag)) continue;
    const close = tag.endsWith("/>") ? end : src.indexOf("</button>", end);
    const body = tag.endsWith("/>") ? "" : src.slice(end + 1, close);
    const text = body
      .replace(/\{\/\*[\s\S]*?\*\/\}/g, "")        // JSX comments
      .replace(/<[A-Za-z][^<>]*?\/>/g, "")          // <Icon ... />
      .replace(/<\/?[A-Za-z][^<>]*?>/g, "");        // wrappers, keeping what is inside them
    // An expression may render text: a variable ({label}, {c}) or a string with a word in it. One whose
    // only strings are glyphs ({"|<"}, {open ? "<" : ">"}) renders a glyph whichever branch is taken.
    const exprs = [...text.matchAll(/\{([^{}]*)\}/g)].map((x) => x[1]);
    const mayRenderText = exprs.some((x) => {
      const strings = x.match(/"[^"]*"|'[^']*'/g) ?? [];
      if (strings.some((str) => /[A-Za-z]{2,}/.test(str))) return true;
      // Exactly a literal, or a ternary between two literals: whatever it renders is one of those glyphs.
      const literal = String.raw`(?:"[^"]*"|'[^']*')`;
      if (new RegExp(String.raw`^\s*(?:[^?]*\?\s*${literal}\s*:\s*)?${literal}\s*$`).test(x)) return false;
      return /[A-Za-z_]/.test(x);
    });
    const words = text.replace(/\{[^{}]*\}/g, "").replace(/&[a-z]+;/g, "").match(/[A-Za-z]{2,}/g);
    if (!words && !mayRenderText) {
      const line = src.slice(0, m.index).split("\n").length;
      out.push(`${rel}:${line}: ${body.replace(/\s+/g, " ").trim().slice(0, 60) || "(empty)"}`);
    }
  }
  return out;
}

describe("icon-only buttons have a name", () => {
  it("recognises the cases it exists for", () => {
    const src = [
      '<button onClick={a}><Icon name="trash" size={12} /></button>',
      '<button onClick={a} title="close">x</button>',
      "<button onClick={a}>&minus;</button>",
      '<button onClick={a}>{"|<"}</button>',
      '<button onClick={a} aria-label="Remove"><Icon name="trash" /></button>',
      "<button onClick={a}>Save</button>",
      "<button onClick={a}>{label}</button>",
      "<button onClick={a}>{c}</button>",
      '<button onClick={a}>{open ? "<" : ">"}</button>',
      '<button onClick={a}><Icon name="plus" /> Add</button>',
      "<button {...pressable(a)}><Icon name=\"plus\" /></button>",
    ].join("\n");
    const found = unnamedButtons("x.tsx", src);
    expect(found.map((f) => f.split(":")[1])).toEqual(["1", "2", "3", "4", "9"]);
  });

  it("scans a non-trivial number of files", () => {
    let n = 0;
    for (const dir of SCAN_DIRS) {
      try { n += walk(join(WEB_ROOT, dir)).length; } catch { /* directory moved */ }
    }
    expect(n, "the source scan found almost nothing; SCAN_DIRS has drifted").toBeGreaterThan(150);
  });

  it("no button outside the editor is a bare icon or glyph without aria-label", () => {
    const offenders: string[] = [];
    for (const dir of SCAN_DIRS) {
      let files: string[];
      try { files = walk(join(WEB_ROOT, dir)); } catch { continue; }
      for (const file of files) {
        const rel = file.slice(WEB_ROOT.length + 1);
        if (SKIP.some((re) => re.test(rel))) continue;
        offenders.push(...unnamedButtons(rel, readFileSync(file, "utf8")));
      }
    }
    expect(offenders, "give each an aria-label that says what it does (title alone is a tooltip, not a name)").toEqual([]);
  });
});
