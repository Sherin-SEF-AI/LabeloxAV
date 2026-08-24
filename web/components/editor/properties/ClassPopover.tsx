"use client";

// The class picker. It used to be a permanently expanded block near the top of the objects tab: a label, a
// swatch, a search box and a scrolling list, about 180px of panel height that an annotator working one
// class for an hour never looked at. It is now behind the header's "change" button, and the header keeps
// the one line that was actually being read, which class you are painting with.
//
// THE NUMBER BADGES MUST NOT LIE. The page relabels on 1-9 by indexing `onto.classes`, the raw ontology
// order, while this list is filtered. The old palette got away with printing `{i + 1}` only because it
// gated the badge on an empty search box. Number a filtered list and every badge is wrong, which is worse
// than no badge: the annotator presses 3 expecting the third row and gets the ontology's third class.

import { useEffect, useRef, useState, type RefObject } from "react";

import { classColor } from "@/lib/colors";
import type { OntologyClass } from "@/lib/types";

import Popover from "./Popover";

// Same normalisation the page uses, so "E Rickshaw" and "e-rickshaw" resolve to the one class rather than
// creating a second.
export const normClass = (s: string) =>
  s.trim().toLowerCase().replace(/[\s-]+/g, "_").replace(/[^a-z0-9_]/g, "");

/** The list is capped: an ontology can run to hundreds and a popover is not a browser. */
const MAX_ROWS = 40;

export default function ClassPopover({
  anchorRef, open, onClose, classes, currentId, onPick, onAdd,
}: {
  anchorRef: RefObject<HTMLElement | null>;
  open: boolean;
  onClose: () => void;
  classes: OntologyClass[];
  currentId: number | null;
  onPick: (c: OntologyClass) => void;
  onAdd: (raw: string) => void;
}) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const norm = normClass(query);
  const filtered = classes.filter((c) => c.name.includes(norm)).slice(0, MAX_ROWS);
  const canAdd = query.trim() !== "" && norm !== "" && !classes.some((c) => c.name === norm);
  // Badges only with an empty query, because only then does row order match ontology order.
  const numbered = query === "";

  useEffect(() => {
    if (!open) return;
    setQuery("");
    // A ref rather than autoFocus: the node stays mounted between opens, and autoFocus only fires on
    // mount, so the second open would leave focus on the trigger.
    inputRef.current?.focus();
  }, [open]);

  const pick = (c: OntologyClass) => { onPick(c); onClose(); };

  // Handled inside the popover and hidden from the page keymap. Returning true is what stops the key.
  const onKey = (e: KeyboardEvent): boolean => {
    if (/^[1-9]$/.test(e.key) && numbered) {
      // Only meaningful while the badges are showing. With a query typed the rows are not the ontology's
      // first nine, and picking by position would contradict the badge that is deliberately not drawn.
      const c = filtered[parseInt(e.key, 10) - 1];
      if (c) { pick(c); return true; }
      return false;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      const rows = [...(listRef.current?.querySelectorAll<HTMLButtonElement>("[data-class-row]") ?? [])];
      if (!rows.length) return false;
      const at = rows.indexOf(document.activeElement as HTMLButtonElement);
      const next = e.key === "ArrowDown"
        ? (at < 0 ? 0 : Math.min(at + 1, rows.length - 1))
        : (at <= 0 ? -1 : at - 1);
      if (next < 0) inputRef.current?.focus();
      else rows[next].focus();
      return true;
    }
    return false;
  };

  return (
    <Popover anchorRef={anchorRef} open={open} onClose={onClose} label="Choose class" onKey={onKey}>
      <div className="p-1.5 border-b hairline">
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="search or add class..."
          aria-label="search or add class"
          onKeyDown={(e) => {
            if (e.key !== "Enter") return;
            // Preserved exactly from the old palette: an exact normalised match relabels, anything else
            // non-empty creates the class and relabels to it.
            const exact = classes.find((c) => c.name === norm);
            if (exact) pick(exact);
            else if (norm) { onAdd(query); onClose(); }
          }}
          className="w-full bg-bg-2 border border-line rounded px-2 py-1 font-mono text-[11px] text-ink placeholder:text-ink-3/70 focus:border-accent outline-none" />
      </div>

      <div ref={listRef} className="max-h-72 overflow-auto p-1 space-y-0.5">
        {canAdd && (
          <button data-class-row onClick={() => { onAdd(query); onClose(); }}
            className="w-full flex items-center gap-1.5 px-1 py-1 rounded font-mono text-[11px] text-left text-accent hover:bg-line/40 hover:text-accent-2">
            <span className="shrink-0">+</span>
            <span className="truncate">add &quot;{norm}&quot; as custom class</span>
          </button>
        )}
        {filtered.map((c, i) => (
          <button key={c.id} data-class-row onClick={() => pick(c)}
            className={`w-full flex items-center gap-1.5 px-1 py-1 rounded font-mono text-[11px] text-left hover:bg-line/40 hover:text-ink ${currentId === c.id ? "text-ink bg-line/25" : "text-ink-3"}`}>
            <span className="w-2.5 h-2.5 inline-block shrink-0 rounded-sm" style={{ background: classColor(c.id) }} />
            <span className="truncate">{c.name}</span>
            {c.india && <span className="text-accent shrink-0" title="India-specific class">*</span>}
            {numbered && i < 9 && <span className="ml-auto text-ink-3 shrink-0">{i + 1}</span>}
          </button>
        ))}
        {!filtered.length && !canAdd && (
          <div className="px-1 py-2 font-mono text-[11px] text-ink-3">no class matches that.</div>
        )}
      </div>
    </Popover>
  );
}
