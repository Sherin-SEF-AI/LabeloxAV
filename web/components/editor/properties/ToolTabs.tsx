"use client";

// The tool tab strip. The panel's old two-tab strip was six lines of inline JSX with no role, no
// aria-selected, no aria-controls and no key handling, so to a screen reader it was three unlabelled
// buttons and to the keyboard it was three separate tab stops. This is the standard pattern instead.
//
// Roving tabindex matters more here than it looks: without it, Tab from the search box lands on every tab
// in turn before reaching the panel below, which on a rail people traverse all day is three extra stops
// for no information.

import { useRef } from "react";

export type Tab<T extends string> = { key: T; label: string };

export default function ToolTabs<T extends string>({ tabs, value, onChange, idPrefix, label }: {
  tabs: Tab<T>[];
  value: T;
  onChange: (v: T) => void;
  /** Ties each tab to its panel through aria-controls / aria-labelledby. */
  idPrefix: string;
  label: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);

  const move = (delta: number | "first" | "last") => {
    const at = tabs.findIndex((t) => t.key === value);
    const next = delta === "first" ? 0
      : delta === "last" ? tabs.length - 1
      // Wraps, because a three-tab strip that stops at the end makes the user reverse direction to reach
      // the tab one step the other way.
      : (at + delta + tabs.length) % tabs.length;
    onChange(tabs[next].key);
    // Activation follows focus, which is the standard for a small strip with nothing expensive behind it.
    // Focus has to be moved explicitly or the arrow keys change the panel while the old tab keeps the ring.
    ref.current?.querySelectorAll<HTMLButtonElement>("[role=tab]")[next]?.focus();
  };

  return (
    <div ref={ref} role="tablist" aria-label={label}
      className="flex shrink-0 border-b hairline font-mono text-[10px] uppercase tracking-wide"
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") { e.preventDefault(); move(1); }
        else if (e.key === "ArrowLeft") { e.preventDefault(); move(-1); }
        else if (e.key === "Home") { e.preventDefault(); move("first"); }
        else if (e.key === "End") { e.preventDefault(); move("last"); }
      }}>
      {tabs.map((t) => {
        const on = t.key === value;
        return (
          <button key={t.key} role="tab" type="button"
            id={`${idPrefix}-tab-${t.key}`}
            aria-selected={on}
            aria-controls={`${idPrefix}-panel-${t.key}`}
            tabIndex={on ? 0 : -1}
            onClick={() => onChange(t.key)}
            className={`flex-1 py-2 transition-colors ${on ? "text-accent border-b-2 border-accent -mb-px" : "text-ink-3 hover:text-ink-2"}`}>
            {t.label}
          </button>
        );
      })}
    </div>
  );
}
