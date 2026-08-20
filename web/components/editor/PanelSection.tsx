"use client";

// A collapsible section for the editor's right rail. Gives every tool cluster the same header (title + optional
// right-aligned badge + a chevron) and lets the advanced ones default-collapse, so the panel reads as a tidy
// accordion instead of one long wall of expanded controls.


import { prefSection, usePanelFlag } from "./properties/panelPrefs";

export default function PanelSection({ title, badge, defaultOpen = false, accent = false, storageKey, children }: {
  title: string;
  badge?: React.ReactNode;
  defaultOpen?: boolean;
  accent?: boolean;
  /** Opt in to remembering open/closed across reloads. Omit and this behaves exactly as it always has. */
  storageKey?: string;
  children: React.ReactNode;
}) {
  // A null key makes this a plain useState, so a section without a storageKey behaves exactly as it did
  // before and touches localStorage not at all. Branching on the prop at the call site instead would call
  // a different number of hooks per render.
  const [open, setOpen] = usePanelFlag(storageKey ? prefSection(storageKey) : null, defaultOpen);
  return (
    <div className="border-b hairline">
      <button onClick={() => setOpen(!open)} aria-expanded={open}
        className="w-full flex items-center justify-between px-2 py-1.5 hover:bg-line/30 group transition-colors">
        <span className={`font-mono text-[10px] uppercase tracking-wide ${accent ? "text-ink-2" : "text-ink-3"}`}>{title}</span>
        <span className="flex items-center gap-2">
          {badge != null && <span className="font-mono text-[10px] text-ink-3">{badge}</span>}
          <span className={`text-ink-3 group-hover:text-ink w-3 text-center text-[9px] leading-none transition-transform duration-200 ${open ? "rotate-90" : ""}`}>▸</span>
        </span>
      </button>
      {open && <div className="px-2 pb-2 reveal">{children}</div>}
    </div>
  );
}
