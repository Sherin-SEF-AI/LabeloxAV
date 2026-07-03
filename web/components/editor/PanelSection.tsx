"use client";

// A collapsible section for the editor's right rail. Gives every tool cluster the same header (title + optional
// right-aligned badge + a chevron) and lets the advanced ones default-collapse, so the panel reads as a tidy
// accordion instead of one long wall of expanded controls.

import { useState } from "react";

export default function PanelSection({ title, badge, defaultOpen = false, accent = false, children }: {
  title: string;
  badge?: React.ReactNode;
  defaultOpen?: boolean;
  accent?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b hairline">
      <button onClick={() => setOpen((o) => !o)}
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
