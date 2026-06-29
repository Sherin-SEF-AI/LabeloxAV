"use client";

import { useState } from "react";

// A collapsible side rail for list-plus-detail routes. Header with a title and a collapse toggle (the same
// open/close idiom as FloatingLayers), a scrollable body, and a fixed width that collapses to a thin spine
// with the title turned vertical. Opt-in: a page keeps its existing detail JSX and just moves it into the
// children, gaining consistent width, border, collapse, and scroll. The canvas/content keeps the rest of
// the width, and collapsing gives it all of it.

export default function Inspector({ title, side = "right", defaultOpen = true, width = "w-72", meta, footer, children }: {
  title: string;
  side?: "left" | "right";
  defaultOpen?: boolean;
  width?: string;
  meta?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const border = side === "right" ? "border-l" : "border-r";

  if (!open) {
    return (
      <div className={`${border} hairline shrink-0 w-8 flex flex-col items-center py-2`}>
        <button onClick={() => setOpen(true)} title={`show ${title}`}
          className="font-mono text-[10px] uppercase tracking-wide text-ink-3 hover:text-accent [writing-mode:vertical-rl]">
          {title}
        </button>
      </div>
    );
  }

  return (
    <aside className={`${border} hairline shrink-0 ${width} flex flex-col min-h-0`}>
      <div className="flex items-center justify-between gap-2 px-2 h-9 border-b hairline shrink-0">
        <span className="font-mono text-[10px] uppercase tracking-wide text-ink-3 truncate">{title}</span>
        <div className="flex items-center gap-2 shrink-0">
          {meta && <span className="font-mono text-[10px] text-ink-3">{meta}</span>}
          <button onClick={() => setOpen(false)} title="collapse"
            className="font-mono text-[11px] text-ink-3 hover:text-accent">{side === "right" ? ">" : "<"}</button>
        </div>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto">{children}</div>
      {footer && <div className="border-t hairline shrink-0 p-2">{footer}</div>}
    </aside>
  );
}
