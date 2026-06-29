"use client";

import { useState } from "react";
import type { ToolGroup } from "@/lib/editor/registry";

// The contextual tool strip. Renders one button per GROUP, not per tool: a single-tool group is a direct
// button, a multi-tool group collapses to one button showing the active tool with the alternates in a
// flyout. This is the mechanism that keeps the strip a single row forever, no matter how many tools a
// mode accumulates. The active tool's group is the only accent. A contextual options slot holds the
// active tool's parameters so they never become extra header buttons.

export default function ToolStrip({ groups, tool, onSelect, options }: {
  groups: ToolGroup[];
  tool: string;
  onSelect: (toolKey: string) => void;
  options?: React.ReactNode;
}) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="flex items-center gap-1 font-mono text-[11px] min-w-0">
      {groups.map((g) => {
        const active = g.tools.find((t) => t.key === tool);
        const single = g.tools.length === 1;
        const shown = active ?? g.tools[0];
        return (
          <div key={g.key} className="relative shrink-0">
            <button
              onClick={() => (single ? onSelect(g.tools[0].key) : setOpen(open === g.key ? null : g.key))}
              title={`${g.label} (${shown.hotkey})`}
              className={`px-2 py-1 border ${active ? "border-accent text-ink" : "border-line text-ink-3 hover:border-ink-3"}`}>
              {single ? g.tools[0].label : active ? active.label : g.label}
              {!single && <span className="text-ink-3"> &#9662;</span>}
            </button>
            {!single && open === g.key && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setOpen(null)} />
                <div className="absolute left-0 mt-1 z-50 panel p-1 min-w-[9rem]">
                  {g.tools.map((t) => (
                    <button key={t.key} onClick={() => { onSelect(t.key); setOpen(null); }}
                      className={`flex w-full items-center justify-between gap-4 px-2 py-1 hover:bg-bg-2 ${t.key === tool ? "text-accent" : "text-ink-2"}`}>
                      <span>{t.label}</span><span className="text-ink-3">{t.hotkey}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        );
      })}
      {options && <div className="flex items-center gap-1 ml-1 pl-2 border-l hairline shrink-0">{options}</div>}
    </div>
  );
}
