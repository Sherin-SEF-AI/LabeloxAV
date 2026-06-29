"use client";

import { useState } from "react";
import type { ToolGroup } from "@/lib/editor/registry";
import Icon, { TOOL_ICON } from "@/components/shell/Icon";

// The contextual tool strip. Renders the current mode (icon + label) then one button per GROUP, not per
// tool: a single-tool group is a direct button, a multi-tool group collapses to one button showing the
// active tool with the alternates in a flyout. This is the mechanism that keeps the strip a single row
// forever, no matter how many tools a mode accumulates. The active tool's group is the only accent. A
// contextual options slot holds the active tool's parameters so they never become extra header buttons.

export default function ToolStrip({ groups, tool, onSelect, options, modeIcon, modeLabel }: {
  groups: ToolGroup[];
  tool: string;
  onSelect: (toolKey: string) => void;
  options?: React.ReactNode;
  modeIcon?: string;
  modeLabel?: string;
}) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="flex items-center gap-1.5 min-w-0">
      {modeLabel && (
        <div className="flex items-center gap-1.5 h-[30px] pr-3 mr-0.5 border-r hairline shrink-0">
          <span className="flex text-accent"><Icon name={modeIcon ?? "box"} size={16} /></span>
          <span className="font-display font-semibold text-[12.5px] text-ink">{modeLabel}</span>
        </div>
      )}
      {groups.map((g) => {
        const active = g.tools.find((t) => t.key === tool);
        const single = g.tools.length === 1;
        const shown = active ?? g.tools[0];
        const on = !!active;
        return (
          <div key={g.key} className="relative shrink-0">
            <button
              onClick={() => (single ? onSelect(g.tools[0].key) : setOpen(open === g.key ? null : g.key))}
              title={`${g.label} (${shown.hotkey})`}
              className={`flex items-center gap-1.5 h-8 px-2.5 rounded-md border ${on ? "border-accent/40 bg-accent/10 text-accent" : "border-transparent text-ink-2 hover:bg-line/40"}`}>
              <span className="flex"><Icon name={TOOL_ICON[shown.key] ?? "dot"} size={16} /></span>
              <span className="font-body text-[12px]">{single ? g.tools[0].label : active ? active.label : g.label}</span>
              <span className={`font-mono text-[9px] leading-none px-1 py-0.5 rounded border ${on ? "border-accent/30" : "border-line text-ink-3"}`}>{shown.hotkey}</span>
              {!single && <span className="flex text-ink-3"><Icon name="chevD" size={13} /></span>}
            </button>
            {!single && open === g.key && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setOpen(null)} />
                <div className="absolute left-0 top-[42px] z-50 min-w-[204px] panel p-1.5">
                  <div className="flex items-center gap-1.5 px-2 pt-1 pb-1.5">
                    <span className="font-display text-[10px] font-semibold uppercase tracking-wider text-ink-3">{g.label}</span>
                    <span className="ml-auto font-mono text-[9px] text-ink-3/70">cycle</span>
                  </div>
                  {g.tools.map((t) => (
                    <button key={t.key} onClick={() => { onSelect(t.key); setOpen(null); }}
                      className={`flex w-full items-center gap-2 px-2 py-1.5 rounded ${t.key === tool ? "text-accent bg-accent/10" : "text-ink-2 hover:bg-line/50"}`}>
                      <span className="flex"><Icon name={TOOL_ICON[t.key] ?? "dot"} size={15} /></span>
                      <span className="flex-1 text-left font-body text-[12px]">{t.label}</span>
                      <span className="font-mono text-[10px] text-ink-3 min-w-[14px] text-center px-1 py-0.5 rounded border border-line bg-bg-2">{t.hotkey}</span>
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
