"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import type { ToolGroup } from "@/lib/editor/registry";
import Icon, { TOOL_ICON } from "@/components/shell/Icon";

// The contextual tool strip. Renders the current mode (icon + label) then one button per GROUP, not per
// tool: a single-tool group is a direct button, a multi-tool group is one button showing the active tool
// plus a caret that opens a flyout of the alternates. This is the mechanism that keeps the strip a single
// row forever, no matter how many tools a mode accumulates. The active tool's group is the only accent.
//
// The flyout is rendered through a portal at a measured screen position, NOT as an absolutely-positioned
// child: the strip lives inside an `overflow-x-auto` row, and a non-visible overflow-x clips overflow-y as
// well, so an in-flow dropdown would be invisible (and the buttons would feel dead to a mouse).

export default function ToolStrip({ groups, tool, onSelect, options, modeIcon, modeLabel }: {
  groups: ToolGroup[];
  tool: string;
  onSelect: (toolKey: string) => void;
  options?: React.ReactNode;
  modeIcon?: string;
  modeLabel?: string;
}) {
  const [open, setOpen] = useState<{ key: string; x: number; y: number } | null>(null);
  const openGroup = open && groups.find((g) => g.key === open.key);
  return (
    <div className="flex items-center gap-1.5 min-w-0">
      {modeLabel && (
        <div className="flex items-center gap-1.5 h-[30px] pr-3 mr-0.5 border-r hairline shrink-0">
          <span className="flex text-accent"><Icon name={modeIcon ?? "box"} size={16} /></span>
          <span className="hidden lg:inline font-display font-semibold text-[12.5px] text-ink">{modeLabel}</span>
        </div>
      )}
      {groups.map((g) => {
        const active = g.tools.find((t) => t.key === tool);
        const single = g.tools.length === 1;
        const shown = active ?? g.tools[0];
        const on = !!active;
        return (
          <div key={g.key} className="relative shrink-0 flex items-center">
            <button
              onClick={() => { onSelect(shown.key); setOpen(null); }}
              title={`${g.label} (${shown.hotkey})`}
              className={`flex items-center gap-1.5 h-8 pl-2.5 ${single ? "pr-2.5" : "pr-1.5"} rounded-md border ${on ? "border-accent/40 bg-accent/10 text-accent" : "border-transparent text-ink-2 hover:bg-line/40"}`}>
              <span className="flex"><Icon name={TOOL_ICON[shown.key] ?? "dot"} size={16} /></span>
              <span className="hidden lg:inline font-body text-[12px]">{single ? g.tools[0].label : active ? active.label : g.label}</span>
              <span className={`hidden lg:inline-block font-mono text-[9px] leading-none px-1 py-0.5 rounded border ${on ? "border-accent/30" : "border-line text-ink-3"}`}>{shown.hotkey}</span>
            </button>
            {!single && (
              <button
                aria-label={`${g.label} tools`}
                title={`${g.label} tools`}
                onClick={(e) => {
                  const r = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
                  setOpen(open?.key === g.key ? null : { key: g.key, x: r.left, y: r.bottom + 4 });
                }}
                className={`flex items-center h-8 pl-0.5 pr-1 rounded-md ${open?.key === g.key ? "text-accent" : "text-ink-3 hover:text-ink-2"}`}>
                <Icon name="chevD" size={13} />
              </button>
            )}
          </div>
        );
      })}
      {options && <div className="flex items-center gap-1 ml-1 pl-2 border-l hairline shrink-0">{options}</div>}
      {open && openGroup && typeof document !== "undefined" && createPortal(
        <>
          <div className="fixed inset-0 z-[60]" onClick={() => setOpen(null)} />
          <div className="fixed z-[61] min-w-[204px] panel p-1.5" style={{ left: open.x, top: open.y }}>
            <div className="flex items-center gap-1.5 px-2 pt-1 pb-1.5">
              <span className="font-display text-[10px] font-semibold uppercase tracking-wider text-ink-3">{openGroup.label}</span>
              <span className="ml-auto font-mono text-[9px] text-ink-3/70">cycle</span>
            </div>
            {openGroup.tools.map((t) => (
              <button key={t.key} onClick={() => { onSelect(t.key); setOpen(null); }}
                className={`flex w-full items-center gap-2 px-2 py-1.5 rounded ${t.key === tool ? "text-accent bg-accent/10" : "text-ink-2 hover:bg-line/50"}`}>
                <span className="flex"><Icon name={TOOL_ICON[t.key] ?? "dot"} size={15} /></span>
                <span className="flex-1 text-left font-body text-[12px]">{t.label}</span>
                <span className="font-mono text-[10px] text-ink-3 min-w-[14px] text-center px-1 py-0.5 rounded border border-line bg-bg-2">{t.hotkey}</span>
              </button>
            ))}
          </div>
        </>,
        document.body,
      )}
    </div>
  );
}
