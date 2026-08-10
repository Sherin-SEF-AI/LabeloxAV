"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { ToolGroup } from "@/lib/editor/registry";
import Icon, { TOOL_ICON } from "@/components/shell/Icon";
import { fitCount } from "@/lib/editor/toolStripFit";

// The contextual tool strip. Renders the current mode (icon + label) then one button per GROUP, not per
// tool: a single-tool group is a direct button, a multi-tool group is one button showing the active tool
// plus a caret that opens a flyout of the alternates. This is the mechanism that keeps the strip a single
// row forever, no matter how many tools a mode accumulates. The active tool's group is the only accent.
//
// When even the groups do not fit, the tail collapses into one overflow button rather than scrolling out of
// view. The row was `overflow-x-auto`, which honoured "one row" but not "no clip": at 1280px, a 13in laptop,
// `measure` simply was not there, with no scrollbar and no chevron to suggest otherwise. Its hotkey still
// worked, which made it worse rather than better, because pressing R highlighted a button nobody could see.
// Collapsing reuses the flyout the strip already has for alternates, one level up.
//
// The flyout is rendered through a portal at a measured screen position, NOT as an absolutely-positioned
// child: the strip lives inside a clipped row, and a non-visible overflow-x clips overflow-y as well, so an
// in-flow dropdown would be invisible (and the buttons would feel dead to a mouse).

const OVERFLOW_KEY = "__overflow__";

export default function ToolStrip({ groups, tool, onSelect, options, modeIcon, modeLabel }: {
  groups: ToolGroup[];
  tool: string;
  onSelect: (toolKey: string) => void;
  options?: React.ReactNode;
  modeIcon?: string;
  modeLabel?: string;
}) {
  const [open, setOpen] = useState<{ key: string; x: number; y: number } | null>(null);
  const rowRef = useRef<HTMLDivElement | null>(null);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [widths, setWidths] = useState<number[]>([]);
  const [avail, setAvail] = useState(0);

  const signature = groups.map((g) => g.key).join("|");

  // A mode switch replaces the groups, so last mode's widths say nothing about this one. Clearing forces a
  // measure pass in which every group is rendered, which is the only moment their widths are observable.
  useLayoutEffect(() => {
    setWidths([]);
    setOpen(null);
  }, [signature]);

  // Group widths are not constant across widths: the labels and hotkey chips are `lg:` only, so crossing
  // 1024px changes every group from ~110px to ~30px. Measuring once would then evict groups that had since
  // become small enough to fit. Re-measuring on resize costs one unpainted pass and keeps the two in step.
  useLayoutEffect(() => {
    setWidths([]);
  }, [avail]);

  const measuring = widths.length !== groups.length;

  useLayoutEffect(() => {
    if (!measuring) return;
    const els = itemRefs.current.slice(0, groups.length);
    if (els.length !== groups.length || els.some((e) => !e)) return;
    const next = els.map((e) => e!.getBoundingClientRect().width);
    // A width of zero means the row is display:none (a background tab, a collapsed panel). Committing those
    // would collapse the strip to an overflow button and leave it that way once the tab came back.
    if (next.some((w) => w <= 0)) return;
    setWidths(next);
  }, [measuring, signature, groups.length]);

  useEffect(() => {
    const el = rowRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setAvail(el.clientWidth));
    ro.observe(el);
    setAvail(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  // While measuring, show everything: the overflow is one unpainted frame under useLayoutEffect, whereas
  // guessing would flash a collapsed strip on every mode switch.
  const shownCount = measuring ? groups.length : fitCount(widths, avail);
  const visible = groups.slice(0, shownCount);
  const hidden = groups.slice(shownCount);
  const hiddenHoldsActive = hidden.some((g) => g.tools.some((t) => t.key === tool));

  const openAt = useCallback((key: string, el: HTMLElement) => {
    const r = el.getBoundingClientRect();
    setOpen((cur) => (cur?.key === key ? null : { key, x: r.left, y: r.bottom + 4 }));
  }, []);

  const openGroup = open && open.key !== OVERFLOW_KEY ? groups.find((g) => g.key === open.key) : null;
  const flyoutTools = open?.key === OVERFLOW_KEY ? hidden.flatMap((g) => g.tools) : openGroup?.tools ?? [];
  const flyoutLabel = open?.key === OVERFLOW_KEY ? "more tools" : openGroup?.label ?? "";

  return (
    <div className="flex items-center gap-1.5 min-w-0 flex-1">
      {modeLabel && (
        <div className="flex items-center gap-1.5 h-[30px] pr-3 mr-0.5 border-r hairline shrink-0">
          <span className="flex text-accent"><Icon name={modeIcon ?? "box"} size={16} /></span>
          <span className="hidden lg:inline font-display font-semibold text-[12.5px] text-ink">{modeLabel}</span>
        </div>
      )}
      <div ref={rowRef} className="flex items-center gap-1.5 min-w-0 flex-1 overflow-hidden">
        {visible.map((g, i) => {
          const active = g.tools.find((t) => t.key === tool);
          const single = g.tools.length === 1;
          const shown = active ?? g.tools[0];
          const on = !!active;
          return (
            <div key={g.key} ref={(el) => { itemRefs.current[i] = el; }}
              className="relative shrink-0 flex items-center">
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
                  onClick={(e) => openAt(g.key, e.currentTarget.parentElement as HTMLElement)}
                  className={`flex items-center h-8 pl-0.5 pr-1 rounded-md ${open?.key === g.key ? "text-accent" : "text-ink-3 hover:text-ink-2"}`}>
                  <Icon name="chevD" size={13} />
                </button>
              )}
            </div>
          );
        })}
        {hidden.length > 0 && (
          <button
            aria-label={`${hidden.length} more tools`}
            title={`${hidden.length} more tools: ${hidden.map((g) => g.label).join(", ")}`}
            onClick={(e) => openAt(OVERFLOW_KEY, e.currentTarget)}
            // Accented when the active tool lives in here, so a narrow window never shows a strip with no
            // tool selected anywhere while one plainly is.
            className={`flex items-center gap-0.5 h-8 px-1.5 shrink-0 rounded-md border ${hiddenHoldsActive || open?.key === OVERFLOW_KEY ? "border-accent/40 bg-accent/10 text-accent" : "border-transparent text-ink-3 hover:bg-line/40 hover:text-ink-2"}`}>
            <Icon name="more" size={16} />
            <span className="font-mono text-[9px] leading-none">{hidden.length}</span>
          </button>
        )}
      </div>
      {options && <div className="flex items-center gap-1 ml-1 pl-2 border-l hairline shrink-0">{options}</div>}
      {open && flyoutTools.length > 0 && typeof document !== "undefined" && createPortal(
        <>
          <div className="fixed inset-0 z-[60]" onClick={() => setOpen(null)} />
          <div className="fixed z-[61] min-w-[204px] panel p-1.5" style={{ left: open.x, top: open.y }}>
            <div className="flex items-center gap-1.5 px-2 pt-1 pb-1.5">
              <span className="font-display text-[10px] font-semibold uppercase tracking-wider text-ink-3">{flyoutLabel}</span>
              <span className="ml-auto font-mono text-[9px] text-ink-3/70">cycle</span>
            </div>
            {flyoutTools.map((t) => (
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
