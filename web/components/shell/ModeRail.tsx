"use client";

import { MODES } from "@/lib/editor/registry";
import Icon, { MODE_ICON } from "@/components/shell/Icon";

// The fixed-width left rail of modes (the design's 58px rail): one icon + short label + switch key per mode,
// always the same width no matter how many tools a mode owns. Switching mode swaps the tool strip, the
// default panel, and the canvas. A new mode is one entry in the registry, one button here, zero layout cost.

export default function ModeRail({ mode, onMode }: { mode: string; onMode: (key: string) => void }) {
  return (
    <div className="w-[58px] shrink-0 flex flex-col items-center py-2 gap-[3px] border-r hairline bg-bg">
      {MODES.map((m) => {
        const active = mode === m.key;
        const short = m.label.split(" ")[0];
        return (
          <button key={m.key} onClick={() => onMode(m.key)} title={`${m.label} (Shift ${m.hotkey})`}
            className={`relative w-[50px] flex flex-col items-center gap-1 pt-2 pb-1.5 rounded-md ${active ? "text-accent bg-accent/10" : "text-ink-3 hover:text-ink-2 hover:bg-line/40"}`}>
            {active && <span className="absolute left-0 top-1 bottom-1 w-0.5 rounded-full bg-accent" />}
            <Icon name={MODE_ICON[m.key] ?? "box"} size={19} />
            <span className="font-display text-[8.5px] font-medium tracking-wide">{short}</span>
            <span className="absolute top-1.5 right-1.5 font-mono text-[8px] text-ink-3/70">{m.hotkey}</span>
          </button>
        );
      })}
    </div>
  );
}
