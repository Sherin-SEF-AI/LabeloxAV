"use client";

import { MODES } from "@/lib/editor/registry";

// The fixed-width left rail of modes. This is the top-level toolset switcher: one icon per mode, always
// the same width no matter how many tools a mode owns. Switching mode swaps the tool strip and the
// default panel (and, in 3b, the canvas). A new mode is one entry in the registry, one button here.

export default function ModeRail({ mode, onMode }: { mode: string; onMode: (key: string) => void }) {
  return (
    <div className="flex flex-col items-stretch border-r hairline shrink-0 w-12 py-1 gap-1">
      {MODES.map((m) => (
        <button key={m.key} onClick={() => onMode(m.key)} title={`${m.label} (Shift ${m.hotkey})`}
          className={`mx-1 px-1 py-2 font-mono text-[10px] border text-center ${mode === m.key ? "border-accent text-accent" : "border-transparent text-ink-3 hover:text-ink-2 hover:border-line"}`}>
          {m.rail}
        </button>
      ))}
    </div>
  );
}
