"use client";

import { useState } from "react";

// Layer visibility is view state, not a tool, so it lives as a small collapsible cluster floating on the
// canvas instead of competing for a toolbar row. Reads like a layers panel (toggle on/off), keyed off the
// layers object, so a new layer is one more key with zero toolbar impact.

export default function FloatingLayers({ layers, onToggle, extra }: {
  layers: Record<string, boolean>;
  onToggle: (key: string) => void;
  extra?: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="absolute top-2 right-2 z-20 panel font-mono text-[10px] min-w-[7rem]">
      <button onClick={() => setOpen((o) => !o)} className="flex items-center justify-between w-full px-2 py-1 text-ink-3 hover:text-ink-2">
        <span className="uppercase tracking-wide">layers</span><span>{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="px-2 pb-2 space-y-0.5">
          {Object.keys(layers).map((k) => (
            <button key={k} onClick={() => onToggle(k)}
              className={`flex items-center gap-1.5 w-full ${layers[k] ? "text-ink" : "text-ink-3"}`}>
              <span className={`w-2.5 h-2.5 inline-block border ${layers[k] ? "bg-accent border-accent" : "border-line"}`} />
              {k}
            </button>
          ))}
          {extra && <div className="pt-1.5 mt-1 border-t hairline space-y-1">{extra}</div>}
        </div>
      )}
    </div>
  );
}
