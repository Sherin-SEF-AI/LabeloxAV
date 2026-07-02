"use client";

import { useState } from "react";
import Icon from "@/components/shell/Icon";

// Layer visibility is view state, not a tool, so it lives as a small collapsible cluster floating on the
// canvas (the design's top-right Layers control) instead of competing for a toolbar row. Reads like a
// layers panel: an eye toggle per layer keyed off the layers object, so a new layer is one more key with
// zero toolbar impact.

export default function FloatingLayers({ layers, onToggle, extra, meta }: {
  layers: Record<string, boolean>;
  onToggle: (key: string) => void;
  extra?: React.ReactNode;
  // per-layer provenance line ("proposed - mask2former"), so it is obvious who produced each overlay
  // (an AI model on the pod vs a human vs an import) without opening the object list or the database.
  meta?: Record<string, string>;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="absolute top-3 right-3 z-20 w-[190px] panel overflow-hidden">
      <button onClick={() => setOpen((o) => !o)} className="flex items-center gap-1.5 w-full px-2.5 py-2 border-b hairline">
        <span className="flex text-ink-3"><Icon name="layers" size={14} /></span>
        <span className="font-display font-semibold text-[10px] uppercase tracking-wider text-ink-2">Layers</span>
        <span className="ml-auto font-mono text-[9px] text-ink-3/70">{open ? "view state" : "show"}</span>
      </button>
      {open && (
        <div className="p-1">
          {Object.keys(layers).map((k) => {
            const on = layers[k];
            const prov = meta?.[k];
            return (
              <button key={k} onClick={() => onToggle(k)} title={prov ? `${k}: ${prov}` : k}
                className={`flex items-center gap-2 w-full px-2 py-1 rounded hover:bg-line/40 ${on ? "text-ink-2" : "text-ink-3"}`}>
                <span className={`flex ${on ? "text-ink-2" : "text-ink-3/60"}`}><Icon name={on ? "eye" : "eyeOff"} size={14} /></span>
                <span className={`w-2 h-2 rounded-sm shrink-0 ${on ? "bg-accent" : "bg-line"}`} />
                <span className="flex-1 min-w-0 text-left">
                  <span className="block font-body text-[11.5px] leading-tight">{k}</span>
                  {prov && <span className="block font-mono text-[9px] leading-tight text-ink-3/80 truncate">{prov}</span>}
                </span>
              </button>
            );
          })}
          {extra && <div className="pt-1.5 mt-1 border-t hairline px-1 space-y-1">{extra}</div>}
        </div>
      )}
    </div>
  );
}
