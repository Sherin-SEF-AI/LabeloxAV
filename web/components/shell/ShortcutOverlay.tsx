"use client";

import { useEffect, useState } from "react";

// A searchable keyboard-shortcut reference. Opens on "?" (or a window "lbx:shortcuts" event from a button),
// closes on Escape. Lists the global keys and the editor tool keys exactly as the frame editor's keyboard
// handler binds them (the tool keys are global: the same letter selects the same tool in every mode, though
// a tool may be inert on a swapped canvas). Discoverability without cluttering the canvas.

const GLOBAL: { keys: string; label: string }[] = [
  { keys: "?", label: "this shortcut help" },
  { keys: "Cmd K", label: "command palette (jump to any screen)" },
  { keys: "Cmd S", label: "save" },
  { keys: "Cmd Z", label: "undo" },
  { keys: "Cmd Shift Z", label: "redo" },
  { keys: "Cmd C", label: "copy selected object" },
  { keys: "Cmd V", label: "paste object" },
  { keys: "Space", label: "pan the canvas (hold)" },
  { keys: "[", label: "previous frame" },
  { keys: "]", label: "next frame" },
  { keys: "1 to 9", label: "relabel selected to class N" },
  { keys: "Shift 1 to 5", label: "switch mode" },
  { keys: "A", label: "accept all (Review mode: accept selected)" },
  { keys: "X", label: "reject selected (Review mode)" },
  { keys: "Enter", label: "finish the AI mask in progress" },
  { keys: "Esc", label: "discard the AI mask / close overlays" },
];

// The editor tool keys, exactly as the frame editor's keyboard handler binds them (global across modes).
const TOOLS: { keys: string; label: string }[] = [
  { keys: "V", label: "select" },
  { keys: "B", label: "box" },
  { keys: "G", label: "polygon" },
  { keys: "L", label: "polyline" },
  { keys: "S", label: "SAM point" },
  { keys: "M", label: "SAM box" },
  { keys: "W", label: "magic wand" },
  { keys: "P", label: "brush" },
  { keys: "E", label: "eraser" },
  { keys: "U", label: "superpixels (cells)" },
  { keys: "K", label: "keypoint (pose)" },
  { keys: "D", label: "adverse region" },
  { keys: "C", label: "cuboid (3D box on image)" },
  { keys: "R", label: "measure" },
];

export default function ShortcutOverlay() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");

  useEffect(() => {
    const typing = (el: EventTarget | null) => {
      const t = el as HTMLElement | null;
      return !!t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA");
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "?" && !typing(e.target)) { e.preventDefault(); setOpen((o) => !o); }
      else if (e.key === "Escape") setOpen(false);
    };
    const onEvt = () => setOpen(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("lbx:shortcuts", onEvt);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("lbx:shortcuts", onEvt); };
  }, []);

  if (!open) return null;
  const needle = q.toLowerCase();
  const match = (label: string, keys: string) => !needle || label.toLowerCase().includes(needle) || keys.toLowerCase().includes(needle);

  const sections = [
    { title: "global", rows: GLOBAL.filter((r) => match(r.label, r.keys)) },
    { title: "tools", rows: TOOLS.filter((r) => match(r.label, r.keys)) },
  ].filter((s) => s.rows.length);

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[12vh] bg-bg/70" onClick={() => setOpen(false)}>
      <div className="panel w-[34rem] max-h-[70vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 px-3 h-10 border-b hairline shrink-0">
          <span className="font-mono text-[11px] uppercase tracking-wide text-ink-3">keyboard shortcuts</span>
          <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="search shortcuts..."
            className="ml-auto w-48 bg-bg border border-line px-2 py-1 font-mono text-[11px] text-ink" />
          <button onClick={() => setOpen(false)} className="font-mono text-[11px] text-ink-3 hover:text-accent">esc</button>
        </div>
        <div className="overflow-y-auto p-3 space-y-3">
          {sections.map((s) => (
            <div key={s.title}>
              <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3 mb-1">{s.title}</div>
              <div className="space-y-0.5">
                {s.rows.map((r, i) => (
                  <div key={`${r.keys}-${i}`} className="flex items-center justify-between gap-4 font-mono text-[11px]">
                    <span className="text-ink-2 truncate">{r.label}</span>
                    <span className="text-accent shrink-0">{r.keys}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {!sections.length && <div className="font-mono text-[11px] text-ink-3 text-center py-4">no shortcuts match</div>}
        </div>
      </div>
    </div>
  );
}
