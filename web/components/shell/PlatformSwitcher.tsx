"use client";

// The platform switcher: the primary navigation of the data engine. A two-pane menu, platforms on the left
// and the selected platform's destinations on the right, so the whole app is organized by plane rather than
// by a flat list. The trigger shows the current platform (derived from the URL). Operational Materialism:
// matte, monospace, color earned only by state (a platform's gate dot, the active row).

import { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { PLATFORMS_ORDERED, platformForPath, type Platform } from "@/platforms/registry";

const GATE_DOT: Record<string, string> = {
  health: "bg-pass", calibration: "bg-info", eval: "bg-warn", benchmark: "bg-accent",
};

export default function PlatformSwitcher() {
  const router = useRouter();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const current = platformForPath(pathname) ?? PLATFORMS_ORDERED[0];
  const [preview, setPreview] = useState<Platform>(current);

  const go = (href: string) => { setOpen(false); router.push(href); };

  return (
    <div className="relative">
      <button
        onClick={() => { setPreview(current); setOpen((o) => !o); }}
        title="switch platform"
        className={`flex items-center gap-2 font-mono text-[11px] border px-2 py-1 hover:border-accent ${open ? "border-accent" : "border-line"}`}
      >
        <span className="text-ink-3 tracking-wider">{current.glyph}</span>
        <span className="text-ink-2">{current.label}</span>
        <span className="text-ink-3">&#9662;</span>
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute left-0 mt-1 z-50 panel grid grid-cols-[190px_1fr] w-[42rem] max-w-[92vw]">
            {/* left: the platforms */}
            <div className="border-r hairline py-1">
              <div className="px-3 py-1 font-mono text-[9px] uppercase tracking-wide text-ink-3">platforms</div>
              {PLATFORMS_ORDERED.map((p) => {
                const active = p.id === current.id;
                const focused = p.id === preview.id;
                return (
                  <button
                    key={p.id}
                    onMouseEnter={() => setPreview(p)}
                    onClick={() => go(p.home)}
                    className={`w-full flex items-center gap-2 px-3 py-1.5 font-mono text-[11px] text-left
                      ${focused ? "bg-line/50" : ""} ${active ? "text-accent" : "text-ink-2 hover:text-ink"}`}
                  >
                    <span className="w-8 text-ink-3 tracking-wider">{p.glyph}</span>
                    <span className="flex-1 truncate">{p.label}</span>
                    {p.gate && <span className={`w-1.5 h-1.5 rounded-full ${GATE_DOT[p.gate] ?? "bg-ink-3"}`} title={`gate: ${p.gate}`} />}
                  </button>
                );
              })}
            </div>

            {/* right: the previewed platform's destinations */}
            <div className="p-3 min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-[11px] text-ink">{preview.label}</span>
                {preview.flywheelStage != null && <span className="font-mono text-[9px] text-ink-3">stage {preview.flywheelStage}</span>}
                {preview.gate && <span className="font-mono text-[9px] text-ink-3 border border-line rounded px-1 uppercase">gate: {preview.gate}</span>}
              </div>
              <div className="font-mono text-[10px] text-ink-3 leading-snug mt-0.5">{preview.role}</div>

              <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-0.5">
                {preview.nav.map((n) => (
                  <button
                    key={n.href}
                    onClick={() => go(n.href)}
                    title={n.hint}
                    className={`text-left font-mono text-[11px] px-1.5 py-1 hover:bg-bg-2 ${pathname === n.href ? "text-accent" : "text-ink-2"}`}
                  >
                    {n.label}
                  </button>
                ))}
              </div>

              <button onClick={() => go(preview.home)}
                className="mt-3 font-mono text-[10px] text-ink-3 hover:text-accent">open {preview.label} &rarr;</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
