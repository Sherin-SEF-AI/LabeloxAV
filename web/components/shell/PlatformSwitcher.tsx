"use client";

// The platform switcher: the primary navigation of the data engine. A two-pane menu, platforms on the left
// and the selected platform's destinations on the right (each with a one-line description, so the nav explains
// itself). Blender-style: raised header tone, rounded rows, the blue active state, a gate dot for planes that
// can block progression.

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
        data-tip="Switch platform: jump between the planes of the data engine"
        className={`btn gap-2 ${open ? "btn-on" : ""}`}
      >
        <span className="tracking-wider text-[10px] opacity-70">{current.glyph}</span>
        <span>{current.label}</span>
        <span className="opacity-60">&#9662;</span>
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          {/* Anchored to the button's right edge, not its left. This button lives near the right of the
              header, so a left-anchored panel 46rem wide started there and ran off the screen: at 1700px it
              ended at 2078, and everything past the fold was simply unreachable. Growing leftward from the
              button keeps it on screen at any width the header itself fits in.

              One column until there is room for two, for the same reason. The rail was a hard 200px, so on
              a narrow window it took most of the width and left the descriptions a strip too thin to read,
              and a nav that explains itself does not if the explanations are clipped. */}
          <div className="absolute right-0 mt-1 z-50 bg-panel border border-line rounded shadow-2xl
            grid grid-cols-1 sm:grid-cols-[180px_1fr] w-[46rem] max-w-[calc(100vw-1.5rem)]
            max-h-[80vh] overflow-y-auto overflow-x-hidden">
            {/* left: the platforms */}
            <div className="sm:border-r border-b sm:border-b-0 border-line bg-bg/40 py-1">
              <div className="px-3 py-1.5 text-[9px] uppercase tracking-wide text-ink-3">Platforms</div>
              {PLATFORMS_ORDERED.map((p) => {
                const active = p.id === current.id;
                const focused = p.id === preview.id;
                return (
                  <button
                    key={p.id}
                    onMouseEnter={() => setPreview(p)}
                    onClick={() => go(p.home)}
                    className={`w-full flex items-center gap-2 px-3 py-1.5 text-[12px] text-left rounded-sm mx-0
                      ${focused ? "bg-accent/20" : ""} ${active ? "text-accent-2 font-medium" : "text-ink-2 hover:text-ink"}`}
                  >
                    <span className="w-8 text-ink-3 tracking-wider text-[10px]">{p.glyph}</span>
                    <span className="flex-1 truncate">{p.label}</span>
                    {p.gate && <span className={`w-1.5 h-1.5 rounded-full ${GATE_DOT[p.gate] ?? "bg-ink-3"}`} data-tip={`gate: ${p.gate}`} />}
                  </button>
                );
              })}
            </div>

            {/* right: the previewed platform's destinations, each described */}
            <div className="p-3 min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="text-[13px] text-ink font-medium">{preview.label}</span>
                {preview.flywheelStage != null && <span className="text-[10px] text-ink-3">stage {preview.flywheelStage}</span>}
                {preview.gate && <span className="text-[9px] text-ink-3 border border-line rounded px-1 uppercase">gate: {preview.gate}</span>}
              </div>
              <div className="text-[11px] text-ink-3 leading-snug mt-1">{preview.role}</div>

              <div className="mt-3 grid grid-cols-1 lg:grid-cols-2 gap-1.5">
                {preview.nav.map((n) => (
                  <button
                    key={n.href}
                    onClick={() => go(n.href)}
                    className={`text-left px-2 py-1.5 rounded hover:bg-head border border-transparent hover:border-line
                      ${pathname === n.href ? "bg-accent/15 border-accent/40" : ""}`}
                  >
                    <div className={`text-[12px] ${pathname === n.href ? "text-accent-2" : "text-ink"}`}>{n.label}</div>
                    {/* Wraps rather than truncates. The hint is the whole reason this menu is two panes
                        instead of a list of names, and half a sentence explains nothing. */}
                    {n.hint && <div className="text-[10px] text-ink-3 leading-tight mt-0.5">{n.hint}</div>}
                  </button>
                ))}
              </div>

              <button onClick={() => go(preview.home)}
                className="mt-3 text-[11px] text-ink-3 hover:text-accent-2">open {preview.label} &rarr;</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
