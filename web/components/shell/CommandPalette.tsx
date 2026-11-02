"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
// Every sidebar destination (lib/sidebar.ts), then the deep links only the palette offers (each import and
// export format, saved views...) from lib/menus.ts. Sidebar labels come first so a page is found by the name
// the sidebar shows; a duplicate href+label pair from the older catalogue is dropped.
import { MENU_DESTINATIONS, type Destination } from "@/lib/menus";
import { NAV_DESTINATIONS, packOf } from "@/lib/sidebar";
import { packEnabled, useFeatures } from "@/lib/features";

const DESTINATIONS: Destination[] = (() => {
  const seen = new Set<string>();
  const out: Destination[] = [];
  for (const d of [...NAV_DESTINATIONS.map((n) => ({ key: n.key, label: n.label, href: n.href, hint: n.hint })),
                   ...MENU_DESTINATIONS]) {
    const id = `${d.href}::${d.label}`;
    if (!seen.has(id)) { seen.add(id); out.push(d); }
  }
  return out;
})();
import { rankDestinations } from "@/lib/paletteRank";

// Cmd+K fuzzy navigation to any destination. Opens on Cmd/Ctrl+K or a "lbx:palette" event (so a button
// can open it). Keyboard-first: type to filter, arrows to move, Enter to go. This is how nav scales: no
// flat link row, just search across everything the registry knows about.

export default function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [cur, setCur] = useState(0);
  const features = useFeatures();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
        setQ("");
        setCur(0);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    const onOpen = () => { setOpen(true); setQ(""); setCur(0); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("lbx:palette", onOpen);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("lbx:palette", onOpen); };
  }, []);

  if (!open) return null;
  // Ranked, not just filtered. Matching label and hint as one string put Projects above Jobs for the query
  // "jobs", because Projects is described as "assign jobs, stages, scorecards" and comes first in the menu.
  // An optional pack's pages are offered only where that pack is installed; elsewhere their routes do not exist.
  const offered = DESTINATIONS.filter((d) => { const p = packOf(d.href); return !p || packEnabled(features, p); });
  const results = rankDestinations(offered, q);
  const go = (href: string) => { setOpen(false); router.push(href); };

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center bg-black/60 pt-[12vh]" onClick={() => setOpen(false)}>
      <div className="panel w-full max-w-lg" onClick={(e) => e.stopPropagation()}>
        <input autoFocus value={q} placeholder="Go to a page..." spellCheck={false}
          onChange={(e) => { setQ(e.target.value); setCur(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setCur((c) => Math.min(c + 1, results.length - 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setCur((c) => Math.max(c - 1, 0)); }
            else if (e.key === "Enter" && results[cur]) go(results[cur].href);
          }}
          className="w-full bg-bg border-b hairline px-3 py-2 font-mono text-sm text-ink outline-none" />
        <div className="max-h-80 overflow-y-auto">
          {results.map((d, i) => (
            <button key={`${d.href}:${d.label}`} onClick={() => go(d.href)} onMouseEnter={() => setCur(i)}
              className={`w-full text-left px-3 py-1.5 font-mono text-xs flex items-center justify-between gap-3 ${i === cur ? "bg-bg-2 text-ink" : "text-ink-2"}`}>
              <span>{d.label}</span>
              <span className="text-ink-3 truncate">{d.hint}</span>
            </button>
          ))}
          {!results.length && <div className="px-3 py-3 font-mono text-xs text-ink-3">no match</div>}
        </div>
      </div>
    </div>
  );
}
