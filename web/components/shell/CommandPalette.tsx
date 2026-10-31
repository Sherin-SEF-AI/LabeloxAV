"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
// Every sidebar destination (lib/sidebar.ts), then the deep links only the palette offers (each import and
// export format, saved views...) from lib/menus.ts. Sidebar labels come first so a page is found by the name
// the sidebar shows; a duplicate href+label pair from the older catalogue is dropped.
import { MENU_DESTINATIONS, type Destination } from "@/lib/menus";
import { NAV_DESTINATIONS, packOf } from "@/lib/sidebar";
import { packEnabled, useFeatures } from "@/lib/features";
import { apiGet } from "@/lib/api";

// Sessions, frames, objects and tracks, found by name or by the first characters of an id (GET /api/find).
type Found = { kind: string; id: string; label: string; detail: string; href: string };
type Row = { key: string; label: string; hint?: string; href: string; group: "Pages" | "Data"; kind?: string };
const KIND_LABEL: Record<string, string> = { session: "Session", frame: "Frame", object: "Object", track: "Track" };
// Enough pages to find one by name without pushing the data matches out of view.
const MAX_PAGES = 8;

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
  const [found, setFound] = useState<Found[]>([]);
  const [searching, setSearching] = useState(false);
  const features = useFeatures();

  // Data matches arrive a beat after the page matches: debounced so typing a word costs one request, and
  // discarded if the query has moved on by the time they return.
  useEffect(() => {
    const term = q.trim();
    if (!open || term.length < 2) { setFound([]); setSearching(false); return; }
    let live = true;
    setSearching(true);
    const t = setTimeout(() => {
      apiGet<{ results: Found[] }>(`/api/find?q=${encodeURIComponent(term)}`)
        .then((r) => { if (live) setFound(r.results); })
        .catch(() => { if (live) setFound([]); })
        .finally(() => { if (live) setSearching(false); });
    }, 180);
    return () => { live = false; clearTimeout(t); };
  }, [q, open]);

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
  const pages = rankDestinations(offered, q).slice(0, q.trim() ? MAX_PAGES : undefined);
  const results: Row[] = [
    ...pages.map((d) => ({ key: `p:${d.href}:${d.label}`, label: d.label, hint: d.hint, href: d.href, group: "Pages" as const })),
    ...found.map((f) => ({ key: `d:${f.kind}:${f.id}`, label: f.label, hint: f.detail, href: f.href, group: "Data" as const, kind: f.kind })),
  ];
  const go = (href: string) => { setOpen(false); router.push(href); };

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center bg-black/60 pt-[12vh]" onClick={() => setOpen(false)}>
      <div className="panel w-full max-w-lg" onClick={(e) => e.stopPropagation()}>
        <input autoFocus value={q} placeholder="Go to a page, or find a session, frame, object or track by name or id" spellCheck={false}
          onChange={(e) => { setQ(e.target.value); setCur(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setCur((c) => Math.min(c + 1, results.length - 1)); }
            else if (e.key === "ArrowUp") { e.preventDefault(); setCur((c) => Math.max(c - 1, 0)); }
            else if (e.key === "Enter" && results[cur]) go(results[cur].href);
          }}
          className="w-full bg-bg border-b hairline px-3 py-2.5 text-[13.5px] text-ink outline-none placeholder:text-ink-3" />
        <div className="max-h-96 overflow-y-auto py-1">
          {results.map((d, i) => (
            <div key={d.key}>
              {(i === 0 || results[i - 1].group !== d.group) && (
                <div className="px-3 pt-2 pb-1 text-[10.5px] font-medium uppercase tracking-[0.06em] text-ink-3">{d.group}</div>
              )}
              <button onClick={() => go(d.href)} onMouseEnter={() => setCur(i)}
                className={`w-full text-left px-3 py-1.5 text-[12.5px] flex items-center justify-between gap-3 ${i === cur ? "bg-head text-ink" : "text-ink-2"}`}>
                <span className="flex items-center gap-2 min-w-0">
                  {d.kind && <span className="shrink-0 text-[10.5px] text-ink-3 w-12">{KIND_LABEL[d.kind] ?? d.kind}</span>}
                  <span className="truncate">{d.label}</span>
                </span>
                <span className="text-ink-3 truncate">{d.hint}</span>
              </button>
            </div>
          ))}
          {searching && <div className="px-3 py-2 text-[12px] text-ink-3">Searching sessions and ids...</div>}
          {!results.length && !searching && <div className="px-3 py-3 text-[12.5px] text-ink-3">Nothing matches &ldquo;{q}&rdquo;</div>}
        </div>
      </div>
    </div>
  );
}
