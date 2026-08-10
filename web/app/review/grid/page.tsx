"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useRouter } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import { api, humanizeError } from "@/lib/api";
import {
  cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle,
} from "@/lib/gridSelection";
import { toast } from "@/lib/toast";
import { describeScope, triageQuery } from "@/lib/triageScope";
import { acceptState, useCurrentUser } from "@/lib/user";
import type { TriageRow } from "@/lib/types";

// Crop-grid review: many crops on screen, one keystroke each.
//
// Rapid review already removed the navigation from a verdict, and what is left is the screen itself. One
// crop at a time means one judgement per screenful, and the reviewer spends most of the interval waiting
// for the next image rather than deciding anything. A grid puts the whole batch in front of them, so the
// eye moves instead of the page, and a run of identical mistakes is one selection rather than twenty
// keystrokes.
//
// This matters more than it sounds: 252 of 570,379 objects in this corpus have ever been reviewed, and
// label supply is what every model number downstream is starved of.
//
// Deliberately one class at a time by default. A grid of mixed classes forces a context switch per cell and
// throws away the reason a grid is fast, which is that the eye can compare neighbours.

const CELL = 128;                 // px per tile in the sheet the server builds
const PAGE = 120;                 // tiles fetched per sheet; a screenful and change

type Verdict = "accept" | "reject";

export default function ReviewGridPage() {
  return (
    <Suspense fallback={
      <PageShell active="REVIEW GRID" title="Crop grid">
        <div className="p-4 font-mono text-[11px] text-ink-3">loading the grid...</div>
      </PageShell>
    }>
      <Grid />
    </Suspense>
  );
}

function Grid() {
  const params = useSearchParams();
  const router = useRouter();
  const me = useCurrentUser();

  const [queue, setQueue] = useState<TriageRow[]>([]);
  const [sheet, setSheet] = useState<string | null>(null);
  const [cols, setCols] = useState(1);
  const [placements, setPlacements] = useState<Record<string, { row: number; col: number; ok: boolean }>>({});
  const [cursor, setCursor] = useState(0);
  const [selection, setSelection] = useState<Set<number>>(new Set());
  const [anchor, setAnchor] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [decided, setDecided] = useState(0);
  const startedAt = useRef<number>(Date.now());
  const batchStart = useRef<number>(Date.now());

  const scope = params.toString();
  const scopeLabel = describeScope(params);

  // Load the queue, then one sheet for the page of tiles on screen.
  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const rows = await api.triage(triageQuery(new URLSearchParams(scope)));
        setQueue(rows);
        setCursor(0);
        setSelection(new Set());
        startedAt.current = Date.now();
        batchStart.current = Date.now();
      } catch (e) {
        toast(humanizeError(e), "error");
      } finally {
        setLoading(false);
      }
    })();
  }, [scope]);

  const page = useMemo(() => queue.slice(0, PAGE), [queue]);

  useEffect(() => {
    if (!page.length) { setSheet(null); setPlacements({}); return; }
    let cancelled = false;
    (async () => {
      try {
        const s = await api.cropSheet(page.map((r) => r.object_id), CELL);
        if (cancelled) return;
        setSheet(s.sheet);
        setCols(s.cols || 1);
        const m: Record<string, { row: number; col: number; ok: boolean }> = {};
        for (const p of s.placements) m[p.object_id] = { row: p.row, col: p.col, ok: p.ok };
        setPlacements(m);
      } catch (e) {
        if (!cancelled) toast(humanizeError(e), "error");
      }
    })();
    return () => { cancelled = true; };
    // Keyed on the ids so a verdict that removes tiles refetches exactly one sheet, not one per keystroke.
  }, [page.map((r) => r.object_id).join(",")]);

  const decide = useCallback(async (verdict: Verdict, className?: string) => {
    const idx = targetIndices(selection, cursor);
    const rows = idx.map((i) => page[i]).filter(Boolean);
    if (!rows.length) return;

    const ids = rows.map((r) => r.object_id);
    const elapsed = Math.max(0, Date.now() - batchStart.current);
    batchStart.current = Date.now();

    // Optimistic: the tiles go now and the request settles behind them, because waiting for a round trip
    // per verdict is the cost this whole mode exists to remove.
    const removing = new Set(ids);
    setQueue((q) => q.filter((r) => !removing.has(r.object_id)));
    setSelection(new Set());
    setAnchor(null);
    setCursor((c) => cursorAfterRemoval(c, idx, Math.max(0, page.length - idx.length)));
    setDecided((d) => d + ids.length);

    try {
      const res = await api.bulkReview(
        ids,
        verdict === "accept" ? "confirm" : className ? "reclassify" : "reject",
        className,
        verdict === "reject" ? "rejected" : acceptState(me?.role),
        undefined,
        // The batch's real elapsed time, divided across its members by the server. A grid that reported
        // zero would quietly corrupt every throughput and cost-per-label number built on it.
        { time_spent_ms: elapsed },
      );
      const lost = (res.skipped_stale?.length ?? 0) + (res.skipped_missing?.length ?? 0);
      if (lost) {
        toast(`${res.updated} applied, ${lost} skipped (changed or gone since this grid loaded)`, "error");
        setDecided((d) => d - lost);
      }
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, [selection, cursor, page, me]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return;
      const k = e.key.toLowerCase();
      const n = page.length;
      if (!n) return;

      if (k === "arrowleft" || k === "h") { e.preventDefault(); step("left", e.shiftKey); }
      else if (k === "arrowright" || k === "l") { e.preventDefault(); step("right", e.shiftKey); }
      else if (k === "arrowup" || k === "k") { e.preventDefault(); step("up", e.shiftKey); }
      else if (k === "arrowdown" || k === "j") { e.preventDefault(); step("down", e.shiftKey); }
      else if (k === "a") { e.preventDefault(); void decide("accept"); }
      else if (k === "r") { e.preventDefault(); void decide("reject"); }
      else if (k === " ") { e.preventDefault(); setSelection((s) => toggle(s, cursor)); setAnchor(cursor); }
      else if (k === "escape") { setSelection(new Set()); setAnchor(null); }
      else if (k === "o") { const row = page[cursor]; if (row) router.push(`/frame/${row.frame_id}`); }

      function step(dir: "left" | "right" | "up" | "down", extend: boolean) {
        const next = moveCursor(cursor, dir, n, cols);
        setCursor(next);
        if (!extend) return;
        // Shift extends from wherever the range started, so reversing direction shrinks it rather than
        // leaving a stale tail selected.
        const a = anchor ?? cursor;
        setAnchor(a);
        setSelection(new Set(rangeBetween(a, next)));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, cols, page, anchor, decide, router]);

  const perMin = useMemo(() => {
    const mins = (Date.now() - startedAt.current) / 60000;
    return mins > 0.05 ? Math.round(decided / mins) : null;
  }, [decided]);

  const current = page[cursor];

  return (
    <PageShell
      active="REVIEW GRID"
      title="Crop grid"
      subtitle={scopeLabel ? `${scopeLabel} · many crops, one keystroke each` : "many crops, one keystroke each"}
    >
      <div className="p-3 space-y-2">
        <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] text-ink-3">
          <span className="text-ink">{queue.length} left</span>
          <span>{decided} decided</span>
          {/* The claim this mode makes is a throughput claim, so it is measured on screen rather than
              asserted in a commit message. */}
          {perMin != null && <span className="text-accent">{perMin}/min</span>}
          {selection.size > 0 && <span className="text-warn">{selection.size} selected</span>}
          {current && <span className="text-ink-2">{current.class_name} · conf {current.conf?.toFixed(2)}</span>}
        </div>

        {loading && <div className="font-mono text-[11px] text-ink-3">loading the grid...</div>}

        {!loading && !page.length && (
          <div className="panel px-3 py-6 font-mono text-[11px] text-ink-3 text-center">
            Nothing left in this queue. A scoped grid ending is the batch finished, not an empty corpus.
          </div>
        )}

        {sheet && page.length > 0 && (
          <div className="panel p-2 overflow-x-auto">
            <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, ${CELL}px)` }}>
              {page.map((row, i) => {
                const p = placements[row.object_id];
                const isCursor = i === cursor;
                const isSel = selection.has(i);
                return (
                  <button
                    key={row.object_id}
                    onClick={(e) => {
                      if (e.shiftKey) {
                        const a = anchor ?? cursor;
                        setAnchor(a);
                        setSelection(new Set(rangeBetween(a, i)));
                      } else {
                        setSelection((s) => toggle(s, i));
                        setAnchor(i);
                      }
                      setCursor(i);
                    }}
                    title={`${row.class_name} · ${row.why}`}
                    className={`relative block ${isCursor ? "ring-2 ring-accent" : isSel ? "ring-2 ring-warn" : "ring-1 ring-line"}`}
                    style={{ width: CELL, height: CELL }}
                  >
                    {p?.ok ? (
                      // One image, positioned. Each tile is a window onto the same sprite sheet, so the
                      // grid is a single decode on the client as well as on the server.
                      <span
                        className="block w-full h-full"
                        style={{
                          backgroundImage: `url(${sheet})`,
                          backgroundPosition: `-${p.col * CELL}px -${p.row * CELL}px`,
                        }}
                      />
                    ) : (
                      <span className="flex w-full h-full items-center justify-center text-[9px] text-ink-3">
                        no crop
                      </span>
                    )}
                    {isSel && <span className="absolute top-0 right-0 bg-warn text-bg text-[9px] px-1">+</span>}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div className="font-mono text-[10px] text-ink-3 flex flex-wrap gap-3">
          <span><b className="text-ink">arrows/hjkl</b> move</span>
          <span><b className="text-ink">shift+move</b> range</span>
          <span><b className="text-ink">space</b> select</span>
          <span><b className="text-accent">A</b> accept</span>
          <span><b className="text-block">R</b> reject</span>
          <span><b className="text-ink">O</b> open the frame</span>
          <span><b className="text-ink">esc</b> clear</span>
          <span className="text-ink-2">a verdict with nothing selected applies to the tile under the cursor</span>
        </div>
      </div>
    </PageShell>
  );
}
