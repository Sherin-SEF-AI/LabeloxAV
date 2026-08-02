"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import { api, humanizeError } from "@/lib/api";
import {
  cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle,
} from "@/lib/gridSelection";
import { toast } from "@/lib/toast";
import type { ErrorCandidateRow } from "@/lib/types";

// Ruling on error candidates the way review rules on labels: many crops on screen, one keystroke each.
//
// The detectors have produced 298,529 candidates and hold one verdict between them. Confirm and dismiss
// both took a single candidate id, so the queue was not reviewable in principle: at one request per
// candidate nobody was ever going to get through it, and until verdicts exist in volume no detector has a
// measurable precision. The scores rank, they do not predict, and there was no way to find out which.
//
// One detector at a time by default, which matters more here than in label review. The queue ranks across
// detectors by score and the detectors do not emit commensurable scores, so a mixed page is largely
// whichever detector produces the biggest numbers. Judging a detector also requires seeing only that
// detector, and judging them is the entire point of the verdicts.
//
// Dismissal is the common action and is deliberately cheap. Dismissing forty near-duplicate candidates is
// one judgement about a detector, not forty judgements about objects, and the note carries that judgement.

const CELL = 128;
const PAGE = 120;

export default function ErrorGridPage() {
  return (
    <Suspense fallback={
      <PageShell active="ERROR QUEUE" title="Error candidates">
        <div className="p-4 font-mono text-[11px] text-ink-3">loading the queue...</div>
      </PageShell>
    }>
      <ErrorGrid />
    </Suspense>
  );
}

type Precision = Awaited<ReturnType<typeof api.errorPrecision>>;

function ErrorGrid() {
  const params = useSearchParams();
  const router = useRouter();
  const kind = params.get("kind") || "";

  const [queue, setQueue] = useState<ErrorCandidateRow[]>([]);
  const [sheet, setSheet] = useState<string | null>(null);
  const [cols, setCols] = useState(1);
  const [placements, setPlacements] = useState<Record<string, { row: number; col: number; ok: boolean }>>({});
  const [cursor, setCursor] = useState(0);
  const [selection, setSelection] = useState<Set<number>>(new Set());
  const [anchor, setAnchor] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [decided, setDecided] = useState(0);
  const [precision, setPrecision] = useState<Precision | null>(null);
  const [note, setNote] = useState("");
  const startedAt = useRef<number>(Date.now());

  const loadPrecision = useCallback(async () => {
    try {
      setPrecision(await api.errorPrecision());
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const rows = await api.errorCandidates("pending", PAGE, kind || undefined);
        setQueue(rows);
        setCursor(0);
        setSelection(new Set());
        startedAt.current = Date.now();
      } catch (e) {
        toast(humanizeError(e), "error");
      } finally {
        setLoading(false);
      }
    })();
    void loadPrecision();
  }, [kind, loadPrecision]);

  const page = useMemo(() => queue.slice(0, PAGE), [queue]);
  const pageKey = page.map((r) => r.object_id).join(",");

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageKey]);

  const decide = useCallback(async (verdict: "confirmed_error" | "dismissed") => {
    const idx = targetIndices(selection, cursor);
    const rows = idx.map((i) => page[i]).filter(Boolean);
    if (!rows.length) return;
    const ids = rows.map((r) => r.candidate_id);

    // Optimistic, like the review grid: waiting for a round trip per verdict is the cost this mode exists
    // to remove. The reconciliation below puts back anything the server did not actually apply.
    const removing = new Set(ids);
    setQueue((q) => q.filter((r) => !removing.has(r.candidate_id)));
    setSelection(new Set());
    setAnchor(null);
    setCursor((c) => cursorAfterRemoval(c, idx, Math.max(0, page.length - idx.length)));
    setDecided((d) => d + ids.length);

    try {
      const res = await api.errorBulk(ids, verdict, note.trim() || undefined);
      const lost = res.missing + res.already_decided;
      if (lost) {
        // already_decided is the interesting one: somebody else ruled on these while this grid was open,
        // and their verdict stands rather than being overwritten.
        toast(`${res.applied} applied, ${lost} skipped (already ruled on, or gone)`, "error");
        setDecided((d) => d - lost);
      }
      void loadPrecision();
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, [selection, cursor, page, note, loadPrecision]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return;
      // A note field is on this page, so typing "d" into it must not dismiss forty candidates.
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) return;

      const k = e.key.toLowerCase();
      const n = page.length;
      if (!n) return;

      if (k === "arrowleft" || k === "h") { e.preventDefault(); step("left", e.shiftKey); }
      else if (k === "arrowright" || k === "l") { e.preventDefault(); step("right", e.shiftKey); }
      else if (k === "arrowup" || k === "k") { e.preventDefault(); step("up", e.shiftKey); }
      else if (k === "arrowdown" || k === "j") { e.preventDefault(); step("down", e.shiftKey); }
      else if (k === "c") { e.preventDefault(); void decide("confirmed_error"); }
      else if (k === "d") { e.preventDefault(); void decide("dismissed"); }
      else if (k === " ") { e.preventDefault(); setSelection((s) => toggle(s, cursor)); setAnchor(cursor); }
      else if (k === "a") {
        e.preventDefault();
        setSelection(new Set(page.map((_, i) => i)));   // the whole page, for a detector that is all noise
        setAnchor(0);
      }
      else if (k === "escape") { setSelection(new Set()); setAnchor(null); }
      else if (k === "o") { const row = page[cursor]; if (row) router.push(`/object/${row.object_id}`); }

      function step(dir: "left" | "right" | "up" | "down", extend: boolean) {
        const next = moveCursor(cursor, dir, n, cols);
        setCursor(next);
        if (!extend) return;
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
  const kinds = precision ? Object.keys(precision.per_kind).sort() : [];

  return (
    <PageShell
      active="ERROR QUEUE"
      title="Error candidates"
      subtitle={kind ? `${kind} · confirm or dismiss, one keystroke each` : "confirm or dismiss, one keystroke each"}
    >
      <div className="p-3 space-y-2">
        {/* The detector scoreboard. Shown above the grid because which detector you are judging is the
            decision that matters most, and because a detector nobody has ruled on should say so rather
            than showing an interval that looks like a bad score. */}
        {precision && (
          <div className="panel p-2 overflow-x-auto">
            <div className="font-mono text-[10px] text-ink-3 mb-1">
              detector precision, from human verdicts
              {precision.caveat && <span className="text-warn"> · {precision.caveat}</span>}
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => router.push("/quality/errors")}
                className={`px-2 py-1 font-mono text-[10px] ring-1 ${kind ? "ring-line text-ink-3" : "ring-accent text-accent"}`}
              >
                all detectors
              </button>
              {kinds.map((k) => {
                const d = precision.per_kind[k];
                return (
                  <button
                    key={k}
                    onClick={() => router.push(`/quality/errors?kind=${encodeURIComponent(k)}`)}
                    title={d.note || `${d.decided} verdicts`}
                    className={`px-2 py-1 font-mono text-[10px] ring-1 ${k === kind ? "ring-accent text-accent" : "ring-line text-ink-2"}`}
                  >
                    {k} <span className="text-ink-3">{d.pending}</span>{" "}
                    {d.usable
                      ? <span className="text-ink">P {d.precision.p?.toFixed(2)} ({d.precision.lo.toFixed(2)}-{d.precision.hi.toFixed(2)})</span>
                      : <span className="text-ink-3">unmeasured</span>}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] text-ink-3">
          <span className="text-ink">{queue.length} left</span>
          <span>{decided} decided</span>
          {perMin != null && <span className="text-accent">{perMin}/min</span>}
          {selection.size > 0 && <span className="text-warn">{selection.size} selected</span>}
          {current && <span className="text-ink-2">{current.kind} · score {current.score?.toFixed(3)}</span>}
        </div>

        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="why (recorded on every candidate in the batch, e.g. 'scores frame similarity, not the object')"
          className="w-full bg-bg-2 px-2 py-1 font-mono text-[11px] text-ink ring-1 ring-line"
        />

        {loading && <div className="font-mono text-[11px] text-ink-3">loading the queue...</div>}

        {!loading && !page.length && (
          <div className="panel px-3 py-6 font-mono text-[11px] text-ink-3 text-center">
            Nothing pending{kind ? ` for ${kind}` : ""}. Every candidate here has been ruled on.
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
                    key={row.candidate_id}
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
                    title={`${row.kind} · score ${row.score?.toFixed(3)}${row.proposed_label?.class_name ? ` · proposes ${row.proposed_label.class_name}` : ""}`}
                    className={`relative block ${isCursor ? "ring-2 ring-accent" : isSel ? "ring-2 ring-warn" : "ring-1 ring-line"}`}
                    style={{ width: CELL, height: CELL }}
                  >
                    {p?.ok ? (
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
                    {row.proposed_label?.class_name && (
                      <span className="absolute bottom-0 left-0 right-0 bg-bg/80 text-[9px] text-accent truncate px-1">
                        {row.proposed_label.class_name}
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
          <span><b className="text-ink">A</b> select the page</span>
          <span><b className="text-accent">C</b> confirm the error</span>
          <span><b className="text-block">D</b> dismiss</span>
          <span><b className="text-ink">O</b> open the object</span>
          <span><b className="text-ink">esc</b> clear</span>
          <span className="text-ink-2">confirming applies the proposed class where there is one; dismissing says the detector was wrong</span>
        </div>
      </div>
    </PageShell>
  );
}
