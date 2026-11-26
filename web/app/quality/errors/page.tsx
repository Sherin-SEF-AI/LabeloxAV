"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import DetectorBenchmark from "@/components/quality/DetectorBenchmark";
import ErrorWhyPanel from "@/components/quality/ErrorWhyPanel";
import { api, humanizeError } from "@/lib/api";
import {
  cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle,
} from "@/lib/gridSelection";
import { toast } from "@/lib/toast";
import type { ErrorCandidateRow } from "@/lib/types";
import { useCurrentUser } from "@/lib/user";
import { invalidate, useQuery } from "@/lib/useQuery";
import { REVIEW_QUEUE_KEY } from "@/lib/queryKeys";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, pct } from "@/lib/format";
import { verdictTone } from "@/lib/tools/quality";
import { getErrorWhy, getVlmChecks, type ErrorWhy, type VlmCheck } from "@/lib/tools/qualityApi";

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

// What each detector checks, in words a labeller uses. The stored kind (`confident_learning`) stays the
// value in the URL and the API; this is display only, with the code kept in the tooltip.
const DETECTOR: Record<string, string> = {
  confident_learning: "Disagrees with the model",
  embedding_outlier: "Looks unlike its class",
  track_inconsistent: "Class changes along its track",
  track_consistency: "Class changes along its track",
  cross_cam_inconsistent: "Disagrees across cameras",
  critic_flag: "Flagged by the AI critic",
  near_dup_inconsistent: "Differs from near-duplicates",
  policy_violation: "Breaks a labelling rule",
  reanalyze: "Flagged on re-check",
};
const detectorName = (k: string) => DETECTOR[k] ?? humanize(k);

// A key cap for the shortcut hints.
const KBD = "inline-block min-w-[18px] text-center px-1 py-px rounded border hairline bg-head text-[11px] text-ink-2 font-mono";

export default function ErrorGridPage() {
  return (
    <Suspense fallback={
      <PageShell active="ERROR QUEUE" title="Error candidates">
        <div className="p-6 text-[13px] text-ink-3" role="status">Loading the queue...</div>
      </PageShell>
    }>
      <ErrorGrid />
    </Suspense>
  );
}

type Precision = Awaited<ReturnType<typeof api.errorPrecision>>;

function ErrorGrid() {
  const me = useCurrentUser();
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
  // Per-detector precision is the same whichever kind is being ruled on, so it is one cached read,
  // refreshed after each batch of verdicts.
  const precisionQ = useQuery(["errorPrecision"], () => api.errorPrecision());
  const precision: Precision | null = precisionQ.data ?? null;
  const loadPrecision = precisionQ.reload;
  useEffect(() => { if (precisionQ.error) toast(humanizeError(precisionQ.error), "error"); }, [precisionQ.error]);
  const [note, setNote] = useState("");
  const startedAt = useRef<number>(Date.now());

  // The candidate queue and its crop sheet stay outside the query cache, on purpose. The queue is
  // consumed by position with optimistic removal, so cached rows swapped for fresh ones would move tiles
  // under a verdict; and the sheet is a new image per set of ids, which a cache with no eviction would
  // keep for the whole shift.
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
  }, [kind]);

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

  // Why the tile under the cursor was flagged (tool #12), fetched as the cursor settles so arrowing through
  // the grid does not fire a request per keystroke.
  const [why, setWhy] = useState<ErrorWhy | null>(null);
  const [whyErr, setWhyErr] = useState<string | null>(null);
  const cursorCid = page[cursor]?.candidate_id ?? null;
  useEffect(() => {
    setWhyErr(null);
    if (!cursorCid) { setWhy(null); return; }
    let stop = false;
    const t = setTimeout(() => {
      getErrorWhy(cursorCid)
        .then((w) => { if (!stop) setWhy(w); })
        .catch((e) => { if (!stop) { setWhy(null); setWhyErr(humanizeError(e)); } });
    }, 180);
    return () => { stop = true; clearTimeout(t); };
  }, [cursorCid]);

  // The vision model's stored verdicts for the page, one request, marked on the tiles (tool #13).
  const [vlm, setVlm] = useState<Record<string, VlmCheck>>({});
  useEffect(() => {
    if (!page.length) { setVlm({}); return; }
    let stop = false;
    getVlmChecks(page.map((r) => r.object_id))
      .then((r) => { if (!stop) setVlm(r.checks); })
      .catch(() => { if (!stop) setVlm({}); });
    return () => { stop = true; };
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
        toast(`${res.applied} applied, ${lost} skipped (already decided by someone else, or removed)`, "error");
        setDecided((d) => d - lost);
      }
      void loadPrecision();
      // The same candidates are listed on the review queue's error tab.
      void invalidate(REVIEW_QUEUE_KEY);
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
      subtitle={kind ? `${detectorName(kind)} · confirm or dismiss, one key each` : "Labels that checks think are wrong: confirm or dismiss, one key each"}
      filters={
        // Pinned above the grid, so the counts and the keys stay in view however far it scrolls.
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 w-full">
          <span className="text-ink tabular-nums">{queue.length.toLocaleString()} left</span>
          <span className="text-ink-3 tabular-nums">{decided.toLocaleString()} decided</span>
          {perMin != null && <span className="text-accent-2 tabular-nums">{perMin} a minute</span>}
          {selection.size > 0 && <span className="text-warn">{selection.size} selected</span>}
          {current && (
            <span className="text-ink-2" title={current.kind}>
              {detectorName(current.kind)}
              <span className="text-ink-3"> · score {current.score?.toFixed(2)}</span>
            </span>
          )}
          <span className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-1 text-ink-3">
            <span><kbd className={KBD}>←↑↓→</kbd> or <kbd className={KBD}>HJKL</kbd> move</span>
            <span><kbd className={KBD}>Shift</kbd> + move: range</span>
            <span><kbd className={KBD}>Space</kbd> select</span>
            <span><kbd className={KBD}>A</kbd> select page</span>
            <span><kbd className={`${KBD} text-block`}>C</kbd> confirm error</span>
            <span><kbd className={KBD}>D</kbd> dismiss</span>
            <span><kbd className={KBD}>O</kbd> open object</span>
            <span><kbd className={KBD}>Esc</kbd> clear</span>
          </span>
        </div>
      }
    >
      <div className="p-4 space-y-3">
        {me?.role === "admin" && <DetectorBenchmark />}
        {/* The detector scoreboard. Shown above the grid because which detector you are judging is the
            decision that matters most, and because a detector nobody has ruled on should say so rather
            than showing an interval that looks like a bad score. */}
        {precision && (
          <section className="panel">
            <div className="panel-head">
              <h2 className="section-title">Checks</h2>
              <span className="section-desc ml-2">How often each check is right, from people&apos;s decisions</span>
              {precision.caveat && <span className="ml-auto text-[12px] text-warn">{precision.caveat}</span>}
            </div>
            <div className="p-3 flex flex-wrap gap-2">
              <button
                onClick={() => router.push("/quality/errors")}
                aria-pressed={!kind}
                className={`chip ${kind ? "" : "chip-on"}`}
              >
                All checks
              </button>
              {kinds.map((k) => {
                const d = precision.per_kind[k];
                return (
                  <button
                    key={k}
                    onClick={() => router.push(`/quality/errors?kind=${encodeURIComponent(k)}`)}
                    title={`${k}${d.note ? ` · ${d.note}` : ` · ${d.decided} decisions`}`}
                    aria-pressed={k === kind}
                    className={`chip ${k === kind ? "chip-on" : ""}`}
                  >
                    {detectorName(k)}
                    <span className="text-ink-3 tabular-nums">{d.pending.toLocaleString()} waiting</span>
                    {d.usable && d.precision.p != null
                      ? <span className="text-ink">· {pct(d.precision.p)} right ({pct(d.precision.lo)} to {pct(d.precision.hi)})</span>
                      : <span className="text-ink-3">· not measured yet</span>}
                  </button>
                );
              })}
            </div>
          </section>
        )}

        <label className="block">
          <span className="sr-only">Reason</span>
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Why (optional, saved on every candidate you decide next, e.g. 'scores frame similarity, not the object')"
            className="field w-full text-[12.5px]"
          />
        </label>
        <p className="text-[12px] text-ink-3">
          Confirming applies the suggested class where there is one; dismissing says the check was wrong.
          With nothing selected, a key acts on the tile under the cursor.
        </p>

        {loading && <div className="text-[13px] text-ink-3" role="status">Loading the queue...</div>}

        {!loading && !page.length && (
          <div className="panel">
            <EmptyState icon="check" title={kind ? `Nothing waiting for "${detectorName(kind)}"` : "Nothing waiting"}
              action={kind ? <button onClick={() => router.push("/quality/errors")} className="btn">Show all checks</button> : undefined}>
              Every possible error here has been decided.
            </EmptyState>
          </div>
        )}

        <div className="flex flex-col xl:flex-row gap-3 items-start">
        {sheet && page.length > 0 && (
          <div className="panel p-2 overflow-x-auto w-fit max-w-full">
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
                    title={`${detectorName(row.kind)} · score ${row.score?.toFixed(2)}${row.proposed_label?.class_name ? ` · suggests ${humanize(row.proposed_label.class_name)}` : ""}`}
                    aria-label={`${detectorName(row.kind)}${row.proposed_label?.class_name ? `, suggests ${humanize(row.proposed_label.class_name)}` : ""}${isSel ? ", selected" : ""}`}
                    aria-pressed={isSel}
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
                      <span className="flex w-full h-full items-center justify-center text-center px-2 text-[11px] text-ink-3 bg-bg-2">
                        No image for this object
                      </span>
                    )}
                    {row.proposed_label?.class_name && (
                      <span className="absolute bottom-0 left-0 right-0 bg-bg/85 text-[11px] text-accent-2 truncate px-1 leading-4">
                        → {humanize(row.proposed_label.class_name)}
                      </span>
                    )}
                    {isSel && <span className="absolute top-1 right-1 rounded-full bg-warn text-bg text-[11px] font-semibold w-4 h-4 leading-4 text-center" aria-hidden>✓</span>}
                    {vlm[row.object_id] && (
                      <span className={`absolute top-1 left-1 rounded px-1 text-[11px] leading-4 bg-bg/85 ${
                        { pass: "text-pass", warn: "text-warn", block: "text-block", muted: "text-ink-3", info: "text-info" }[verdictTone(vlm[row.object_id].verdict)]}`}
                        title={`Vision model: ${vlm[row.object_id].reason}`}>
                        VLM {vlm[row.object_id].verdict === "correct" ? "ok" : vlm[row.object_id].verdict === "incorrect" ? "no" : "?"}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}
        {page.length > 0 && (
          <aside className="panel w-full xl:w-[480px] xl:sticky xl:top-2 shrink-0" aria-label="Why this was flagged">
            <div className="panel-head">
              <h2 className="section-title">Why this was flagged</h2>
              <span className="section-desc ml-2">the tile under the cursor</span>
            </div>
            <div className="p-3">
              {whyErr && <div className="text-[12px] text-block" role="alert">{whyErr}</div>}
              {(!why || why.candidate_id !== cursorCid) && !whyErr && <div className="text-[12px] text-ink-3" role="status">Loading the explanation...</div>}
              {why && why.candidate_id === cursorCid && <ErrorWhyPanel why={why} />}
            </div>
          </aside>
        )}
        </div>

      </div>
    </PageShell>
  );
}
