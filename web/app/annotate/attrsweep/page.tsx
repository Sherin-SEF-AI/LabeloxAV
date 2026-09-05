"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import { api, humanizeError } from "@/lib/api";
import { attrKeymap, noKeymapReason } from "@/lib/attrKeys";
import {
  cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle,
} from "@/lib/gridSelection";
import { toast } from "@/lib/toast";
import type { AttrCoverage, AttrQueue } from "@/lib/types";

// Attribute sweep: one attribute, a grid of crops, one key per answer.
//
// The write path for attributes has been complete and revertible for a long time and had exactly one
// caller, the modal that opens after somebody has already made a correction. Nothing asked the prior
// question, which is where the attributes are missing, so there was no way to go and fill them. Measured
// over the corpus: 282,061 objects are in scope for `load_type` and 0 carry it; 139,613 for
// `occupant_count` and 0 carry it, which is why `triple_riding`, derived from it, is empty too.
//
// The track is the unit wherever the attribute describes the object rather than the moment. A truck's
// load does not change between frames, so the queue offers one crop per track, the largest box on it, and
// the answer lands on every member. Over the real corpus the first eight `load_type` crops cover 2,199
// objects between them.
//
// A sweep fills and does not overwrite. Re-running one is safe, and a track-wide answer can never replace
// a value somebody set deliberately on one frame.

const CELL = 128;

type Placement = { row: number; col: number; ok: boolean };

function AttrSweep() {
  const router = useRouter();
  const params = useSearchParams();
  const sessionId = params.get("session_id") ?? undefined;

  const [cov, setCov] = useState<AttrCoverage[] | null>(null);
  const [attr, setAttr] = useState<string>(params.get("attr") ?? "");
  const [klass, setKlass] = useState<string>(params.get("class_name") ?? "");
  const [queue, setQueue] = useState<AttrQueue | null>(null);
  const [loading, setLoading] = useState(false);

  const [sheet, setSheet] = useState<string | null>(null);
  const [placements, setPlacements] = useState<Record<string, Placement>>({});
  const [cols, setCols] = useState(1);
  const [cursor, setCursor] = useState(0);
  const [selection, setSelection] = useState<ReadonlySet<number>>(new Set());
  const [anchor, setAnchor] = useState<number | null>(null);
  const [multiPick, setMultiPick] = useState<unknown[]>([]);
  const [decided, setDecided] = useState(0);
  const [covered, setCovered] = useState(0);
  const startedAt = useRef<number>(Date.now());
  const batchStart = useRef<number>(Date.now());

  useEffect(() => {
    api.attrCoverage(sessionId)
      .then((r) => setCov(r.attributes))
      .catch((e) => toast(humanizeError(e), "error"));
  }, [sessionId]);

  const loadQueue = useCallback(async () => {
    if (!attr) { setQueue(null); return; }
    setLoading(true);
    try {
      const q = await api.attrQueue({ attr, class_name: klass || undefined, session_id: sessionId });
      setQueue(q);
      setCursor(0);
      setSelection(new Set());
      setAnchor(null);
      setMultiPick([]);
      startedAt.current = Date.now();
      batchStart.current = Date.now();
    } catch (e) {
      toast(humanizeError(e), "error");
      setQueue(null);
    } finally {
      setLoading(false);
    }
  }, [attr, klass, sessionId]);

  useEffect(() => { void loadQueue(); }, [loadQueue]);

  const items = useMemo(() => queue?.items ?? [], [queue]);

  useEffect(() => {
    if (!items.length) { setSheet(null); setPlacements({}); return; }
    let cancelled = false;
    (async () => {
      try {
        const s = await api.cropSheet(items.map((i) => i.object_id), CELL);
        if (cancelled) return;
        setSheet(s.sheet);
        setCols(s.cols || 1);
        const m: Record<string, Placement> = {};
        for (const p of s.placements) m[p.object_id] = { row: p.row, col: p.col, ok: p.ok };
        setPlacements(m);
      } catch (e) {
        if (!cancelled) toast(humanizeError(e), "error");
      }
    })();
    return () => { cancelled = true; };
  }, [items.map((i) => i.object_id).join(",")]);

  const spec = queue ? { type: queue.type, values: queue.values, range: queue.range } : null;
  const keymap = useMemo(() => (spec ? attrKeymap(spec) : null), [queue]);

  /** Land one value on the selected tiles, or on the tile under the cursor when nothing is selected. */
  const answer = useCallback(async (value: unknown) => {
    if (!queue) return;
    const idx = targetIndices(selection, cursor);
    const rows = idx.map((i) => items[i]).filter(Boolean);
    if (!rows.length) return;

    // The id the server resolves: a track for an attribute that describes the object, an object for one
    // that describes the moment. A row with no track falls back to itself, so an untracked object is
    // still answerable rather than being silently skipped.
    const unit = queue.unit;
    const ids = rows.map((r) => (unit === "track" && r.track_id ? r.track_id : r.object_id));
    const perObject = unit === "track" && rows.some((r) => !r.track_id);
    const willCover = rows.reduce((n, r) => n + r.covers, 0);

    const elapsed = Math.max(0, Date.now() - batchStart.current);
    batchStart.current = Date.now();

    // Optimistic: the tiles go now and the request settles behind them, because a round trip per answer
    // is the cost this mode exists to remove.
    const gone = new Set(idx);
    setQueue((q) => (q ? { ...q, items: q.items.filter((_, i) => !gone.has(i)),
                           remaining: Math.max(0, q.remaining - willCover) } : q));
    setSelection(new Set());
    setAnchor(null);
    setMultiPick([]);
    setCursor((c) => cursorAfterRemoval(c, idx, Math.max(0, items.length - idx.length)));
    setDecided((d) => d + rows.length);
    setCovered((c) => c + willCover);

    try {
      const res = await api.attrApply({
        attr: queue.attribute, value,
        unit: perObject ? "object" : unit,
        ids: perObject ? rows.map((r) => r.object_id) : ids,
        time_spent_ms: elapsed,
      });
      if (res.updated === 0) {
        // Not silence. A sweep working a shrinking queue will hit "already answered", and "did nothing"
        // and "done" are different outcomes.
        toast(res.reason ?? "nothing to write", "info");
        setCovered((c) => c - willCover);
        setDecided((d) => d - rows.length);
      } else if (res.run_id) {
        toast(`${res.updated} objects set to ${String(value)}`, "success", 12000,
              { label: "undo", run: () => api.agentRevert(res.run_id!).then(() => { void loadQueue(); }) });
      }
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, [queue, selection, cursor, items, loadQueue]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      const n = items.length;
      if (!n || !queue) return;
      const k = e.key.toLowerCase();

      if (k === "arrowleft" || k === "h") { e.preventDefault(); step("left", e.shiftKey); return; }
      if (k === "arrowright" || k === "l") { e.preventDefault(); step("right", e.shiftKey); return; }
      if (k === "arrowup") { e.preventDefault(); step("up", e.shiftKey); return; }
      if (k === "arrowdown") { e.preventDefault(); step("down", e.shiftKey); return; }
      if (k === " ") { e.preventDefault(); setSelection((s) => toggle(s, cursor)); setAnchor(cursor); return; }
      if (k === "escape") { setSelection(new Set()); setAnchor(null); setMultiPick([]); return; }
      if (k === "o") {
        const it = items[cursor];
        if (it) router.push(`/frame/${it.frame_id}?focus=${it.object_id}`);
        return;
      }
      if (k === "s") { e.preventDefault(); skip(); return; }

      if (!keymap) return;
      const hit = keymap.keys.find((b) => b.key === k);
      if (!hit) {
        if (keymap.multi && k === "enter" && multiPick.length) { e.preventDefault(); void answer([...multiPick]); }
        return;
      }
      e.preventDefault();
      if (keymap.multi) {
        // A signboard carries Kannada and English together, so a press adds to the answer and Enter
        // commits it. Committing on the first press would record the wrong half.
        setMultiPick((p) => (p.includes(hit.value) ? p.filter((v) => v !== hit.value) : [...p, hit.value]));
      } else {
        void answer(hit.value);
      }

      function step(dir: "left" | "right" | "up" | "down", extend: boolean) {
        const next = moveCursor(cursor, dir, n, cols);
        setCursor(next);
        if (!extend) { setSelection(new Set()); setAnchor(null); return; }
        const a = anchor ?? cursor;
        setAnchor(a);
        setSelection(new Set(rangeBetween(a, next)));
      }
      function skip() {
        // Off the page, not answered. An annotator who cannot tell from the crop must be able to move on
        // without recording a guess, and the object stays in the queue for whoever looks next.
        const idx = targetIndices(selection, cursor);
        const gone = new Set(idx);
        setQueue((q) => (q ? { ...q, items: q.items.filter((_, i) => !gone.has(i)) } : q));
        setSelection(new Set());
        setAnchor(null);
        setCursor((c) => cursorAfterRemoval(c, idx, Math.max(0, n - idx.length)));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, cols, anchor, items, selection, keymap, multiPick, queue, answer, router]);

  const perMin = useMemo(() => {
    const mins = (Date.now() - startedAt.current) / 60000;
    return mins > 0.05 && decided > 0 ? Math.round(decided / mins) : null;
  }, [decided]);

  const current = items[cursor];
  const targets = targetIndices(selection, cursor).map((i) => items[i]).filter(Boolean);
  const willCover = targets.reduce((n, r) => n + r.covers, 0);

  return (
    <PageShell active="ATTR SWEEP" title="Attribute sweep"
      subtitle="one attribute, one key, and the track as the unit where the attribute allows it">
      <div className="p-3 flex gap-3 items-start">
        {/* what work exists */}
        <aside className="panel w-72 shrink-0 max-h-[80vh] overflow-auto">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            unanswered
          </div>
          {!cov && <div className="px-3 py-3 font-mono text-[11px] text-ink-3">counting...</div>}
          {cov?.filter((c) => c.missing > 0).map((c) => (
            <button key={c.attribute} onClick={() => { setAttr(c.attribute); setKlass(""); }}
              className={`w-full text-left px-3 py-1.5 font-mono text-[11px] border-b hairline hover:bg-bg-2 ${
                c.attribute === attr ? "bg-bg-2 text-ink" : "text-ink-2"}`}>
              <div className="flex items-center gap-2">
                <span className="truncate">{c.attribute}</span>
                <span className={`ml-auto text-[9px] uppercase ${c.track_constant ? "text-accent" : "text-ink-3"}`}>
                  {c.track_constant ? "track" : "frame"}
                </span>
              </div>
              <div className="text-ink-3 text-[10px]">
                {c.missing.toLocaleString()} of {c.in_scope.toLocaleString()} unanswered
              </div>
            </button>
          ))}
          {cov && !cov.some((c) => c.missing > 0) && (
            <div className="px-3 py-3 font-mono text-[11px] text-ink-3">
              Every attribute in scope is answered.
            </div>
          )}
        </aside>

        <div className="flex-1 min-w-0 space-y-2">
          {!attr && (
            <div className="panel px-3 py-6 font-mono text-[11px] text-ink-3 text-center">
              Pick an attribute on the left. The list is ordered by how much of it is unanswered.
            </div>
          )}

          {queue && (
            <>
              <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] text-ink-3">
                <span className="text-ink">{queue.attribute}</span>
                <span className={queue.unit === "track" ? "text-accent" : ""}>
                  {queue.unit === "track" ? "one answer per track" : "one answer per object"}
                </span>
                <span>{queue.remaining.toLocaleString()} unanswered</span>
                {decided > 0 && <span>{decided} answered, covering {covered.toLocaleString()} objects</span>}
                {perMin != null && <span className="text-accent">{perMin}/min</span>}
                {selection.size > 0 && <span className="text-warn">{selection.size} selected</span>}
                {queue.unit === "track" && queue.untracked > 0 && (
                  <span className="text-warn" title="objects with no track cannot be swept track-wise">
                    {queue.untracked.toLocaleString()} untracked
                  </span>
                )}
                {/* Class filter, from the classes that actually have work. */}
                {klass && <span className="text-ink-2">class: {klass}
                  <button onClick={() => setKlass("")} className="ml-1 text-ink-3 hover:text-block">x</button>
                </span>}
              </div>

              {!klass && (cov?.find((c) => c.attribute === attr)?.classes.length ?? 0) > 0 && (
                <div className="flex flex-wrap gap-1 font-mono text-[10px]">
                  <span className="text-ink-3 py-0.5">narrow to a class:</span>
                  {cov!.find((c) => c.attribute === attr)!.classes.slice(0, 8).map((c) => (
                    <button key={c.class_id} onClick={() => setKlass(c.class_name)}
                      className="border border-line px-1.5 py-0.5 rounded text-ink-2 hover:border-accent">
                      {c.class_name} <span className="text-ink-3">{c.missing.toLocaleString()}</span>
                    </button>
                  ))}
                </div>
              )}

              {/* the answer keys */}
              {keymap ? (
                <div className="panel px-3 py-2 flex flex-wrap items-center gap-2 font-mono text-[11px]">
                  {keymap.keys.map((b) => (
                    <button key={b.key} onClick={() => (keymap.multi
                      ? setMultiPick((p) => (p.includes(b.value) ? p.filter((v) => v !== b.value) : [...p, b.value]))
                      : void answer(b.value))}
                      className={`border px-2 py-1 rounded hover:border-accent ${
                        keymap.multi && multiPick.includes(b.value)
                          ? "border-accent text-accent bg-accent/10" : "border-line text-ink-2"}`}>
                      <b className="text-accent">{b.key}</b> {b.label}
                    </button>
                  ))}
                  {keymap.multi && (
                    <button onClick={() => multiPick.length && void answer([...multiPick])}
                      disabled={!multiPick.length}
                      className="border border-pass/60 text-pass px-2 py-1 rounded hover:bg-pass/10 disabled:opacity-40">
                      <b>enter</b> commit {multiPick.length ? `(${multiPick.length})` : ""}
                    </button>
                  )}
                  <span className="ml-auto text-ink-3">
                    {targets.length === 1
                      ? `answers 1 crop, ${willCover.toLocaleString()} object${willCover === 1 ? "" : "s"}`
                      : `answers ${targets.length} crops, ${willCover.toLocaleString()} objects`}
                  </span>
                </div>
              ) : (
                <div className="panel px-3 py-2 font-mono text-[11px] text-warn">
                  No keyboard for {queue.attribute}: {noKeymapReason({ type: queue.type, range: queue.range })}.
                  Press <b className="text-ink">O</b> to open the frame editor on the crop under the cursor.
                </div>
              )}

              {loading && <div className="font-mono text-[11px] text-ink-3">loading the grid...</div>}

              {!loading && !items.length && (
                <div className="panel px-3 py-6 font-mono text-[11px] text-ink-3 text-center">
                  {queue.remaining > 0
                    ? <>This page is done. <button onClick={() => void loadQueue()}
                        className="text-accent underline">load the next {queue.remaining.toLocaleString()}</button></>
                    : queue.reason ?? "Nothing left unanswered in this scope."}
                </div>
              )}

              {sheet && items.length > 0 && (
                <div className="panel p-2 overflow-x-auto">
                  <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, ${CELL}px)` }}>
                    {items.map((row, i) => {
                      const p = placements[row.object_id];
                      const isCursor = i === cursor;
                      const isSel = selection.has(i);
                      return (
                        <button key={row.object_id}
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
                          title={`${row.class_name} · covers ${row.covers} object${row.covers === 1 ? "" : "s"}`}
                          className={`relative block ${isCursor ? "ring-2 ring-accent" : isSel ? "ring-2 ring-warn" : "ring-1 ring-line"}`}
                          style={{ width: CELL, height: CELL }}>
                          {p?.ok ? (
                            <span className="block w-full h-full"
                              style={{
                                backgroundImage: `url(${sheet})`,
                                backgroundPosition: `-${p.col * CELL}px -${p.row * CELL}px`,
                              }} />
                          ) : (
                            <span className="flex w-full h-full items-center justify-center text-[9px] text-ink-3">
                              no crop
                            </span>
                          )}
                          {/* What this one answer covers, so the leverage is visible before the key. */}
                          {row.covers > 1 && (
                            <span className="absolute bottom-0 left-0 bg-accent/80 text-bg text-[9px] px-1">
                              x{row.covers}
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
                <span><b className="text-ink">arrows</b> move</span>
                <span><b className="text-ink">shift+move</b> range</span>
                <span><b className="text-ink">space</b> select</span>
                <span><b className="text-ink">S</b> skip without answering</span>
                <span><b className="text-ink">O</b> open the frame</span>
                <span><b className="text-ink">esc</b> clear</span>
                {current && <span className="text-ink-2">{current.class_name}</span>}
              </div>
            </>
          )}
        </div>
      </div>
    </PageShell>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<div className="p-4 font-mono text-[11px] text-ink-3">loading...</div>}>
      <AttrSweep />
    </Suspense>
  );
}
