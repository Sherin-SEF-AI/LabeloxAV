"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import type { Ontology, OntologyClass, SimilarResponse } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import ClassPopover from "@/components/editor/properties/ClassPopover";
import { cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle } from "@/lib/gridSelection";
import { toast } from "@/lib/toast";
import { acceptState, useCurrentUser } from "@/lib/user";

// Similarity search. One surface for every query kind: natural-language text (SigLIP 2), an uploaded image,
// a frame, or an object crop (DINOv3), all reranked for diversity and a similarity floor so the grid shows
// distinct neighbours rather than fifteen copies of the same thing. The query is remembered so moving a
// control (threshold, diversity, same-class) re-runs it in place instead of making you start over.

type QueryKind = "text" | "image" | "frame" | "object";
type LastQuery =
  | { kind: "text"; q: string }
  | { kind: "image"; b64: string }
  | { kind: "frame"; id: string }
  | { kind: "object"; id: string };

export default function SearchPage() {
  return (
    <Suspense fallback={null}>
      <SearchBody />
    </Suspense>
  );
}

// Label by example lives here now as well as in CorrectionModal.
//
// The modal is a complete version of this flow - crop grid, similarity slider, same-camera and same-class
// toggles, bulk apply, twelve-second undo - and it opens only after you have already made a correction.
// So the way to find every other object that looks like this one was to relabel it wrongly and change
// your mind. This page had the grid and the rerank controls and no way to act on what it found: clicking
// a result navigated away one object at a time.
//
// Embedding coverage is 567,488 of 578,436 objects, 98.1%, so the neighbours are there to be found.

function SearchBody() {
  const router = useRouter();
  const params = useSearchParams();
  const me = useCurrentUser();
  const frameParam = params.get("frame");
  const objectParam = params.get("object");

  const [res, setRes] = useState<SimilarResponse | null>(null);
  const [mode, setMode] = useState<"visual" | "semantic" | "fused">("visual");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [parsed, setParsed] = useState<{ filters: Record<string, string>; classes: string[] } | null>(null);

  // rerank controls
  const [k, setK] = useState(24);
  const [minSim, setMinSim] = useState(0);
  const [diversity, setDiversity] = useState(true);
  const [sameClass, setSameClass] = useState(false);
  const [excludeTrack, setExcludeTrack] = useState(true);
  const [city, setCity] = useState("");

  // selection over object results, so a found set can be labelled together rather than one at a time
  const [selection, setSelection] = useState<ReadonlySet<number>>(new Set());
  const [cursor, setCursor] = useState(0);
  const [anchor, setAnchor] = useState<number | null>(null);
  const [pickOpen, setPickOpen] = useState(false);
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [applying, setApplying] = useState(false);
  const pickAnchor = useRef<HTMLButtonElement | null>(null);
  const batchStart = useRef<number>(Date.now());

  const last = useRef<LastQuery | null>(null);

  const controls = useCallback(
    () => ({ k, min_sim: minSim, diversity, same_class: sameClass, exclude_track: excludeTrack, city: city.trim() || undefined }),
    [k, minSim, diversity, sameClass, excludeTrack, city],
  );

  const run = useCallback(async (query: LastQuery) => {
    last.current = query;
    setBusy(true);
    setNote(null);
    try {
      if (query.kind === "text") {
        setParsed(null);
        // NL text search parses scene/class filters then reranks SigLIP 2; it is frame-scoped.
        const r = await api.searchSemantic(query.q, k);
        setParsed({ filters: r.filters, classes: r.classes });
        setRes({ kind: "frame", mode: "semantic", results: r.results });
      } else if (query.kind === "image") {
        setRes(await api.searchSimilar({ image_b64: query.b64, mode, ...controls() }));
      } else if (query.kind === "frame") {
        setRes(await api.searchSimilar({ frame_id: query.id, mode, ...controls() }));
      } else {
        setRes(await api.searchSimilar({ object_id: query.id, ...controls() }));
      }
    } catch (e) {
      setNote(humanizeError(e));
    } finally {
      setBusy(false);
    }
  }, [mode, k, controls]);

  // Open from ?frame= or ?object= (e.g. the "find similar" action on a frame or object page).
  useEffect(() => {
    if (objectParam) run({ kind: "object", id: objectParam });
    else if (frameParam) run({ kind: "frame", id: frameParam });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameParam, objectParam]);

  // A control changed: re-run the current query in place (except NL text, which does not take rerank args).
  const rerun = useCallback(() => { if (last.current && last.current.kind !== "text") run(last.current); }, [run]);
  useEffect(() => { rerun(); }, [mode, k, minSim, diversity, sameClass, excludeTrack]); // eslint-disable-line react-hooks/exhaustive-deps

  const onFile = async (f: File) => {
    const b64 = await new Promise<string>((resolve) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result).split(",")[1]);
      r.readAsDataURL(f);
    });
    run({ kind: "image", b64 });
  };

  const kind: QueryKind | null = last.current?.kind ?? null;
  const isObject = res?.kind === "object";
  const results = useMemo(() => res?.results ?? [], [res]);

  // The ontology is only needed once there is something to relabel, so it is fetched with the first
  // object result rather than on every visit to a text search.
  useEffect(() => {
    if (!isObject || onto) return;
    api.ontology().then(setOnto).catch((e) => {
      // Said out loud rather than swallowed. Without the ontology the "set class" button stays disabled,
      // and a disabled button with no reason reads as "this page cannot relabel", which is a different
      // and wrong conclusion from "the ontology did not load".
      toast(`class list unavailable, so relabelling is off: ${humanizeError(e)}`, "error");
    });
  }, [isObject, onto]);

  useEffect(() => { setSelection(new Set()); setAnchor(null); setCursor(0); }, [res]);

  /**
   * Apply one verdict to the selected results, or to the one under the cursor.
   *
   * Goes through bulk review, so it inherits the optimistic lock, the role clamp, the attribute
   * revalidation and the revertible run. The undo is offered for twelve seconds because that is the only
   * offer of it, and a mislabelled found set is the exact thing this mode can produce quickly.
   */
  const applyVerdict = useCallback(async (action: "confirm" | "reject" | "reclassify", className?: string) => {
    const idx = targetIndices(selection, cursor);
    const ids = idx.map((i) => results[i]?.object_id).filter((v): v is string => !!v);
    if (!ids.length) return;
    const elapsed = Math.max(0, Date.now() - batchStart.current);
    batchStart.current = Date.now();
    setApplying(true);
    try {
      const r = await api.bulkReview(
        ids, action, className,
        action === "reject" ? "rejected" : acceptState(me?.role),
        undefined, { time_spent_ms: elapsed });
      const lost = (r.skipped_stale?.length ?? 0) + (r.skipped_missing?.length ?? 0);
      const what = action === "reclassify" ? `set to ${className}` : action === "reject" ? "rejected" : "accepted";
      if (r.run_id) {
        toast(`${r.updated} ${what}${lost ? `, ${lost} skipped` : ""}`, lost ? "error" : "success", 12000,
              { label: "undo", run: () => api.agentRevert(r.run_id!).then(() => {}) });
      } else {
        toast(`nothing changed${lost ? `; ${lost} had moved on since this search ran` : ""}`, "error");
      }
      // The results stay on screen. A found set is a set somebody chose, and clearing it after one
      // verdict would make a second pass over the same neighbours impossible.
      setSelection(new Set());
      setAnchor(null);
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setApplying(false);
    }
  }, [selection, cursor, results, me]);

  useEffect(() => {
    if (!isObject) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      const n = results.length;
      if (!n) return;
      const k2 = e.key.toLowerCase();
      if (k2 === "arrowleft" || k2 === "h") { e.preventDefault(); step("left", e.shiftKey); }
      else if (k2 === "arrowright" || k2 === "l") { e.preventDefault(); step("right", e.shiftKey); }
      else if (k2 === "arrowup") { e.preventDefault(); step("up", e.shiftKey); }
      else if (k2 === "arrowdown") { e.preventDefault(); step("down", e.shiftKey); }
      else if (k2 === " ") { e.preventDefault(); setSelection((sel) => toggle(sel, cursor)); setAnchor(cursor); }
      else if (k2 === "escape") { setSelection(new Set()); setAnchor(null); }
      else if (k2 === "a") { e.preventDefault(); void applyVerdict("confirm"); }
      else if (k2 === "r") { e.preventDefault(); void applyVerdict("reject"); }
      else if (k2 === "c") { e.preventDefault(); setPickOpen(true); }
      else if (k2 === "o") {
        const r = results[cursor];
        if (r?.object_id) router.push(`/object/${r.object_id}`);
      }
      function step(dir: "left" | "right" | "up" | "down", extend: boolean) {
        const next = moveCursor(cursor, dir, n, 8);
        setCursor(next);
        if (!extend) { setSelection(new Set()); setAnchor(null); return; }
        const a = anchor ?? cursor;
        setAnchor(a);
        setSelection(new Set(rangeBetween(a, next)));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isObject, results, cursor, anchor, applyVerdict, router]);

  const Chip = ({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) => (
    <button onClick={onClick}
      className={`border px-2 py-0.5 font-mono text-[11px] ${on ? "border-accent text-accent" : "border-line text-ink-3 hover:text-ink-2"}`}>
      {children}
    </button>
  );

  const filters = (
    <div className="flex flex-col gap-2 w-full">
      {/* row 1: the query */}
      <div className="flex items-center gap-2 flex-wrap">
        <input value={q} onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && q.trim() && run({ kind: "text", q })}
          placeholder='natural language, e.g. "night rain autorickshaw" or "dense urban traffic"'
          className="flex-1 min-w-[18rem] bg-bg border border-line px-3 py-1.5 font-mono text-xs text-ink" />
        <button onClick={() => q.trim() && run({ kind: "text", q })} disabled={busy}
          className="border border-accent text-accent px-3 py-1.5 font-mono text-xs hover:bg-accent/10 disabled:opacity-50">search</button>
        <label className="border border-line px-2 py-1.5 font-mono text-xs text-ink-3 hover:border-accent cursor-pointer">
          upload image
          <input type="file" accept="image/*" className="hidden"
            onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
        <span className="font-mono text-[10px] text-ink-3">space:</span>
        {(["visual", "semantic", "fused"] as const).map((m) => <Chip key={m} on={mode === m} onClick={() => setMode(m)}>{m}</Chip>)}
      </div>

      {/* row 2: the rerank controls */}
      <div className="flex items-center gap-3 flex-wrap font-mono text-[11px] text-ink-3">
        <Chip on={diversity} onClick={() => setDiversity((v) => !v)}>diverse</Chip>
        {isObject && <Chip on={sameClass} onClick={() => setSameClass((v) => !v)}>same class</Chip>}
        {isObject && <Chip on={excludeTrack} onClick={() => setExcludeTrack((v) => !v)}>exclude self</Chip>}
        <label className="flex items-center gap-1">
          min sim <span className="text-ink-2 w-8 text-right">{minSim.toFixed(2)}</span>
          <input type="range" min={0} max={0.95} step={0.05} value={minSim}
            onChange={(e) => setMinSim(parseFloat(e.target.value))} className="w-28 accent-accent" />
        </label>
        <label className="flex items-center gap-1">
          results
          <select value={k} onChange={(e) => setK(parseInt(e.target.value))}
            className="bg-bg border border-line px-1 py-0.5 text-ink">
            {[12, 24, 48, 96].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <input value={city} onChange={(e) => setCity(e.target.value)} onKeyDown={(e) => e.key === "Enter" && rerun()}
          placeholder="city (optional)" className="bg-bg border border-line px-2 py-0.5 text-ink w-28" />
        {busy && <span className="text-warn">searching...</span>}
        {note && <span className="text-block">{note}</span>}
      </div>

      {parsed && (parsed.classes.length > 0 || Object.keys(parsed.filters).length > 0) && (
        <div className="font-mono text-[10px] text-ink-3">
          parsed:{" "}
          {Object.entries(parsed.filters).map(([a, v]) => <span key={a} className="text-info mr-2">{a}={v}</span>)}
          {parsed.classes.map((c) => <span key={c} className="text-pass mr-2">class={c}</span>)}
        </div>
      )}
    </div>
  );

  return (
    <PageShell active="SEARCH" title="Search" filters={filters}>
      <div className="p-4 space-y-4">
        {results.length > 0 ? (
          <div className="panel p-3">
            <div className="font-mono text-[11px] text-ink-3 mb-2 flex items-center gap-3 flex-wrap">
              <span>{results.length} {res!.kind}s</span>
              <span className="text-ink-3">{res!.mode} · DINOv3/SigLIP2</span>
              {diversity && <span className="text-info">deduped</span>}
              {minSim > 0 && <span className="text-info">≥ {minSim.toFixed(2)}</span>}
              {isObject && selection.size > 0 && <span className="text-warn">{selection.size} selected</span>}
              {/* Label the found set. Without this the page could find every object that looks like the
                  one you are fixing and offer nothing to do about it but visit them one at a time. */}
              {isObject && (
                <span className="ml-auto flex items-center gap-1.5 relative">
                  <span className="text-ink-3">
                    {selection.size > 0 ? `apply to ${selection.size}` : "apply to the one under the cursor"}
                  </span>
                  <button ref={pickAnchor} onClick={() => setPickOpen((o) => !o)} disabled={applying || !onto}
                    aria-haspopup="dialog" aria-expanded={pickOpen}
                    title="set a class on the selection (C)"
                    className="border border-accent text-accent px-2 py-0.5 rounded hover:bg-accent/10 disabled:opacity-40">
                    <b>C</b> set class
                  </button>
                  <button onClick={() => void applyVerdict("confirm")} disabled={applying}
                    title="confirm the selection as it is (A)"
                    className="border border-pass/60 text-pass px-2 py-0.5 rounded hover:bg-pass/10 disabled:opacity-40">
                    <b>A</b> accept
                  </button>
                  <button onClick={() => void applyVerdict("reject")} disabled={applying}
                    title="reject the selection (R)"
                    className="border border-line text-ink-3 px-2 py-0.5 rounded hover:border-block hover:text-block disabled:opacity-40">
                    <b>R</b> reject
                  </button>
                  {onto && (
                    <ClassPopover anchorRef={pickAnchor} open={pickOpen} onClose={() => setPickOpen(false)}
                      classes={onto.classes} currentId={null}
                      onPick={(c: OntologyClass) => { void applyVerdict("reclassify", c.name); }}
                      onAdd={(raw: string) => {
                        void api.addClass(raw)
                          .then(async (c) => { setOnto(await api.ontology()); await applyVerdict("reclassify", c.name); })
                          .catch((e) => toast(humanizeError(e), "error"));
                      }} />
                  )}
                </span>
              )}
            </div>
            <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2">
              {results.map((r, i) => (
                <button key={i}
                  onClick={(e) => {
                    // Frame results still navigate: there is no bulk verdict for a frame. Object results
                    // select, because navigating away one crop at a time is what this mode replaces;
                    // double-click still opens the object.
                    if (!isObject) {
                      if (r.frame_id) router.push(`/frame/${r.frame_id}`);
                      return;
                    }
                    if (e.shiftKey) {
                      const a = anchor ?? cursor;
                      setAnchor(a);
                      setSelection(new Set(rangeBetween(a, i)));
                    } else {
                      setSelection((sel) => toggle(sel, i));
                      setAnchor(i);
                    }
                    setCursor(i);
                  }}
                  onDoubleClick={() => { if (r.object_id) router.push(`/object/${r.object_id}`); }}
                  className={`relative border text-left group ${
                    isObject && i === cursor ? "border-accent ring-1 ring-accent"
                      : isObject && selection.has(i) ? "border-warn" : "border-line hover:border-accent"}`}
                  title={`${r.class_name ?? r.scene?.weather ?? ""} · sim ${r.score.toFixed(3)}${
                    isObject ? " · click to select, double-click to open" : ""}`}>
                  {isObject && selection.has(i) && (
                    <span className="absolute top-0 right-0 z-10 bg-warn text-bg text-[9px] px-1">+</span>
                  )}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={r.image_url || r.crop_url} alt="" className="w-full h-20 object-cover bg-bg-2" />
                  <div className="flex items-center justify-between px-1 py-0.5">
                    <span className="font-mono text-[9px] text-ink-3 truncate">{r.class_name ?? ""}</span>
                    <span className="font-mono text-[9px] text-accent">{r.score.toFixed(2)}</span>
                  </div>
                </button>
              ))}
            </div>
            {isObject && (
              <div className="mt-2 font-mono text-[10px] text-ink-3 flex flex-wrap gap-3">
                <span><b className="text-ink">arrows</b> move</span>
                <span><b className="text-ink">shift+move</b> range</span>
                <span><b className="text-ink">space</b> select</span>
                <span><b className="text-accent">C</b> set class</span>
                <span><b className="text-pass">A</b> accept</span>
                <span><b className="text-block">R</b> reject</span>
                <span><b className="text-ink">O</b> open</span>
                <span className="text-ink-2">a verdict with nothing selected applies to the tile under the cursor</span>
              </div>
            )}
          </div>
        ) : (
          !busy && (
            <div className="panel px-3 py-10 text-center font-mono text-xs text-ink-3">
              {res
                ? kind === "object"
                  ? "no similar objects above the threshold. lower min sim, or the object may not be embedded yet."
                  : "no similar results. lower min sim, or the frame may not be embedded yet."
                : "type a description, upload an image, or open a frame or object and click “find similar”."}
            </div>
          )
        )}
      </div>
    </PageShell>
  );
}
