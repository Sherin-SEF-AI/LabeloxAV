"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter, useSearchParams } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import type { Ontology, OntologyClass, SimilarResponse } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import ClassPopover from "@/components/editor/properties/ClassPopover";
import { cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle } from "@/lib/gridSelection";
import { toast } from "@/lib/toast";
import { acceptState, useCurrentUser } from "@/lib/user";
import { invalidate, useQuery } from "@/lib/useQuery";
import Icon from "@/components/shell/Icon";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, plural } from "@/lib/format";

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
  // Set once an object result has been shown, and never unset: see the ontology read below.
  const [wantOnto, setWantOnto] = useState(false);
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
  useEffect(() => { if (isObject) setWantOnto(true); }, [isObject]);
  const ontoQ = useQuery(["ontology"], () => api.ontology(), { enabled: wantOnto });
  const onto: Ontology | null = ontoQ.data ?? null;
  useEffect(() => {
    // Said out loud rather than swallowed. Without the ontology the "set class" button stays disabled,
    // and a disabled button with no reason reads as "this page cannot relabel", which is a different
    // and wrong conclusion from "the ontology did not load".
    if (ontoQ.error) toast(`class list unavailable, so relabelling is off: ${humanizeError(ontoQ.error)}`, "error");
  }, [ontoQ.error]);

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
      // The verdict moved these objects out of the review queues other pages cache.
      void invalidate("triage", "/api/home");
      const lost = (r.skipped_stale?.length ?? 0) + (r.skipped_missing?.length ?? 0);
      const what = action === "reclassify" ? `set to ${className}` : action === "reject" ? "rejected" : "accepted";
      if (r.run_id) {
        toast(`${r.updated} ${what}${lost ? `, ${lost} skipped` : ""}`, lost ? "error" : "success", 12000,
              { label: "undo", run: () => api.agentRevert(r.run_id!).then(() => { void invalidate("triage", "/api/home"); }) });
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

  const Chip = ({ on, onClick, children, tip }: { on: boolean; onClick: () => void; children: React.ReactNode; tip?: string }) => (
    <button type="button" onClick={onClick} aria-pressed={on} data-tip={tip}
      className={`chip ${on ? "chip-on" : ""}`}>
      {children}
    </button>
  );

  const runText = (text: string) => { const t = text.trim(); if (t) run({ kind: "text", q: t }); };

  const filters = (
    <div className="flex flex-col gap-2.5 w-full py-0.5">
      {/* row 1: the query */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="relative flex-1 min-w-[18rem]">
          <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none" aria-hidden>
            <Icon name="search" size={15} />
          </span>
          <input value={q} onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && q.trim() && run({ kind: "text", q })}
            aria-label="Describe what to find"
            placeholder='Describe a scene, e.g. "night rain autorickshaw" or "dense urban traffic"'
            className="field w-full pl-8 text-[13px]" />
        </div>
        <button onClick={() => q.trim() && run({ kind: "text", q })} disabled={busy}
          className="btn btn-primary"><T k="action.search">Search</T></button>
        <label className="btn cursor-pointer" data-tip="Find frames that look like an image from your computer">
          <Icon name="upload" size={14} />
          Search by image
          <input type="file" accept="image/*" className="hidden"
            onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
        <span className="text-[12px] text-ink-3 ml-1">Match on</span>
        <div className="flex items-center gap-1" role="group" aria-label="Embedding space">
          {([
            ["visual", "Look", "Visual similarity (DINOv3): what it looks like"],
            ["semantic", "Meaning", "Semantic similarity (SigLIP 2): what it depicts"],
            ["fused", "Both", "Both spaces combined"],
          ] as const).map(([m, label, tip]) => (
            <Chip key={m} on={mode === m} onClick={() => setMode(m)} tip={tip}>{label}</Chip>
          ))}
        </div>
      </div>

      {/* row 2: the rerank controls */}
      <div className="flex items-center gap-x-4 gap-y-2 flex-wrap text-[12px] text-ink-2">
        <div className="flex items-center gap-1.5">
          <Chip on={diversity} onClick={() => setDiversity((v) => !v)} tip="Drop near-duplicates so the grid shows distinct neighbours">Diverse</Chip>
          {isObject && <Chip on={sameClass} onClick={() => setSameClass((v) => !v)} tip="Only objects of the same class">Same class</Chip>}
          {isObject && <Chip on={excludeTrack} onClick={() => setExcludeTrack((v) => !v)} tip="Leave out the query object's own track">Exclude self</Chip>}
        </div>
        <label className="flex items-center gap-2">
          Min similarity
          <input type="range" min={0} max={0.95} step={0.05} value={minSim}
            onChange={(e) => setMinSim(parseFloat(e.target.value))} className="w-28 accent-accent" />
          <span className="text-ink tabular-nums w-8">{minSim.toFixed(2)}</span>
        </label>
        <label className="flex items-center gap-2">
          Results
          <select value={k} onChange={(e) => setK(parseInt(e.target.value))}
            className="field py-0.5">
            {[12, 24, 48, 96].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <input value={city} onChange={(e) => setCity(e.target.value)} onKeyDown={(e) => e.key === "Enter" && rerun()}
          aria-label="City filter" placeholder="City (optional)" className="field py-0.5 w-32" />
        {busy && <span className="inline-flex items-center gap-1.5 text-ink-2"><span className="running-dot" />Searching...</span>}
      </div>

      {parsed && (parsed.classes.length > 0 || Object.keys(parsed.filters).length > 0) && (
        <div className="flex items-center gap-1.5 flex-wrap text-[12px] text-ink-3">
          Understood as
          {Object.entries(parsed.filters).map(([a, v]) => (
            <span key={a} className="px-1.5 py-px rounded border border-info/40 text-info">{humanize(a)}: {humanize(v)}</span>
          ))}
          {parsed.classes.map((c) => (
            <span key={c} className="px-1.5 py-px rounded border border-pass/40 text-pass">{humanize(c)}</span>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <PageShell active="SEARCH" title="Search" filters={filters}>
      <div className="p-4 space-y-4">
        {note && (
          <div role="alert" className="panel border-block/40 bg-block/5 px-3.5 py-3 text-[12.5px] flex items-start gap-3">
            <span className="text-block shrink-0 mt-px"><Icon name="info" size={15} /></span>
            <div className="min-w-0 flex-1">
              <div className="text-ink font-medium">Search failed</div>
              <div className="text-ink-2 mt-0.5 break-words">{note}</div>
            </div>
            {last.current && (
              <button type="button" className="btn shrink-0" onClick={() => last.current && run(last.current)}>Retry</button>
            )}
          </div>
        )}
        {busy && results.length === 0 ? (
          <div className="panel p-3" role="status" aria-label="searching">
            <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2">
              {Array.from({ length: Math.min(k, 16) }, (_, i) => <span key={i} className="skeleton h-28 rounded" />)}
            </div>
          </div>
        ) : results.length > 0 ? (
          <div className="panel p-3">
            <div className="text-[12.5px] text-ink-3 mb-3 flex items-center gap-3 flex-wrap">
              <span className="text-ink font-medium">{plural(results.length, res!.kind === "object" ? "object" : "frame")}</span>
              <span>{humanize(res!.mode)} match</span>
              {diversity && <span className="text-info">Near-duplicates removed</span>}
              {minSim > 0 && <span className="text-info">Similarity ≥ {minSim.toFixed(2)}</span>}
              {isObject && selection.size > 0 && <span className="text-warn font-medium">{selection.size} selected</span>}
              {/* Label the found set. Without this the page could find every object that looks like the
                  one you are fixing and offer nothing to do about it but visit them one at a time. */}
              {isObject && (
                <span className="ml-auto flex items-center gap-1.5 relative">
                  <span className="text-ink-3 mr-1">
                    {selection.size > 0 ? `Apply to ${selection.size}` : "Apply to the tile under the cursor"}
                  </span>
                  <button ref={pickAnchor} onClick={() => setPickOpen((o) => !o)} disabled={applying || !onto}
                    aria-haspopup="dialog" aria-expanded={pickOpen}
                    title="set a class on the selection (C)"
                    className="btn btn-on">
                    Set class <kbd className="font-mono text-[11px] opacity-70">C</kbd>
                  </button>
                  <button onClick={() => void applyVerdict("confirm")} disabled={applying}
                    title="confirm the selection as it is (A)"
                    className="btn text-pass">
                    <T k="action.accept">Accept</T> <kbd className="font-mono text-[11px] opacity-70">A</kbd>
                  </button>
                  <button onClick={() => void applyVerdict("reject")} disabled={applying}
                    title="reject the selection (R)"
                    className="btn btn-danger">
                    <T k="action.reject">Reject</T> <kbd className="font-mono text-[11px] opacity-70">R</kbd>
                  </button>
                  {onto && (
                    <ClassPopover anchorRef={pickAnchor} open={pickOpen} onClose={() => setPickOpen(false)}
                      classes={onto.classes} currentId={null}
                      onPick={(c: OntologyClass) => { void applyVerdict("reclassify", c.name); }}
                      onAdd={(raw: string) => {
                        void api.addClass(raw)
                          .then(async (c) => { await invalidate("ontology"); await applyVerdict("reclassify", c.name); })
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
                  className={`relative border rounded overflow-hidden text-left group transition-colors ${
                    isObject && i === cursor ? "border-accent ring-1 ring-accent"
                      : isObject && selection.has(i) ? "border-warn" : "border-line hover:border-accent"}`}
                  title={`${r.class_name ?? r.scene?.weather ?? ""} · sim ${r.score.toFixed(3)}${
                    isObject ? " · click to select, double-click to open" : ""}`}>
                  {isObject && selection.has(i) && (
                    <span className="absolute top-1 right-1 z-10 w-5 h-5 rounded-full bg-warn text-bg flex items-center justify-center"
                      aria-label="selected"><Icon name="check" size={12} /></span>
                  )}
                  <Thumb src={r.image_url || r.crop_url} />
                  <div className="flex items-center justify-between gap-1 px-1.5 py-1">
                    <span className="text-[11.5px] text-ink-2 truncate">{r.class_name ? humanize(r.class_name) : ""}</span>
                    <span className="text-[11.5px] text-accent-2 tabular-nums">{r.score.toFixed(2)}</span>
                  </div>
                </button>
              ))}
            </div>
            {isObject && (
              <div className="mt-3 text-[12px] text-ink-3 flex flex-wrap gap-x-4 gap-y-1">
                <span><kbd className="text-ink">Arrows</kbd> move</span>
                <span><kbd className="text-ink">Shift + move</kbd> range</span>
                <span><kbd className="text-ink">Space</kbd> select</span>
                <span><kbd className="text-accent-2">C</kbd> set class</span>
                <span><kbd className="text-pass">A</kbd> accept</span>
                <span><kbd className="text-block">R</kbd> reject</span>
                <span><kbd className="text-ink">O</kbd> open</span>
                <span className="text-ink-2">A verdict with nothing selected applies to the tile under the cursor.</span>
              </div>
            )}
          </div>
        ) : (
          !busy && !note && (
            <div className="panel">
              {res ? (
                <EmptyState icon="search" title={kind === "object" ? "No similar objects" : "No similar results"}
                  action={minSim > 0 ? <button type="button" className="btn" onClick={() => setMinSim(0)}>Reset min similarity</button> : undefined}>
                  {minSim > 0
                    ? "Nothing scored above the similarity floor. Lower it to see more distant matches."
                    : kind === "object"
                      ? "This object may not be embedded yet, so it has no neighbours to compare against."
                      : "This frame may not be embedded yet, so it has no neighbours to compare against."}
                </EmptyState>
              ) : (
                <EmptyState icon="search" title="Find frames and objects by description or example"
                  action={
                    <div className="flex flex-wrap justify-center gap-2 max-w-xl">
                      {EXAMPLES.map((ex) => (
                        <button key={ex} type="button" className="chip" onClick={() => { setQ(ex); runText(ex); }}>
                          {ex}
                        </button>
                      ))}
                    </div>
                  }>
                  Describe a scene in plain words, search by an image, or use Find similar on any frame or object.
                  Try one of these:
                </EmptyState>
              )}
            </div>
          )
        )}
      </div>
    </PageShell>
  );
}

// Starting points for an empty search: the kinds of scene people come here to find.
const EXAMPLES = [
  "night rain autorickshaw",
  "dense urban traffic",
  "pedestrian crossing at dusk",
  "cattle on the road",
  "truck overtaking on highway",
  "wet road with reflections",
];

// A result tile's image. A missing or failed crop used to leave a black box with no explanation.
function Thumb({ src }: { src?: string | null }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div className="w-full h-24 bg-bg-2 flex flex-col items-center justify-center gap-1 text-ink-3">
        <Icon name="eyeOff" size={16} />
        <span className="text-[11px]">No image</span>
      </div>
    );
  }
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt="" loading="lazy" onError={() => setFailed(true)} className="w-full h-24 object-cover bg-bg-2" />;
}
