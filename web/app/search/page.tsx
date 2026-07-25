"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { SimilarResponse } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";

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

function SearchBody() {
  const router = useRouter();
  const params = useSearchParams();
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
      setNote(String(e));
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

  const results = res?.results ?? [];

  return (
    <PageShell active="SEARCH" title="Search" filters={filters}>
      <div className="p-4 space-y-4">
        {results.length > 0 ? (
          <div className="panel p-3">
            <div className="font-mono text-[11px] text-ink-3 mb-2 flex items-center gap-3">
              <span>{results.length} {res!.kind}s</span>
              <span className="text-ink-4">{res!.mode} · DINOv3/SigLIP2</span>
              {diversity && <span className="text-info">deduped</span>}
              {minSim > 0 && <span className="text-info">≥ {minSim.toFixed(2)}</span>}
            </div>
            <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2">
              {results.map((r, i) => (
                <button key={i}
                  onClick={() => {
                    if (r.object_id) router.push(`/object/${r.object_id}`);
                    else if (r.frame_id) router.push(`/frame/${r.frame_id}`);
                  }}
                  className="border border-line hover:border-accent text-left group"
                  title={`${r.class_name ?? r.scene?.weather ?? ""} · sim ${r.score.toFixed(3)}`}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={r.image_url || r.crop_url} alt="" className="w-full h-20 object-cover bg-bg-2" />
                  <div className="flex items-center justify-between px-1 py-0.5">
                    <span className="font-mono text-[9px] text-ink-3 truncate">{r.class_name ?? ""}</span>
                    <span className="font-mono text-[9px] text-accent">{r.score.toFixed(2)}</span>
                  </div>
                </button>
              ))}
            </div>
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
