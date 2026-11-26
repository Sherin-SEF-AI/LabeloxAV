"use client";

// Rare scenes (tool #14): the least-covered corners of the corpus, found from the frame embeddings. A frame is
// rare when nothing from any OTHER drive looks like it (cross-session k-nearest-neighbour distance over the DINOv3
// or SigLIP 2 vectors); each tile says why in measured terms. Search by text or by an example frame and sort the
// matches rarest first. Select frames and turn them into a labelling task in one step.
//
// Server: services/intelligence/rare_scenes.py through services/api/routers/tools_quality.py.

import Link from "next/link";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { Badge, Button, Panel, Segmented, Stat } from "@/components/ui/kit";
import { humanizeError } from "@/lib/api";
import { ago, num, pct, shortId } from "@/lib/format";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";
import { rarestShare, sceneLine, toggleId } from "@/lib/tools/quality";
import {
  createRareTask, getRareLatest, listProjects, rankRareScenes, searchRareScenes,
  type RareFrame, type RareLatest, type RareSearch,
} from "@/lib/tools/qualityApi";
import { useToolJob } from "@/lib/tools/toolJobs";

type Mode = "least" | "text" | "example";
const rank = (role?: string) => (role === "admin" ? 3 : role === "reviewer" ? 2 : role === "annotator" ? 1 : 0);

export default function RareScenesPage() {
  return (
    <Suspense fallback={<PageShell active="Rare scenes" title="Rare scenes"><div className="page-wrap text-[13px] text-ink-3" role="status">Loading...</div></PageShell>}>
      <RareScenes />
    </Suspense>
  );
}

function RareScenes() {
  const me = useCurrentUser();
  const canWrite = rank(me?.role) >= 2;
  const params = useSearchParams();
  const router = useRouter();
  const exampleParam = params.get("frame");
  const [mode, setMode] = useState<Mode>(exampleParam ? "example" : (params.get("q") ? "text" : "least"));
  const [sel, setSel] = useState<Set<string>>(new Set());

  return (
    <PageShell active="Rare scenes" title="Rare scenes"
      subtitle="Frames that nothing from another drive resembles: where the corpus is thinnest, and so where a label teaches most">
      <div className="page-wrap space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <Segmented<Mode> value={mode} onChange={(m) => { setMode(m); if (m !== "example") router.replace("/rare-scenes"); }}
            options={[
              { value: "least", label: "Least covered", tip: "The corpus ranked by how little else looks like each frame" },
              { value: "text", label: "Search by text", tip: "Describe a scene; matches are scored for rarity" },
              { value: "example", label: "Search by example", tip: "Frames like a given one, scored for rarity" },
            ]} />
          <span className="text-[12px] text-ink-3">Only real camera frames are ranked; synthetic and perturbed frames are left out.</span>
        </div>
        {mode === "least" && <LeastCovered canWrite={canWrite} sel={sel} setSel={setSel} />}
        {mode === "text" && <TextSearch initial={params.get("q") ?? ""} sel={sel} setSel={setSel} />}
        {mode === "example" && <ExampleSearch initial={exampleParam ?? ""} sel={sel} setSel={setSel} />}
        <SelectionBar sel={sel} clear={() => setSel(new Set())} canWrite={canWrite} />
      </div>
    </PageShell>
  );
}

type SelProps = { sel: Set<string>; setSel: (s: Set<string>) => void };

function LeastCovered({ canWrite, sel, setSel }: { canWrite: boolean } & SelProps) {
  const [space, setSpace] = useState<"dino" | "siglip">("dino");
  const [data, setData] = useState<RareLatest | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [reload, setReload] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    setErr(null);
    getRareLatest(space).then((d) => {
      if (stop) return;
      setData(d);
      if (d.running && !jobId) setJobId(d.running.job_id);
    }).catch((e) => { if (!stop) setErr(e); });
    return () => { stop = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [space, reload]);

  const { job } = useToolJob(jobId, {
    onDone: (j) => {
      if (j.status !== "done") toast(j.error ?? `Ranking ${j.status}`, "error");
      setJobId(null);
      setReload((n) => n + 1);
    },
  });
  const start = async () => {
    try { setJobId((await rankRareScenes({ space })).job_id); } catch (e) { toast(humanizeError(e), "error"); }
  };

  const r = data?.result ?? null;
  return (
    <Panel title="Least covered frames"
      desc="Score: mean cosine distance to the 10 nearest frames from other drives. At most 4 frames per drive and 6 per visual cluster, so one odd drive cannot fill the page."
      right={
        <div className="flex items-center gap-2">
          <Segmented<"dino" | "siglip"> value={space} onChange={setSpace} options={[
            { value: "dino", label: "Looks (DINOv3)", tip: "Visual similarity" },
            { value: "siglip", label: "Meaning (SigLIP 2)", tip: "Semantic similarity, the space text search uses" },
          ]} />
          {canWrite && (
            <Button variant="primary" onClick={start} disabled={!!jobId}
              tip="Rank every embedded real frame again. A minute or two of CPU.">
              {jobId ? `Ranking ${pct(job?.progress ?? 0)}${job?.stage ? ` · ${job.stage}` : ""}` : r ? "Rank again" : "Rank the corpus"}
            </Button>
          )}
        </div>
      }>
      <LoadState loading={data === null && !err} error={err} onRetry={() => setReload((n) => n + 1)}
        empty={data !== null && !r} emptyTitle="Not ranked yet"
        emptyText={canWrite ? "Rank the corpus to find its least-covered scenes." : "A reviewer has not ranked the corpus yet."}>
        {r && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              <Stat label="Frames ranked" value={num(r.n_frames)} sub={`${num(r.n_sessions)} drives`} />
              <Stat label="Typical nearest match" value={r.median_nn_sim != null ? r.median_nn_sim.toFixed(2) : "-"}
                sub="median similarity to another drive" />
              <Stat label="Visual clusters" value={num(r.n_clusters ?? 0)} />
              <Stat label="Left out" value={num(r.excluded_not_real)} sub="synthetic or perturbed" />
              <Stat label="Not embedded" value={num(r.not_embedded)} sub="real frames without a vector" />
            </div>
            <div className="text-[12px] text-ink-3">
              Ranked {data?.job?.finished_at ? ago(data.job.finished_at) : ""} in the {r.space === "dino" ? "DINOv3" : "SigLIP 2"} space.
              {r.detail ? ` ${r.detail}` : ""}
            </div>
            <FrameGrid frames={r.frames} sel={sel} setSel={setSel} />
            {!!r.suspect_frames?.length && (
              <details className="panel">
                <summary className="panel-head cursor-pointer">
                  <span className="section-title">Set aside: {num(r.n_suspect ?? r.suspect_frames.length)} frames that are not a scene</span>
                  <span className="section-desc ml-2">rare only because they are blank, corrupt or a fill; worth a look as a data fault</span>
                </summary>
                <div className="p-3 grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))" }}>
                  {r.suspect_frames.map((f) => (
                    <Link key={f.frame_id} href={`/frame/${f.frame_id}`} className="block group" title={f.reason}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={f.image_url} alt="" loading="lazy" decoding="async"
                        className="w-full aspect-video object-cover rounded border hairline group-hover:border-accent" />
                      <span className="block text-[11.5px] text-ink-3 leading-snug mt-0.5">{f.reason}</span>
                    </Link>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
      </LoadState>
    </Panel>
  );
}

function TextSearch({ initial, sel, setSel }: { initial: string } & SelProps) {
  const [q, setQ] = useState(initial);
  const [rarest, setRarest] = useState(true);
  const [res, setRes] = useState<RareSearch | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const run = useCallback(async (text: string) => {
    if (!text.trim()) return;
    setBusy(true); setErr(null);
    try { setRes(await searchRareScenes({ text, rarest_first: rarest, k: 48 })); }
    catch (e) { setErr(e); } finally { setBusy(false); }
  }, [rarest]);
  useEffect(() => { if (initial) void run(initial); }, [initial, run]);
  return (
    <Panel title="Search by text" desc="SigLIP 2 matches the words to whole frames; each match is then scored for how rare it is across drives.">
      <form className="flex flex-wrap items-center gap-2 mb-3" onSubmit={(e) => { e.preventDefault(); void run(q); }}>
        <input className="field flex-1 min-w-[240px]" value={q} onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. cattle on a flyover at night, a tractor in heavy rain" aria-label="Describe a scene" />
        <label className="flex items-center gap-1.5 text-[12px] text-ink-2">
          <input type="checkbox" checked={rarest} onChange={(e) => setRarest(e.target.checked)} /> Rarest first
        </label>
        <Button variant="primary" disabled={busy || !q.trim()}>{busy ? "Searching..." : "Search"}</Button>
      </form>
      <LoadState loading={busy && !res} error={err} empty={res !== null && res.results.length === 0}
        emptyTitle="No matching frames" emptyText={res?.detail ?? "Try other words."}>
        {res && <FrameGrid frames={res.results} sel={sel} setSel={setSel} showMatch />}
      </LoadState>
    </Panel>
  );
}

function ExampleSearch({ initial, sel, setSel }: { initial: string } & SelProps) {
  const router = useRouter();
  const [fid, setFid] = useState(initial);
  const [space, setSpace] = useState<"dino" | "siglip">("dino");
  const [res, setRes] = useState<RareSearch | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { setFid(initial); }, [initial]);
  useEffect(() => {
    if (!initial) return;
    let stop = false;
    setBusy(true); setErr(null);
    searchRareScenes({ frame_id: initial, space, k: 48, rarest_first: true })
      .then((r) => { if (!stop) setRes(r); }).catch((e) => { if (!stop) setErr(e); })
      .finally(() => { if (!stop) setBusy(false); });
    return () => { stop = true; };
  }, [initial, space]);
  return (
    <Panel title="Search by example" desc="Frames that look like an example, rarest first. Use 'More like this' on any tile, or paste a frame id."
      right={<Segmented<"dino" | "siglip"> value={space} onChange={setSpace} options={[
        { value: "dino", label: "Looks" }, { value: "siglip", label: "Meaning" }]} />}>
      <form className="flex flex-wrap items-center gap-2 mb-3"
        onSubmit={(e) => { e.preventDefault(); if (fid.trim()) router.push(`/rare-scenes?frame=${fid.trim()}`); }}>
        <input className="field w-[360px] font-mono" value={fid} onChange={(e) => setFid(e.target.value)}
          placeholder="frame id" aria-label="Example frame id" />
        <Button variant="primary" disabled={!fid.trim()}>Find</Button>
      </form>
      {initial && (
        <div className="flex items-start gap-3 mb-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`/api/frames/${initial}/image`} alt="The example frame" className="w-[200px] rounded border hairline" />
          <div className="text-[12px] text-ink-3">Example <Link href={`/frame/${initial}`} className="link font-mono">{shortId(initial)}</Link></div>
        </div>
      )}
      <LoadState loading={busy} error={err} empty={!!res && res.results.length === 0} emptyTitle="No similar frames"
        emptyText={res?.detail ?? "Nothing else in the corpus is close to this frame."}>
        {res && <FrameGrid frames={res.results} sel={sel} setSel={setSel} showMatch />}
      </LoadState>
    </Panel>
  );
}

function FrameGrid({ frames, sel, setSel, showMatch = false }: { frames: RareFrame[]; showMatch?: boolean } & SelProps) {
  const router = useRouter();
  const allOn = frames.length > 0 && frames.every((f) => sel.has(f.frame_id));
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[12px]">
        <button type="button" className="chip" onClick={() => {
          const next = new Set(sel);
          for (const f of frames) { if (allOn) next.delete(f.frame_id); else next.add(f.frame_id); }
          setSel(next);
        }}>{allOn ? "Clear these" : `Select all ${frames.length}`}</button>
      </div>
      <div className="grid gap-2" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))" }}>
        {frames.map((f) => {
          const on = sel.has(f.frame_id);
          return (
            <article key={f.frame_id} className={`panel overflow-hidden ${on ? "ring-2 ring-accent" : ""}`}>
              <button type="button" className="relative block w-full aspect-video bg-bg-2" onClick={() => setSel(toggleId(sel, f.frame_id))}
                aria-pressed={on} aria-label={`${on ? "Deselect" : "Select"} frame ${shortId(f.frame_id)}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={f.image_url} alt="" loading="lazy" decoding="async" className="w-full h-full object-cover" />
                <span className="absolute top-1 left-1 flex gap-1">
                  {f.score != null && <span className="rounded bg-bg/85 px-1.5 text-[11px] text-ink tabular-nums" title="Cross-drive rarity">rarity {f.score.toFixed(2)}</span>}
                  {showMatch && f.match != null && <span className="rounded bg-bg/85 px-1.5 text-[11px] text-ink-2 tabular-nums">match {f.match.toFixed(2)}</span>}
                </span>
                {on && <span className="absolute top-1 right-1 rounded-full bg-accent text-bg text-[11px] font-semibold w-5 h-5 leading-5 text-center" aria-hidden>✓</span>}
              </button>
              <div className="p-2 space-y-1">
                <div className="flex flex-wrap items-center gap-1.5 text-[11.5px] text-ink-3">
                  <span>{f.vehicle_id ?? "?"}{f.city ? ` · ${f.city}` : ""}</span>
                  {sceneLine(f.scene) && <Badge>{sceneLine(f.scene)}</Badge>}
                  {f.percentile != null && <span className="tabular-nums">{rarestShare(f.percentile)}</span>}
                </div>
                <ul className="text-[12px] text-ink-2 leading-snug space-y-0.5" title={f.why.join("\n")}>
                  {f.why.slice(0, 3).map((w, i) => <li key={i}>{w}</li>)}
                </ul>
                <div className="flex items-center gap-3 text-[12px] pt-0.5">
                  <Link href={`/frame/${f.frame_id}`} className="link">Open</Link>
                  <button type="button" className="link" onClick={() => router.push(`/rare-scenes?frame=${f.frame_id}`)}>More like this</button>
                  {f.nn_frame_id && <Link href={`/frame/${f.nn_frame_id}`} className="link text-ink-3"
                    title="The closest frame from another drive">closest elsewhere</Link>}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function SelectionBar({ sel, clear, canWrite }: { sel: Set<string>; clear: () => void; canWrite: boolean }) {
  const [projects, setProjects] = useState<{ project_id: string; name: string }[] | null>(null);
  const [pid, setPid] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const n = sel.size;
  useEffect(() => {
    if (!n || !canWrite || projects) return;
    listProjects().then((r) => { setProjects(r.projects); setPid((p) => p || r.projects[0]?.project_id || ""); })
      .catch((e) => toast(humanizeError(e), "error"));
  }, [n, canWrite, projects]);
  const defaultName = useMemo(() => `Rare scenes ${new Date().toISOString().slice(0, 10)}`, []);
  if (!n) return null;
  const make = async () => {
    setBusy(true);
    try {
      const r = await createRareTask({ project_id: pid, name: (name || defaultName).trim(), frame_ids: [...sel] });
      toast(`Task "${r.name}": ${r.n_frames} frames in ${r.n_jobs} job${r.n_jobs === 1 ? "" : "s"}`, "success");
      clear();
    } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };
  return (
    <div className="sticky bottom-2 z-10 panel px-3 py-2 flex flex-wrap items-center gap-2 shadow-lg">
      <span className="text-[13px] text-ink">{n} frame{n === 1 ? "" : "s"} selected</span>
      <Button variant="ghost" onClick={clear}>Clear</Button>
      {canWrite ? (
        <>
          <select className="field" value={pid} onChange={(e) => setPid(e.target.value)} aria-label="Project">
            {(projects ?? []).map((p) => <option key={p.project_id} value={p.project_id}>{p.name}</option>)}
          </select>
          <input className="field w-56" value={name} placeholder={defaultName} onChange={(e) => setName(e.target.value)}
            aria-label="Task name" />
          <Button variant="primary" onClick={make} disabled={busy || !pid}>{busy ? "Creating..." : "Create labelling task"}</Button>
          <Link href="/projects" className="link text-[12px]">Projects</Link>
        </>
      ) : <span className="text-[12px] text-ink-3">A reviewer can turn a selection into a labelling task.</span>}
    </div>
  );
}
