"use client";

// Synthetic variants (#16). Hard frames (ones a person had to correct) re-rendered in rain, at dusk, at night,
// in fog or with motion blur. The perturbations are photometric only: nothing in the frame moves, so the source
// frame's labels are copied onto each variant exactly. Variants are stored as perturbed frames with their
// labels in the synthetic state, which keeps them out of everything that measures a model (gold, precision
// draws, settlement, the blind audit, embedding) and lets training take them only when a build asks for
// synthetic data. The copy-paste generator, which adds positives a starved class lacks, is on the same page.

import { useCallback, useEffect, useMemo, useState } from "react";

import { humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import SessionPicker from "@/components/lidar/SessionPicker";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Bar, Button, ErrLine, Field, HintBar, NumberInput, Panel, Segmented, Stat, TextInput } from "@/components/ui/kit";
import { ago, fromNs, humanize, num, shortId } from "@/lib/format";
import { useToolJob } from "@/lib/tools/toolJobs";
import {
  buildVariants, hardFrames, revertVariants, synthBuild, synthPlan, synthRuns, variantPerturbations,
  variantPreviewUrl, variantRuns, variantSummary, type HardFrame, type Perturbation, type SynthPlan, type SynthRun,
  type VariantRun,
} from "@/lib/tools/sensorsApi";

type Source = "errors" | "corrected" | "session";
const SOURCE_DESC: Record<Source, string> = {
  errors: "Frames with a label error a person confirmed in the error queue: the machine got them wrong.",
  corrected: "Frames where a person edited a machine label.",
  session: "Every frame of one session, in order, for a chosen stretch of road.",
};

export default function SyntheticPage() {
  const [perts, setPerts] = useState<Perturbation[]>([]);
  const [chosen, setChosen] = useState<Set<string>>(new Set(["rain", "night", "fog"]));
  const [strength, setStrength] = useState<Record<string, number>>({});
  const [source, setSource] = useState<Source>("errors");
  const [sessionId, setSessionId] = useState("");
  const [includeReview, setIncludeReview] = useState(false);
  const [frames, setFrames] = useState<HardFrame[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [previewFrame, setPreviewFrame] = useState<string | null>(null);
  const [runs, setRuns] = useState<VariantRun[]>([]);
  const [summary, setSummary] = useState<{ frames: number; by_perturbation: Record<string, number>; labels: number } | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadRuns = useCallback(() => {
    variantRuns().then((r) => setRuns(r.runs)).catch((e) => setErr(humanizeError(e)));
    variantSummary().then(setSummary).catch((e) => setErr(humanizeError(e)));
  }, []);

  useEffect(() => {
    variantPerturbations().then((p) => {
      setPerts(p.perturbations);
      setStrength(Object.fromEntries(p.perturbations.map((x) => [x.name, x.default_strength])));
    }).catch((e) => setErr(humanizeError(e)));
    loadRuns();
  }, [loadRuns]);

  const loadFrames = useCallback(async () => {
    if (source === "session" && !sessionId) { setFrames(null); return; }
    setLoading(true); setErr(null);
    try {
      const r = await hardFrames({ source, session_id: sessionId || undefined, limit: 120, include_review: includeReview });
      setFrames(r.frames);
      const ok = r.frames.filter((f) => f.eligible).slice(0, 12).map((f) => f.frame_id);
      setPicked(new Set(ok));
      setPreviewFrame(ok[0] ?? null);
    } catch (e) { setErr(humanizeError(e)); } finally { setLoading(false); }
  }, [source, sessionId, includeReview]);
  useEffect(() => { void loadFrames(); }, [loadFrames]);

  const { job } = useToolJob(jobId, { onDone: (j) => { loadRuns(); if (j.status === "error") setErr(j.error); } });
  const busy = job && !["done", "error", "canceled"].includes(job.status);

  const build = async () => {
    setErr(null);
    try {
      const names = perts.map((p) => p.name).filter((n) => chosen.has(n));
      const r = await buildVariants({ frame_ids: [...picked], perturbations: names, seed: 7, include_review: includeReview,
        strengths: Object.fromEntries(names.map((n) => [n, strength[n]])) });
      setJobId(r.job_id);
      loadRuns();
    } catch (e) { setErr(humanizeError(e)); }
  };

  const toggle = (set: Set<string>, v: string) => { const n = new Set(set); if (n.has(v)) n.delete(v); else n.add(v); return n; };
  const nVariants = picked.size * chosen.size;

  return (
    <PageShell active="Synthetic variants">
      <div className="page-wrap flex flex-col gap-4">
        <HintBar>Variants never reach gold sets, precision draws, settlement, the blind audit or embedding. Training takes them only when a dataset build turns on synthetic data.</HintBar>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Variant frames" value={num(summary?.frames)} sub="stored as perturbed" />
          <Stat label="Copied labels" value={num(summary?.labels)} sub="in the synthetic state" />
          <Stat label="Most made" value={summary && Object.keys(summary.by_perturbation).length ? humanize(Object.entries(summary.by_perturbation).sort((a, b) => b[1] - a[1])[0][0]) : "none"} />
          <Stat label="Runs" value={num(runs.length)} />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-4">
          <Panel title="1. Pick hard frames" desc={SOURCE_DESC[source]}
            right={<Segmented value={source} onChange={setSource} options={[
              { value: "errors", label: "Error queue" }, { value: "corrected", label: "Corrected" }, { value: "session", label: "A session" }]} />}>
            <div className="flex flex-wrap items-center gap-4 mb-3">
              {source === "session" && <div className="w-80"><SessionPicker value={sessionId} onChange={setSessionId} onLoad={setSessionId} /></div>}
              <label className="flex items-center gap-2 text-[12px] text-ink-2" data-tip="Copies of unreviewed labels would train as if they had been checked">
                <input type="checkbox" checked={includeReview} onChange={(e) => setIncludeReview(e.target.checked)} />
                Allow frames with labels still in review
              </label>
              {frames && <span className="text-[12px] text-ink-3">{picked.size} of {frames.filter((f) => f.eligible).length} usable frames picked</span>}
            </div>
            {loading && <p className="text-[12px] text-ink-3">Loading frames...</p>}
            {!loading && frames && !frames.length && (
              <EmptyState compact icon="wand" title="No frames here">{source === "errors" ? "No label error has been confirmed in the error queue yet. Try corrected frames or a session." : "Nothing matches; try another source."}</EmptyState>
            )}
            {!loading && frames && frames.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-4 2xl:grid-cols-6 gap-2 max-h-[560px] overflow-auto pr-1">
                {frames.map((f) => (
                  <div key={f.frame_id} className={`rounded border overflow-hidden text-[11px] ${picked.has(f.frame_id) ? "border-accent" : "border-line"} ${f.eligible ? "" : "opacity-60"}`}>
                    <button type="button" className="block w-full" onClick={() => setPreviewFrame(f.frame_id)} title="Preview the variants of this frame"
                      aria-label="Preview the variants of this frame">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={`/api/frames/${f.frame_id}/image`} alt="Source frame" loading="lazy" className="w-full aspect-video object-cover" />
                    </button>
                    <div className="px-1.5 py-1 flex items-center gap-1.5">
                      <input type="checkbox" disabled={!f.eligible} checked={picked.has(f.frame_id)} aria-label="Pick this frame"
                        onChange={() => setPicked((p) => toggle(p, f.frame_id))} />
                      <a href={`/frame/${f.frame_id}`} className="link truncate">{f.cam_id} · {fromNs(f.ts_ns)}</a>
                    </div>
                    {f.eligible ? <div className="px-1.5 pb-1 text-ink-3">{f.signal} {source === "errors" ? "confirmed errors" : "labels"}</div>
                      : <div className="px-1.5 pb-1 text-warn">{f.refused}</div>}
                  </div>
                ))}
              </div>
            )}
          </Panel>

          <div className="flex flex-col gap-4 min-w-0">
            <Panel title="2. Choose conditions" desc="Photometric only: occlusion is never offered, because it would hide what a copied label says is visible.">
              <div className="flex flex-col gap-1">
                {perts.map((p) => (
                  <div key={p.name} className="flex items-center gap-3 py-0.5">
                    <label className="flex items-center gap-2 w-28 text-[12px] text-ink">
                      <input type="checkbox" checked={chosen.has(p.name)} onChange={() => setChosen((c) => toggle(c, p.name))} />
                      {humanize(p.name)}
                    </label>
                    <input type="range" min={0.05} max={1} step={0.05} value={strength[p.name] ?? p.default_strength} aria-label={`${p.name} strength`}
                      onChange={(e) => setStrength((s) => ({ ...s, [p.name]: Number(e.target.value) }))} className="flex-1" />
                    <span className="w-9 text-right text-[12px] tabular-nums text-ink-2">{(strength[p.name] ?? p.default_strength).toFixed(2)}</span>
                  </div>
                ))}
              </div>
              {previewFrame && chosen.size > 0 && (
                <div className="grid grid-cols-2 gap-2 mt-3">
                  {perts.filter((p) => chosen.has(p.name)).map((p) => (
                    <figure key={p.name}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={variantPreviewUrl(previewFrame, p.name, strength[p.name] ?? p.default_strength)} alt={`${p.name} preview`} className="w-full rounded bg-bg-2 aspect-video object-cover" />
                      <figcaption className="text-[11px] text-ink-3 mt-0.5">{humanize(p.name)}</figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </Panel>
            <Panel title="3. Make the variants">
              <p className="text-[12px] text-ink-2 mb-2">{num(picked.size)} frames x {chosen.size} conditions = {num(nVariants)} variant frames, each with its source&apos;s labels.</p>
              <div className="flex items-center gap-2">
                <Button variant="primary" onClick={build} disabled={!nVariants || !!busy} tip="Needs the reviewer role">Make {num(nVariants)} variants</Button>
                {busy && <span className="flex-1 flex items-center gap-2 text-[12px]"><Bar frac={job!.progress} />{job!.stage ?? job!.status}</span>}
              </div>
              <ErrLine err={err} />
            </Panel>
          </div>
        </div>

        <Panel title="Runs" desc="Each run is one job; revert deletes exactly the frames it made and their copied labels.">
          {!runs.length ? <p className="text-[12px] text-ink-3">No variants made yet.</p> : (
            <table className="data-table w-full">
              <thead><tr><th>When</th><th>Conditions</th><th>Sources</th><th>Variants</th><th>Labels</th><th>Status</th><th /></tr></thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.job_id}>
                    <td>{ago(r.created_at)}</td>
                    <td>{(r.perturbations ?? []).map(humanize).join(", ")}</td>
                    <td className="tabular-nums">{r.sources}</td>
                    <td className="tabular-nums">{r.first_frame_id ? <a className="link" href={`/frame/${r.first_frame_id}`}>{num(r.frames)}</a> : num(r.frames)}</td>
                    <td className="tabular-nums">{num(r.objects)}</td>
                    <td>{r.reverted ? <Badge>reverted</Badge> : <Badge state={r.status === "done" ? "pass" : r.status === "error" ? "block" : "info"}>{r.status}</Badge>}</td>
                    <td className="text-right">
                      {r.status === "done" && !r.reverted && (
                        <Button variant="ghost" tip="Needs the reviewer role" onClick={async () => {
                          if (!window.confirm(`Delete the ${r.frames} variant frames of this run and their copied labels?`)) return;
                          try { await revertVariants(r.job_id); loadRuns(); } catch (e) { setErr(humanizeError(e)); }
                        }}>Revert</Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        <CopyPaste />
      </div>
    </PageShell>
  );
}

function CopyPaste() {
  const [classes, setClasses] = useState("cattle");
  const [n, setN] = useState(200);
  const [plan, setPlan] = useState<SynthPlan | null>(null);
  const [runs, setRuns] = useState<SynthRun[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const names = useMemo(() => classes.split(",").map((s) => s.trim()).filter(Boolean), [classes]);
  const load = useCallback(() => { synthRuns().then((r) => setRuns(r.runs)).catch((e) => setErr(humanizeError(e))); }, []);
  useEffect(() => { load(); }, [load]);

  return (
    <Panel title="Copy-paste composites" collapsible defaultOpen={false}
      desc="Real, human-masked instances of a starved class pasted onto real labelled road frames at a plausible place and size. Composites are quarantined the same way.">
      <div className="flex flex-wrap items-center gap-4">
        <Field label="Classes (comma separated)"><TextInput value={classes} onChange={setClasses} className="w-56" /></Field>
        <Field label="Frames"><NumberInput value={n} onChange={(v) => setN(Math.max(1, Math.min(5000, Math.round(v || 1))))} w="w-20" /></Field>
        <Button onClick={async () => { setErr(null); try { setPlan(await synthPlan(names, n)); } catch (e) { setErr(humanizeError(e)); } }}>Check donors</Button>
        <Button variant="primary" disabled={!plan?.feasible} tip="Needs the reviewer role" onClick={async () => {
          setErr(null);
          try { const r = await synthBuild(names, n); setNote(`Build ${shortId(r.run_id)} started.`); load(); } catch (e) { setErr(humanizeError(e)); }
        }}>Build</Button>
      </div>
      {plan && (
        <div className="mt-3 text-[12px]">
          <div className="text-ink-2">{num(plan.backgrounds)} usable backgrounds, {plan.batches} batch{plan.batches === 1 ? "" : "es"}{plan.reason ? `: ${plan.reason}` : ""}</div>
          <ul className="mt-1">
            {plan.classes.map((c) => <li key={c.class_name} className={c.donors ? "text-ink" : "text-warn"}>{humanize(c.class_name)}: {num(c.donors)} donors{c.reason ? ` (${c.reason})` : ""}</li>)}
          </ul>
        </div>
      )}
      {note && <p className="text-[12px] text-pass mt-2">{note}</p>}
      <ErrLine err={err} />
      {runs.length > 0 && (
        <table className="data-table w-full mt-3">
          <thead><tr><th>When</th><th>Classes</th><th>Frames</th><th>Status</th></tr></thead>
          <tbody>
            {runs.slice(0, 10).map((r) => (
              <tr key={r.run_id}><td>{ago(r.created_at)}</td><td>{(r.policy.class_names ?? []).map(humanize).join(", ")}</td>
                <td className="tabular-nums">{num(r.report.frames)} of {num(r.policy.n_frames)}</td><td><Badge state={r.status === "committed" ? "pass" : r.status === "error" ? "block" : "muted"}>{r.status}</Badge></td></tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}
