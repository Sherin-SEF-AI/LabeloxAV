"use client";

// Point labels (#5): give every LiDAR point a class. Select points with a brush, lasso, box or polygon (in the
// bird's-eye or orbit view), pick a class from the ontology and apply it. One-click helpers do the bulk:
// the ground plane, the DBSCAN cluster under a click, the classes of the camera's 2D masks copied onto the
// points they cover, the cloud's cuboids, and PTv3 (a 3D segmenter on the GPU) when it is installed. A save
// is a new human version; the projected and model labels are never overwritten.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";

import { api, humanizeError, type LidarCloud } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import Inspector from "@/components/shell/Inspector";
import SessionPicker from "@/components/lidar/SessionPicker";
import EmptyState from "@/components/ui/EmptyState";
import { Segmented } from "@/components/ui/kit";
import type { SegTool } from "@/components/lidar/SegmentViewer";
import { fromNs, humanize, num, shortId } from "@/lib/format";
import { useToolJob } from "@/lib/tools/toolJobs";
import {
  assistAi, assistCluster, assistGround, assistLabels, ptv3Status, saveSegLabels, segLabels, segPoints,
  segVersions, type SegPoints, type SegVersion,
} from "@/lib/tools/scene3dApi";
import {
  LabelHistory, applyClass, classRgb, combineMask, countByClass, countMask, mergeSuggestion, rgbCss, type Combine,
} from "@/lib/tools/scene3dGeom";

const SegmentViewer = dynamic(() => import("@/components/lidar/SegmentViewer"), { ssr: false });

const classLabel = (n: string | null | undefined) => (n ?? "").replace(/_/g, " ");
// Classes a LiDAR labeller reaches for first; the rest are a search away.
const COMMON = ["road", "sidewalk", "vegetation", "buildings", "pole", "vehicle_fallback", "sedan", "truck", "bus",
  "motorcycle", "pedestrian", "rider", "barrier", "fence", "traffic_sign", "tree", "curb", "parking_area"];

const TOOLS: { value: SegTool; label: string; key: string; tip: string }[] = [
  { value: "orbit", label: "Move", key: "O", tip: "pan and zoom the view (O)" },
  { value: "brush", label: "Brush", key: "B", tip: "drag to paint a selection ([ and ] change the size) (B)" },
  { value: "lasso", label: "Lasso", key: "L", tip: "drag a loop around points (L)" },
  { value: "box", label: "Box", key: "R", tip: "drag a rectangle (R)" },
  { value: "polygon", label: "Polygon", key: "P", tip: "click corners, double-click or Enter to close (P)" },
  { value: "pick", label: "Cluster", key: "K", tip: "click an object to select its DBSCAN cluster (K)" },
];

export default function PointLabelsPage() {
  const [sessionId, setSessionId] = useState("");
  const [clouds, setClouds] = useState<LidarCloud[]>([]);
  const [cloud, setCloud] = useState<LidarCloud | null>(null);
  const [pts, setPts] = useState<SegPoints | null>(null);
  const [labels, setLabels] = useState<Int16Array | null>(null);
  const [rev, setRev] = useState(0);
  const [baseSeg, setBaseSeg] = useState<string | null>(null);
  const [versions, setVersions] = useState<SegVersion[]>([]);
  const [selection, setSelection] = useState<Uint8Array | null>(null);
  const [tool, setTool] = useState<SegTool>("lasso");
  const [view, setView] = useState<"bev" | "perspective">("bev");
  const [brushPx, setBrushPx] = useState(18);
  const [classes, setClasses] = useState<OntologyClass[]>([]);
  const [cls, setCls] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [onlyUnlabelled, setOnlyUnlabelled] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ptv3, setPtv3] = useState<{ available: boolean; reason: string } | null>(null);
  const [aiJob, setAiJob] = useState<string | null>(null);
  const history = useRef(new LabelHistory(30));

  useEffect(() => {
    api.ontology().then((o) => {
      setClasses(o.classes);
      setCls((c) => c ?? o.classes.find((x) => x.name === "road")?.id ?? o.classes[0]?.id ?? null);
    }).catch(() => {});
    ptv3Status().then(setPtv3).catch(() => setPtv3(null));
  }, []);

  const byId = useMemo(() => new Map(classes.map((c) => [c.id, c])), [classes]);

  const loadClouds = useCallback(async (sid: string) => {
    if (!sid.trim()) return;
    setErr(null);
    try {
      const r = await api.lidarClouds(sid.trim());
      setClouds(r.clouds);
      if (!r.clouds.length) setErr("This session has no point clouds.");
    } catch (e) { setErr(humanizeError(e)); }
  }, []);

  const refreshVersions = useCallback(async (cid: string) => {
    try { setVersions((await segVersions(cid)).versions); } catch { setVersions([]); }
  }, []);

  const openCloud = useCallback(async (c: LidarCloud, segId?: string) => {
    setCloud(c); setBusy("Loading points"); setErr(null); setSelection(null); setNote(null);
    try {
      const p = await segPoints(c.cloud_id);
      const l = await segLabels(c.cloud_id, p.stride, segId);
      setPts(p);
      setLabels(l.labels);
      setBaseSeg(l.segId || null);
      setRev((r) => r + 1);
      history.current.clear();
      setDirty(false);
      if (l.segId) setNote(`Loaded ${l.source} labels, version ${l.version}.`);
      await refreshVersions(c.cloud_id);
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(null); }
  }, [refreshVersions]);

  const onSelect = useCallback((mask: Uint8Array, how: Combine) => {
    setSelection((cur) => combineMask(cur, mask, how));
  }, []);

  const edit = useCallback((fn: (l: Int16Array) => number, what: string) => {
    if (!labels) return;
    history.current.push(labels);
    const n = fn(labels);
    setRev((r) => r + 1);
    if (n) setDirty(true);
    setNote(`${what}: ${num(n)} points changed.`);
  }, [labels]);

  const applySel = useCallback((value: number) => {
    if (!selection || cls == null) return;
    edit((l) => applyClass(l, selection, value, onlyUnlabelled && value >= 0),
      value < 0 ? "Erased" : `Labelled ${classLabel(byId.get(value)?.name)}`);
    setSelection(null);
  }, [selection, cls, edit, onlyUnlabelled, byId]);

  const undo = useCallback(() => {
    const prev = history.current.undo();
    if (prev) { setLabels(prev); setRev((r) => r + 1); setDirty(true); setNote("Undone."); }
  }, []);

  const run = async (what: string, fn: () => Promise<void>) => {
    setBusy(what); setErr(null);
    try { await fn(); } catch (e) { setErr(humanizeError(e)); } finally { setBusy(null); }
  };

  const ground = () => cloud && pts && run("Fitting the ground", async () => {
    const r = await assistGround(cloud.cloud_id, pts.stride);
    setSelection(r.mask);
    setNote(`Ground: ${num(r.selected)} points selected (${r.info}). Pick road and press A to label them.`);
  });

  const pickCluster = useCallback((p: [number, number, number]) => {
    if (!cloud || !pts) return;
    void run("Clustering", async () => {
      const r = await assistCluster(cloud.cloud_id, pts.stride, p);
      setSelection((cur) => combineMask(cur, r.mask, "add"));
      let info = "";
      try {
        const j = JSON.parse(r.info);
        info = j.extent_m ? ` · ${j.extent_m.map((v: number) => v.toFixed(1)).join(" x ")} m` : j.reason ? ` · ${j.reason}` : "";
      } catch { /* plain text */ }
      setNote(`Cluster: ${num(r.selected)} points${info}. Shift is implied: clicks add clusters.`);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloud, pts]);

  const suggest = (kind: "from-2d" | "from-cuboids") => cloud && pts && labels && run("Copying labels", async () => {
    const r = await assistLabels(cloud.cloud_id, pts.stride, kind);
    edit((l) => mergeSuggestion(l, r.labels, true), kind === "from-2d" ? "Copied from 2D masks" : "Copied from cuboids");
    if (!r.selected) {
      let why = "";
      try { const j = JSON.parse(r.info); why = j.reason || (j.semantic_frames === 0 && j.objects === 0 ? "the camera frame has no masks yet" : ""); } catch { /* */ }
      setNote(`Nothing to copy${why ? `: ${why}` : ""}.`);
    }
  });

  const { job: aiState } = useToolJob(aiJob, {
    onDone: async (j) => {
      setAiJob(null);
      if (j.status !== "done" || !cloud || !pts || !labels) { setErr(j.error || `PTv3 ${j.status}`); return; }
      const v = (await segVersions(cloud.cloud_id)).versions.find((x) => x.source === "model");
      setVersions((await segVersions(cloud.cloud_id)).versions);
      if (!v) return;
      const s = await segLabels(cloud.cloud_id, pts.stride, v.seg_id);
      edit((l) => mergeSuggestion(l, s.labels, true), "PTv3 suggestions filled unlabelled points");
    },
  });

  const ai = () => cloud && run("Queueing PTv3", async () => {
    const j = await assistAi(cloud.cloud_id);
    setAiJob(j.job_id);
  });

  const save = useCallback(() => {
    if (!cloud || !pts || !labels) return;
    void run("Saving", async () => {
      const r = await saveSegLabels(cloud.cloud_id, pts.stride, labels, baseSeg);
      setBaseSeg(r.seg_id);
      setDirty(false);
      setNote(`Saved as human version ${r.version}: ${num(r.labelled)} of ${num(r.n_points)} points labelled.`);
      await refreshVersions(cloud.cloud_id);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloud, pts, labels, baseSeg, refreshVersions]);

  // keys
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); return; }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const tk = TOOLS.find((x) => x.key.toLowerCase() === e.key.toLowerCase());
      if (tk) { setTool(tk.value); return; }
      if (e.key === "a" || e.key === "A") { if (cls != null) applySel(cls); }
      else if (e.key === "Delete" || e.key === "Backspace") applySel(-1);
      else if (e.key === "Escape") setSelection(null);
      else if (e.key === "[") setBrushPx((b) => Math.max(4, b - 4));
      else if (e.key === "]") setBrushPx((b) => Math.min(120, b + 4));
      else if (e.key === "g" || e.key === "G") void ground();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [undo, save, applySel, cls, cloud, pts]);

  const stats = useMemo(() => {
    if (!labels) return [];
    return Array.from(countByClass(labels).entries()).sort((a, b) => b[1] - a[1]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labels, rev]);
  const labelled = labels ? labels.length - (stats.find(([c]) => c < 0)?.[1] ?? 0) : 0;

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase().replace(/ /g, "_");
    const common = COMMON.map((n) => classes.find((c) => c.name === n)).filter(Boolean) as OntologyClass[];
    if (!q) return common;
    return classes.filter((c) => c.name.includes(q) || (c.aliases || []).some((a) => a.toLowerCase().includes(q))).slice(0, 30);
  }, [query, classes]);
  const selCount = countMask(selection);

  return (
    <PageShell active="Point labels">
      <div className="flex h-full">
        <Inspector title="Point labels" side="left" width="w-80">
          <div className="flex flex-col gap-3 p-4 text-[12px]">
            <SessionPicker value={sessionId} onChange={setSessionId} onLoad={loadClouds} />
            {clouds.length > 0 && (
              <div className="flex max-h-36 flex-col gap-1 overflow-y-auto" role="list" aria-label="clouds">
                {clouds.map((c) => (
                  <button key={c.cloud_id} type="button" onClick={() => openCloud(c)} aria-pressed={cloud?.cloud_id === c.cloud_id}
                    className={`rounded border px-2.5 py-1.5 text-left ${cloud?.cloud_id === c.cloud_id ? "border-accent/60 bg-accent/15" : "border-line bg-bg-2 hover:border-line-3"}`}>
                    <div className="flex justify-between"><span className="font-medium">{c.source === "lidar" ? "LiDAR" : humanize(c.source)}</span>
                      <span className="tabular-nums text-ink-3">{num(c.point_count)} points</span></div>
                    <div className="truncate text-[11px] text-ink-3">{fromNs(c.ts_ns)} · <span className="font-mono">{shortId(c.cloud_id)}</span></div>
                  </button>
                ))}
              </div>
            )}

            {pts && (
              <>
                <div>
                  <div className="label-caps mb-1">Class <span className="normal-case text-ink-3">(A applies, Delete erases)</span></div>
                  <input className="field mb-1.5 w-full" placeholder="Search the ontology" value={query}
                    onChange={(e) => setQuery(e.target.value)} aria-label="search classes" />
                  <div className="flex max-h-44 flex-wrap gap-1 overflow-y-auto">
                    {shown.map((c) => (
                      <button key={c.id} type="button" onClick={() => setCls(c.id)} aria-pressed={cls === c.id}
                        className={`chip ${cls === c.id ? "chip-on" : ""}`}>
                        <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: rgbCss(classRgb(c.id)) }} />
                        {classLabel(c.name)}
                      </button>
                    ))}
                  </div>
                  <label className="mt-2 flex items-center gap-2 text-ink-2">
                    <input type="checkbox" checked={onlyUnlabelled} onChange={(e) => setOnlyUnlabelled(e.target.checked)} />
                    Only label unlabelled points
                  </label>
                  <div className="mt-2 flex gap-2">
                    <button type="button" className="btn btn-primary flex-1" disabled={!selCount || cls == null}
                      onClick={() => cls != null && applySel(cls)}>Apply to {num(selCount)} points</button>
                    <button type="button" className="btn" disabled={!selCount} onClick={() => applySel(-1)}>Erase</button>
                  </div>
                </div>

                <div>
                  <div className="label-caps mb-1">Helpers</div>
                  <div className="grid grid-cols-2 gap-1.5">
                    <button type="button" className="btn" onClick={ground} disabled={!!busy} data-tip="select the fitted ground plane (G)">Ground plane</button>
                    <button type="button" className="btn" onClick={() => setTool("pick")} disabled={!!busy} data-tip="click an object in the view to select its cluster (K)">Cluster pick</button>
                    <button type="button" className="btn" onClick={() => suggest("from-2d")} disabled={!!busy} data-tip="copy the classes of the camera frame's 2D masks onto the points they cover; only unlabelled points change">From 2D masks</button>
                    <button type="button" className="btn" onClick={() => suggest("from-cuboids")} disabled={!!busy} data-tip="points inside the cloud's cuboids take their class, ground points become road; only unlabelled points change">From cuboids</button>
                  </div>
                  <button type="button" className="btn mt-1.5 w-full" onClick={ai} disabled={!!busy || !!aiJob || !ptv3?.available}
                    data-tip={ptv3?.available ? "run PTv3 on the GPU and fill unlabelled points with its classes" : ptv3?.reason}>
                    {aiJob ? `PTv3 ${aiState?.status ?? "queued"}${aiState?.quiet_hours ? " (waits for office hours to end)" : ""}` : "AI suggestions (PTv3)"}
                  </button>
                  {ptv3 && !ptv3.available && <div className="mt-1 text-[11px] text-ink-3">PTv3 is not available here: {ptv3.reason}</div>}
                </div>

                <div>
                  <div className="label-caps mb-1">Labels <span className="normal-case text-ink-3">· {num(labelled)} of {num(pts.count)} points</span></div>
                  <div className="flex max-h-32 flex-col gap-0.5 overflow-y-auto">
                    {stats.filter(([c]) => c >= 0).map(([c, n]) => (
                      <div key={c} className="flex items-center justify-between">
                        <span><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: rgbCss(classRgb(c)) }} />
                          {classLabel(byId.get(c)?.name ?? String(c))}</span>
                        <span className="tabular-nums text-ink-3">{num(n)}</span>
                      </div>
                    ))}
                    {!labelled && <div className="text-ink-3">Nothing labelled yet. Start with Ground plane.</div>}
                  </div>
                  <div className="mt-2 flex gap-2">
                    <button type="button" className="btn btn-primary flex-1" onClick={save} disabled={!dirty || !!busy}>
                      {dirty ? "Save new version" : "Saved"}
                    </button>
                    <button type="button" className="btn" onClick={undo} disabled={!history.current.size} data-tip="Ctrl+Z">Undo</button>
                  </div>
                </div>

                {versions.length > 0 && (
                  <div>
                    <div className="label-caps mb-1">Versions</div>
                    <div className="flex max-h-28 flex-col gap-0.5 overflow-y-auto">
                      {versions.map((v) => (
                        <button key={v.seg_id} type="button" className={`flex justify-between rounded px-1.5 py-0.5 text-left hover:bg-bg-2 ${baseSeg === v.seg_id ? "bg-bg-2" : ""}`}
                          onClick={() => cloud && openCloud(cloud, v.seg_id)} data-tip="open this version (unsaved edits are dropped)">
                          <span>{v.source === "human" ? "Human" : v.source === "model" ? "PTv3" : "Projected"} v{v.version}</span>
                          <span className="text-ink-3">{v.created_at ? new Date(v.created_at).toLocaleString() : ""}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
            {note && <div role="status" className="text-ink-2">{note}</div>}
            {err && <div role="alert" className="rounded border border-block/40 bg-block/10 p-2 text-block">{err}</div>}
          </div>
        </Inspector>

        <main className="relative flex flex-1 flex-col">
          <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2 text-[12px]">
            <div className="flex gap-1" role="group" aria-label="selection tool">
              {TOOLS.map((t) => (
                <button key={t.value} type="button" data-tip={t.tip} aria-pressed={tool === t.value}
                  className={`btn ${tool === t.value ? "btn-on" : ""}`} onClick={() => setTool(t.value)}>
                  {t.label} <span className="kbd ml-1">{t.key}</span>
                </button>
              ))}
            </div>
            <Segmented options={[{ value: "bev", label: "Bird's-eye" }, { value: "perspective", label: "Orbit" }]}
              value={view} onChange={(v) => setView(v as "bev" | "perspective")} />
            {tool === "brush" && <span className="text-ink-3">Brush {brushPx} px</span>}
            <span className="ml-auto text-ink-3">
              {selCount ? `${num(selCount)} selected · Shift adds, Alt subtracts, Esc clears` : "Shift adds, Alt subtracts"}
              {pts && pts.stride > 1 ? ` · every ${pts.stride}th of ${num(pts.total)} points shown` : ""}
            </span>
          </div>
          <div className="relative flex-1">
            {pts && labels ? (
              <SegmentViewer points={pts.points} count={pts.count} labels={labels} labelsRev={rev} selection={selection}
                view={view} tool={tool} brushPx={brushPx} onSelect={onSelect} onPick={pickCluster} />
            ) : (
              <div className="flex h-full items-center justify-center">
                {busy ? <span role="status" className="text-ink-3">{busy}...</span> : (
                  <EmptyState icon="cube" title={clouds.length ? "Pick a cloud" : "No cloud loaded"}>
                    {clouds.length ? "Choose a cloud on the left to label its points."
                      : "Choose a session with point clouds (a KITTI drive, or a session lifted to pseudo-LiDAR)."}
                  </EmptyState>
                )}
              </div>
            )}
            {busy && pts && <div role="status" className="absolute right-3 top-3 rounded bg-bg-2 px-2 py-1 text-[12px] text-ink-2">{busy}...</div>}
          </div>
        </main>
      </div>
    </PageShell>
  );
}
