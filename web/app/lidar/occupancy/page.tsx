"use client";

// Occupancy (#6): look at and correct the voxel grids that say which space around the vehicle is filled.
// Grids are generated from LiDAR, or for a session without LiDAR from camera depth (Depth-Anything-V2 metric
// depth lifted to pseudo-LiDAR). A person paints voxels occupied or free, or fills/clears a box, in the
// bird's-eye view over a chosen height band; the edits are a separate human layer, so the generated grid
// stays as it was built. Edits can be carried to the following grids through the ego poses.

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";

import { humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import Inspector from "@/components/shell/Inspector";
import SessionPicker from "@/components/lidar/SessionPicker";
import EmptyState from "@/components/ui/EmptyState";
import { Segmented } from "@/components/ui/kit";
import type { VoxTool, VoxValue } from "@/components/lidar/VoxelEditor";
import { fromNs, humanize, num } from "@/lib/format";
import { useToolJob } from "@/lib/tools/toolJobs";
import {
  occBuild, occEdit, occGrid, occGrids, occPropagate, sceneOverview, type OccGrid, type OccGridRow, type OccOp,
  type Scene3dOverview,
} from "@/lib/tools/scene3dApi";

const VoxelEditor = dynamic(() => import("@/components/lidar/VoxelEditor"), { ssr: false });

export default function OccupancyPage() {
  const [sessionId, setSessionId] = useState("");
  const [ov, setOv] = useState<Scene3dOverview | null>(null);
  const [rows, setRows] = useState<OccGridRow[]>([]);
  const [idx, setIdx] = useState(0);
  const [grid, setGrid] = useState<OccGrid | null>(null);
  const [tool, setTool] = useState<VoxTool>("paint");
  const [value, setValue] = useState<VoxValue>("occupied");
  const [view, setView] = useState<"bev" | "perspective">("bev");
  const [zLo, setZLo] = useState(0);
  const [zHi, setZHi] = useState(11);
  const [count, setCount] = useState(10);
  const [lastOps, setLastOps] = useState<OccOp[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const loadGrid = useCallback(async (r: OccGridRow | undefined) => {
    if (!r) { setGrid(null); return; }
    try { setGrid(await occGrid(r.grid_id)); } catch (e) { setErr(humanizeError(e)); }
  }, []);

  const loadSession = useCallback(async (sid: string, keepIdx = false) => {
    if (!sid.trim()) return;
    setErr(null);
    try {
      const [o, g] = await Promise.all([sceneOverview(sid.trim()), occGrids(sid.trim())]);
      setOv(o);
      setRows(g.grids);
      const i = keepIdx ? Math.min(idx, Math.max(0, g.grids.length - 1)) : 0;
      setIdx(i);
      if (g.grids.length) {
        const first = await occGrid(g.grids[i].grid_id);
        setGrid(first);
        if (!keepIdx) { setZLo(0); setZHi(first.dims[2] - 1); }
      } else setGrid(null);
    } catch (e) { setErr(humanizeError(e)); }
  }, [idx]);

  const { job } = useToolJob(jobId, {
    onDone: (j) => {
      setJobId(null);
      if (j.status === "done") { setNote("Grids built."); void loadSession(sessionId); }
      else setErr(j.error || `The build ${j.status}.`);
    },
  });

  const build = async (mode: "lidar" | "camera") => {
    setErr(null);
    try {
      const j = await occBuild(sessionId, { mode, every: 5, limit: mode === "camera" ? 40 : 500 });
      setJobId(j.job_id);
      setNote(mode === "camera" ? "Lifting camera frames with metric depth, then building grids." : "Building grids from the clouds.");
    } catch (e) { setErr(humanizeError(e)); }
  };

  const step = useCallback((d: number) => {
    setIdx((i) => {
      const n = Math.max(0, Math.min(rows.length - 1, i + d));
      if (n !== i) void loadGrid(rows[n]);
      return n;
    });
  }, [rows, loadGrid]);

  const onOps = useCallback(async (ops: OccOp[]) => {
    if (!grid) return;
    setBusy("Saving edit"); setErr(null);
    try {
      const r = await occEdit(grid.grid_id, ops);
      setLastOps(ops);
      setNote(`${num(r.touched)} voxels set ${ops[0].value}: ${num(r.occupied_before)} generated, ${num(r.occupied_after)} after edits.`);
      setGrid(await occGrid(grid.grid_id));
      setRows((rs) => rs.map((x) => (x.grid_id === grid.grid_id ? { ...x, edited: true, occupied_after: r.occupied_after } : x)));
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(null); }
  }, [grid]);

  const propagate = async (onlyLast: boolean) => {
    if (!grid) return;
    setBusy("Carrying edits"); setErr(null);
    try {
      const r = await occPropagate(grid.grid_id, count, onlyLast && lastOps ? lastOps : undefined);
      setNote(r.reason ? r.reason : `Carried to ${r.propagated_to} grids${r.skipped_outside ? `, ${r.skipped_outside} had the edit outside their window` : ""}${r.skipped_unplaced ? `, ${r.skipped_unplaced} were not placed by a pose` : ""}.`);
      await loadSession(sessionId, true);
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(null); }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT")) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "o") setTool("orbit");
      else if (k === "p") setTool("paint");
      else if (k === "b") setTool("box");
      else if (k === "1") setValue("occupied");
      else if (k === "2") setValue("free");
      else if (k === "3") setValue("reset");
      else if (e.key === "ArrowRight") step(1);
      else if (e.key === "ArrowLeft") step(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  const row = rows[idx];
  const zm = (i: number) => (grid ? (grid.origin[2] + i * grid.voxel_m).toFixed(1) : "");

  return (
    <PageShell active="Occupancy">
      <div className="flex h-full">
        <Inspector title="Occupancy" side="left" width="w-80">
          <div className="flex flex-col gap-3 p-4 text-[12px]">
            <SessionPicker value={sessionId} onChange={setSessionId} onLoad={(s) => loadSession(s)} />
            {ov && (
              <div className="text-ink-2">
                {ov.lidar ? `${num(ov.clouds.lidar ?? 0)} LiDAR scans` : "No LiDAR"} · {num(ov.poses)} ego poses ·{" "}
                {num(Object.values(ov.cameras).reduce((a, b) => a + b, 0))} camera frames
              </div>
            )}
            {ov && (
              <div>
                <div className="label-caps mb-1">Generate</div>
                <div className="grid grid-cols-2 gap-1.5">
                  <button type="button" className="btn" disabled={!!jobId || !Object.keys(ov.clouds).length} onClick={() => build("lidar")}
                    data-tip="voxelise the session's point clouds (reviewer)">From LiDAR</button>
                  <button type="button" className="btn" disabled={!!jobId || !Object.keys(ov.cameras).length} onClick={() => build("camera")}
                    data-tip="metric depth from every 5th camera instant, lifted to points, then voxelised (GPU, reviewer; waits outside office hours)">From cameras</button>
                </div>
                {job && <div className="mt-1 text-ink-3" role="status">{humanize(job.status)} {job.stage ? `· ${job.stage}` : ""} {Math.round(job.progress * 100)}%
                  {job.quiet_hours ? " · waits for office hours to end" : ""}</div>}
              </div>
            )}
            {rows.length > 0 && (
              <div>
                <div className="label-caps mb-1">Grid {idx + 1} of {rows.length} <span className="normal-case text-ink-3">(arrow keys step)</span></div>
                <input type="range" min={0} max={rows.length - 1} value={idx} className="w-full" aria-label="grid"
                  onChange={(e) => { const i = Number(e.target.value); setIdx(i); void loadGrid(rows[i]); }} />
                {row && (
                  <div className="text-ink-2">
                    {fromNs(row.ts_ns)} · {humanize(row.source)} · {num(row.occupied)} voxels
                    {row.edited ? ` · edited (${num(row.occupied_after)} after edits)` : ""}
                    {!row.placed && <div className="text-warn">Not placed by an ego pose: edits cannot be carried forward.</div>}
                  </div>
                )}
              </div>
            )}
            {grid && (
              <>
                <div>
                  <div className="label-caps mb-1">Edit</div>
                  <Segmented options={[{ value: "orbit", label: "Move" }, { value: "paint", label: "Paint" }, { value: "box", label: "Box" }]}
                    value={tool} onChange={(v) => setTool(v as VoxTool)} />
                  <div className="mt-1.5">
                    <Segmented options={[{ value: "occupied", label: "Occupied", tip: "1" }, { value: "free", label: "Free", tip: "2" },
                      { value: "reset", label: "Reset", tip: "3: back to what was generated" }]}
                      value={value} onChange={(v) => setValue(v as VoxValue)} />
                  </div>
                  <div className="mt-2 text-ink-2">Height band {zm(zLo)} to {zm(zHi + 1)} m</div>
                  <label className="flex items-center gap-2"><span className="w-10">From</span>
                    <input type="range" min={0} max={grid.dims[2] - 1} value={zLo} className="flex-1" aria-label="lowest layer"
                      onChange={(e) => { const v = Number(e.target.value); setZLo(v); if (v > zHi) setZHi(v); }} /></label>
                  <label className="flex items-center gap-2"><span className="w-10">To</span>
                    <input type="range" min={0} max={grid.dims[2] - 1} value={zHi} className="flex-1" aria-label="highest layer"
                      onChange={(e) => { const v = Number(e.target.value); setZHi(v); if (v < zLo) setZLo(v); }} /></label>
                  <div className="mt-1 text-ink-3">Paint and box edit every layer in the band; only the band is drawn.</div>
                </div>
                <div>
                  <div className="label-caps mb-1">Carry forward</div>
                  <label className="flex items-center gap-2">Next
                    <input type="number" min={1} max={200} value={count} onChange={(e) => setCount(Number(e.target.value) || 1)}
                      className="field w-16" aria-label="grids to carry to" /> grids</label>
                  <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                    <button type="button" className="btn" disabled={!!busy || !lastOps || !grid.placed} onClick={() => propagate(true)}
                      data-tip="carry only the last edit, through the ego motion">Last edit</button>
                    <button type="button" className="btn" disabled={!!busy || !grid.placed || !(grid.set_free.length + grid.set_occupied.length)}
                      onClick={() => propagate(false)} data-tip="carry every edit on this grid">All edits</button>
                  </div>
                </div>
                <div className="text-ink-2">
                  Generated {num(grid.base_occupied)} · now {num(grid.total)} ·{" "}
                  <span className="text-warn">{num(grid.set_occupied.length)} painted</span> ·{" "}
                  <span className="text-block">{num(grid.set_free.length)} cleared</span>
                  {grid.truncated && <div className="text-warn">Drawing {num(grid.voxels.length)} of {num(grid.total)} voxels.</div>}
                </div>
              </>
            )}
            {note && <div role="status" className="text-ink-2">{note}</div>}
            {err && <div role="alert" className="rounded border border-block/40 bg-block/10 p-2 text-block">{err}</div>}
          </div>
        </Inspector>
        <main className="relative flex flex-1 flex-col">
          <div className="flex items-center gap-2 border-b border-line px-3 py-2 text-[12px]">
            <Segmented options={[{ value: "bev", label: "Bird's-eye (edit)" }, { value: "perspective", label: "Orbit (view)" }]}
              value={view} onChange={(v) => setView(v as "bev" | "perspective")} />
            <span className="text-ink-3">O move · P paint · B box · 1 occupied · 2 free · 3 reset</span>
            <span className="ml-auto flex items-center gap-3 text-ink-3">
              <span><span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-warn" />painted</span>
              <span><span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-block" />cleared</span>
            </span>
          </div>
          <div className="relative flex-1">
            {grid ? (
              <VoxelEditor grid={grid} view={view} tool={tool} value={value} zLo={zLo} zHi={zHi} onOps={onOps} />
            ) : (
              <div className="flex h-full items-center justify-center">
                <EmptyState icon="cube" title={ov ? "No occupancy grids yet" : "No session loaded"}>
                  {ov ? "Generate grids from LiDAR or, for a session without LiDAR, from the cameras." : "Choose a session on the left."}
                </EmptyState>
              </div>
            )}
            {busy && <div role="status" className="absolute right-3 top-3 rounded bg-bg-2 px-2 py-1 text-[12px] text-ink-2">{busy}...</div>}
          </div>
        </main>
      </div>
    </PageShell>
  );
}
