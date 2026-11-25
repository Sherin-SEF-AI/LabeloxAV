"use client";

// The 3D and BEV point cloud viewer (M-L1.3). Pick a session, pick a cloud and a cleaned variant, and the
// packed binary points render in an interactive perspective view and a top-down bird's-eye panel. Colour by
// height, intensity (road markings light up), or source. Large clouds decimate for interactivity, with full
// resolution on demand. Click two points to measure.

import SessionSelect from "@/components/shell/SessionSelect";
import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { api, lidarCloudPoints, type LidarCloud, type LidarPoints , humanizeError } from "@/lib/api";
import type { ColorBy } from "@/components/lidar/PointCloudViewer";
import PageShell from "@/components/shell/PageShell";
import Inspector from "@/components/shell/Inspector";
import ScoreBar from "@/components/shell/ScoreBar";
import EmptyState from "@/components/ui/EmptyState";
import { fromNs, humanize, num, shortId } from "@/lib/format";

const PointCloudViewer = dynamic(() => import("@/components/lidar/PointCloudViewer"), { ssr: false });

const COLOR_OPTIONS: { key: ColorBy; label: string; hint: string }[] = [
  { key: "height", label: "Height", hint: "elevation ramp" },
  { key: "intensity", label: "Intensity", hint: "road markings" },
  { key: "source", label: "Source", hint: "by sensor" },
];

export default function LidarViewerPage() {
  const [sessionId, setSessionId] = useState("");
  const [clouds, setClouds] = useState<LidarCloud[]>([]);
  const [selected, setSelected] = useState<LidarCloud | null>(null);
  const [variant, setVariant] = useState("raw");
  const [colorBy, setColorBy] = useState<ColorBy>("intensity");
  const [budget, setBudget] = useState(300000);
  const [full, setFull] = useState(false);
  const [data, setData] = useState<LidarPoints | null>(null);
  const [segData, setSegData] = useState<{ points: Float32Array; count: number; lowConfFrac: number } | null>(null);
  const [measure, setMeasure] = useState<number | null>(null);
  const [showEgo, setShowEgo] = useState(true);
  const [trajectory, setTrajectory] = useState<{ x: number; y: number }[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [loadedSession, setLoadedSession] = useState<string | null>(null);

  const loadClouds = useCallback(async (sid: string) => {
    if (!sid.trim()) { setErr("Choose a session first."); return; }
    setErr(null);
    try {
      const r = await api.lidarClouds(sid.trim());
      setClouds(r.clouds);
      setLoadedSession(sid.trim());
      if (!r.clouds.length) setErr("No clouds in this session. Build one from its camera frames below.");
    } catch (e) {
      setErr(humanizeError(e));
      setClouds([]);
    }
  }, []);

  const loadPoints = useCallback(async (cloud: LidarCloud, v: string, max: number, isFull: boolean) => {
    setBusy(true);
    setErr(null);
    setMeasure(null);
    try {
      const d = await lidarCloudPoints(cloud.cloud_id, { variant: v, max, full: isFull });
      setData(d);
    } catch (e) {
      setErr(humanizeError(e));
    } finally {
      setBusy(false);
    }
  }, []);

  // refetch points whenever the selection, variant, budget, or full toggle changes
  useEffect(() => {
    if (selected) loadPoints(selected, variant, budget, full);
  }, [selected, variant, budget, full, loadPoints]);

  const runSegment = async () => {
    if (!selected) return;
    setBusy(true);
    setErr(null);
    try {
      const { lidarSegmentationPoints } = await import("@/lib/api");
      await api.lidarSegment(selected.cloud_id);
      const sp = await lidarSegmentationPoints(selected.cloud_id, budget);
      setSegData(sp);
      setColorBy("segment");
    } catch (e) {
      setErr(humanizeError(e));
    } finally {
      setBusy(false);
    }
  };

  const pickCloud = (c: LidarCloud) => {
    setSelected(c);
    setSegData(null);
    if (colorBy === "segment") setColorBy("intensity");
    setVariant(c.variants.includes(variant) ? variant : "raw");
    api.lidarTrajectory(sessionId.trim(), c.ts_ns)
      .then((r) => setTrajectory(r.path || []))
      .catch(() => setTrajectory([]));
  };

  const build = async () => {
    if (!sessionId.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.lidarBuild(sessionId.trim(), 1);
      if (r.clouds === 0) setErr("No camera frame groups to build from in this session.");
      await loadClouds(sessionId);
    } catch (e) {
      setErr(humanizeError(e));
    } finally {
      setBusy(false);
    }
  };

  const loadForm = (
    <form
      onSubmit={(e) => { e.preventDefault(); loadClouds(sessionId); }}
      className="flex gap-2"
    >
      <SessionSelect value={sessionId} onChange={setSessionId} className="min-w-0 flex-1" />
      <button type="submit" className="btn btn-primary">Load</button>
      <button
        type="button"
        onClick={build}
        disabled={busy || !sessionId.trim()}
        title="estimate a point cloud from this session's camera frames (no LiDAR sensor needed)"
        className="btn"
      >
        Build pseudo-LiDAR
      </button>
    </form>
  );

  return (
    <PageShell
      active="LIDAR"
      right={loadedSession ? <span className="text-[12px] text-ink-3">{clouds.length} cloud{clouds.length === 1 ? "" : "s"} in this session</span> : undefined}
      primaryAction={loadForm}
    >
      {/* Pinned dark in either theme. A point cloud is drawn for a dark ground, and the panels over it are
          read against that ground rather than against the page, so the palette under this element is the
          dark one whatever the rest of the app is using (see the selectors in app/globals.css). */}
      <div data-theme="dark" className="flex h-full bg-[#0a0e14] text-ink">
      <Inspector title="Clouds" side="left" width="w-80">
      <div className="flex flex-col gap-4 p-4">
        <div>
          <div className="section-title">Point clouds</div>
          <div className="section-desc mt-0.5">Real LiDAR scans or pseudo-LiDAR estimated from the cameras.
            Choose a session above, press Load, then pick a cloud.</div>
        </div>

        {clouds.length > 0 && (
          <div>
            <div className="mb-1.5 text-[12px] font-medium text-ink-2">Clouds <span className="text-ink-3 font-normal">({clouds.length})</span></div>
            <div className="flex max-h-52 flex-col gap-1 overflow-y-auto">
              {clouds.map((c) => (
                <button
                  key={c.cloud_id}
                  type="button"
                  onClick={() => pickCloud(c)}
                  aria-pressed={selected?.cloud_id === c.cloud_id}
                  className={`rounded border px-2.5 py-1.5 text-left text-[12px] ${
                    selected?.cloud_id === c.cloud_id
                      ? "border-accent/60 bg-accent/15"
                      : "border-line bg-bg-2 hover:border-line-3"
                  }`}
                >
                  <div className="flex justify-between">
                    <span className="font-medium">{c.source === "lidar" ? "LiDAR" : humanize(c.source)}</span>
                    <span className="text-ink-3 tabular-nums">{num(c.point_count)} points</span>
                  </div>
                  <div className="truncate text-[11px] text-ink-3" title={`${c.cloud_id} · ts_ns ${c.ts_ns}`}>
                    {fromNs(c.ts_ns)} · <span className="font-mono">{shortId(c.cloud_id)}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {selected && (
          <>
            <div>
              <div className="mb-1.5 text-[12px] font-medium text-ink-2">Variant</div>
              <div className="flex flex-wrap gap-1">
                {selected.variants.map((v) => (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setVariant(v)}
                    aria-pressed={variant === v}
                    className={`btn ${variant === v ? "btn-on" : ""}`}
                  >
                    {humanize(v)}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-1.5 text-[12px] font-medium text-ink-2">Colour by</div>
              <div className="flex gap-1">
                {COLOR_OPTIONS.map((o) => (
                  <button
                    key={o.key}
                    type="button"
                    title={o.hint}
                    onClick={() => setColorBy(o.key)}
                    aria-pressed={colorBy === o.key}
                    className={`btn flex-1 ${colorBy === o.key ? "btn-on" : ""}`}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button type="button" onClick={runSegment} disabled={busy}
                title="label each point (road, vehicle, vegetation...) and colour the cloud by it"
                className="btn flex-1">
                Segment overlay
              </button>
              {segData && (
                <ScoreBar label="low confidence" value={segData.lowConfFrac} tone="warn" />
              )}
            </div>

            <label className="flex items-center gap-2 text-[12px] text-ink-2">
              <input type="checkbox" checked={showEgo} onChange={(e) => setShowEgo(e.target.checked)} />
              Show ego vehicle, sensors and trajectory ({num(trajectory.length)} GNSS points)
            </label>

            <div>
              <div className="mb-1 flex justify-between text-[12px]">
                <span className="font-medium text-ink-2">Point budget</span>
                <span className="text-ink-2 tabular-nums">{full ? "Full" : num(budget)}</span>
              </div>
              <input
                type="range"
                min={50000}
                max={1000000}
                step={50000}
                value={budget}
                disabled={full}
                onChange={(e) => setBudget(Number(e.target.value))}
                aria-label="point budget"
                className="w-full"
              />
              <label className="mt-1 flex items-center gap-2 text-[12px] text-ink-2">
                <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} />
                Full resolution ({num(selected.point_count)} points)
              </label>
            </div>

            <div className="rounded border border-line bg-bg-2 p-2.5 text-[12px] text-ink-2 space-y-0.5">
              <div className="flex justify-between"><span>Points drawn</span><span className="tabular-nums">{num(data?.count)}</span></div>
              <div className="flex justify-between"><span>Thinned for speed</span><span>{data?.decimated ? "Yes" : "No"}</span></div>
              <div className="flex justify-between">
                <span>Distance</span>
                <span className="text-warn tabular-nums">{measure != null ? `${measure.toFixed(2)} m` : "Click two points"}</span>
              </div>
            </div>
          </>
        )}

        {err && <div role="alert" className="rounded border border-block/40 bg-block/10 p-2.5 text-[12px] text-block">{err}</div>}
      </div>
      </Inspector>

      <main className="relative flex flex-1 flex-col">
        {busy && (
          <div role="status" className="absolute left-1/2 top-4 z-10 -translate-x-1/2 rounded bg-black/70 px-3 py-1 text-[12px]">
            Loading points...
          </div>
        )}
        <div className="relative flex-1 border-b border-line">
          <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">3D view</div>
          {data && (
            <PointCloudViewer
              points={colorBy === "segment" && segData ? segData.points : data.points}
              count={colorBy === "segment" && segData ? segData.count : data.count}
              colorBy={colorBy}
              intensityRange={[data.intensityMin, data.intensityMax]}
              source={data.source}
              mode="perspective"
              onMeasure={setMeasure}
              showEgo={showEgo}
              trajectory={trajectory}
            />
          )}
          {!data && !busy && (
            <div className="flex h-full items-center justify-center">
              <EmptyState icon="cube" title={loadedSession ? (clouds.length ? "Pick a cloud" : "No point clouds in this session") : "No cloud loaded"}>
                {loadedSession
                  ? (clouds.length
                      ? "Choose one of the clouds on the left to draw it here and in the bird's-eye view below."
                      : "This session has no LiDAR scans. Build pseudo-LiDAR to estimate clouds from its camera frames.")
                  : "Choose a session at the top and press Load. Its clouds appear on the left."}
              </EmptyState>
            </div>
          )}
        </div>
        <div className="relative h-1/3 min-h-[180px]">
          <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">Bird&apos;s-eye view</div>
          {data && (
            <PointCloudViewer
              points={colorBy === "segment" && segData ? segData.points : data.points}
              count={colorBy === "segment" && segData ? segData.count : data.count}
              colorBy={colorBy}
              intensityRange={[data.intensityMin, data.intensityMax]}
              source={data.source}
              mode="bev"
              pointSize={0.5}
              showEgo={showEgo}
              trajectory={trajectory}
            />
          )}
        </div>
      </main>
      </div>
    </PageShell>
  );
}
