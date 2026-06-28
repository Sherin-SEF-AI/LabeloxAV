"use client";

// The 3D and BEV point cloud viewer (M-L1.3). Pick a session, pick a cloud and a cleaned variant, and the
// packed binary points render in an interactive perspective view and a top-down bird's-eye panel. Colour by
// height, intensity (road markings light up), or source. Large clouds decimate for interactivity, with full
// resolution on demand. Click two points to measure.

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { api, lidarCloudPoints, type LidarCloud, type LidarPoints } from "@/lib/api";
import type { ColorBy } from "@/components/lidar/PointCloudViewer";

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
  const [measure, setMeasure] = useState<number | null>(null);
  const [showEgo, setShowEgo] = useState(true);
  const [trajectory, setTrajectory] = useState<{ x: number; y: number }[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const loadClouds = useCallback(async (sid: string) => {
    setErr(null);
    try {
      const r = await api.lidarClouds(sid.trim());
      setClouds(r.clouds);
      if (!r.clouds.length) setErr("No clouds in this session. Build one from its camera frames below.");
    } catch (e) {
      setErr(String(e));
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
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  // refetch points whenever the selection, variant, budget, or full toggle changes
  useEffect(() => {
    if (selected) loadPoints(selected, variant, budget, full);
  }, [selected, variant, budget, full, loadPoints]);

  const pickCloud = (c: LidarCloud) => {
    setSelected(c);
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
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-3.5rem)] bg-[#0a0e14] text-neutral-200">
      <aside className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto border-r border-neutral-800 p-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-neutral-500">LiDAR viewer</div>
          <div className="mt-1 text-sm text-neutral-400">3D point clouds, real or pseudo-LiDAR.</div>
        </div>

        <form
          onSubmit={(e) => { e.preventDefault(); loadClouds(sessionId); }}
          className="flex gap-2"
        >
          <input
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            placeholder="session id"
            className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs"
          />
          <button className="rounded bg-cyan-700 px-3 py-1 text-xs font-medium hover:bg-cyan-600">Load</button>
        </form>

        {clouds.length > 0 && (
          <div>
            <div className="mb-1 text-xs uppercase tracking-wider text-neutral-500">Clouds ({clouds.length})</div>
            <div className="flex max-h-44 flex-col gap-1 overflow-y-auto">
              {clouds.map((c) => (
                <button
                  key={c.cloud_id}
                  onClick={() => pickCloud(c)}
                  className={`rounded border px-2 py-1 text-left text-xs ${
                    selected?.cloud_id === c.cloud_id
                      ? "border-cyan-600 bg-cyan-950"
                      : "border-neutral-800 bg-neutral-900 hover:border-neutral-700"
                  }`}
                >
                  <div className="flex justify-between">
                    <span className="font-medium uppercase">{c.source}</span>
                    <span className="text-neutral-500">{c.point_count.toLocaleString()} pts</span>
                  </div>
                  <div className="truncate text-neutral-600">{c.cloud_id.slice(0, 8)} | ts {c.ts_ns}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {selected && (
          <>
            <div>
              <div className="mb-1 text-xs uppercase tracking-wider text-neutral-500">Variant</div>
              <div className="flex flex-wrap gap-1">
                {selected.variants.map((v) => (
                  <button
                    key={v}
                    onClick={() => setVariant(v)}
                    className={`rounded px-2 py-1 text-xs ${
                      variant === v ? "bg-cyan-700" : "bg-neutral-800 hover:bg-neutral-700"
                    }`}
                  >
                    {v.replace("_", " ")}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-1 text-xs uppercase tracking-wider text-neutral-500">Colour by</div>
              <div className="flex gap-1">
                {COLOR_OPTIONS.map((o) => (
                  <button
                    key={o.key}
                    title={o.hint}
                    onClick={() => setColorBy(o.key)}
                    className={`flex-1 rounded px-2 py-1 text-xs ${
                      colorBy === o.key ? "bg-cyan-700" : "bg-neutral-800 hover:bg-neutral-700"
                    }`}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            </div>

            <label className="flex items-center gap-2 text-xs text-neutral-400">
              <input type="checkbox" checked={showEgo} onChange={(e) => setShowEgo(e.target.checked)} />
              Ego, sensors, trajectory ({trajectory.length} GNSS pts)
            </label>

            <div>
              <div className="mb-1 flex justify-between text-xs uppercase tracking-wider text-neutral-500">
                <span>Point budget</span>
                <span className="text-neutral-400">{full ? "full" : budget.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min={50000}
                max={1000000}
                step={50000}
                value={budget}
                disabled={full}
                onChange={(e) => setBudget(Number(e.target.value))}
                className="w-full"
              />
              <label className="mt-1 flex items-center gap-2 text-xs text-neutral-400">
                <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} />
                Full resolution ({selected.point_count.toLocaleString()} pts)
              </label>
            </div>

            <div className="rounded border border-neutral-800 bg-neutral-900 p-2 text-xs text-neutral-400">
              <div className="flex justify-between"><span>rendered</span><span>{data?.count.toLocaleString() ?? "-"}</span></div>
              <div className="flex justify-between"><span>decimated</span><span>{data?.decimated ? "yes" : "no"}</span></div>
              <div className="flex justify-between">
                <span>measure</span>
                <span className="text-amber-300">{measure != null ? `${measure.toFixed(2)} m` : "click 2 pts"}</span>
              </div>
            </div>
          </>
        )}

        <button
          onClick={build}
          disabled={busy || !sessionId.trim()}
          className="rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-xs hover:border-cyan-700 disabled:opacity-40"
        >
          Build pseudo-LiDAR from cameras
        </button>

        {err && <div className="rounded border border-red-900 bg-red-950 p-2 text-xs text-red-300">{err}</div>}
      </aside>

      <main className="relative flex flex-1 flex-col">
        {busy && (
          <div className="absolute left-1/2 top-4 z-10 -translate-x-1/2 rounded bg-black/70 px-3 py-1 text-xs">
            loading points...
          </div>
        )}
        <div className="relative flex-1 border-b border-neutral-800">
          <div className="absolute left-3 top-3 z-10 text-xs uppercase tracking-wider text-neutral-500">3D</div>
          {data && (
            <PointCloudViewer
              points={data.points}
              count={data.count}
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
            <div className="flex h-full items-center justify-center text-sm text-neutral-600">
              Load a session and pick a cloud.
            </div>
          )}
        </div>
        <div className="relative h-1/3 min-h-[180px]">
          <div className="absolute left-3 top-3 z-10 text-xs uppercase tracking-wider text-neutral-500">BEV</div>
          {data && (
            <PointCloudViewer
              points={data.points}
              count={data.count}
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
  );
}
