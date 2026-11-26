"use client";

// Radar and event cameras (#10). Radar: import sweeps (nuScenes RADAR_* .pcd with its tables, an MCAP with
// radar_msgs/RadarScan or PointCloud2-with-velocity topics, or one .pcd), see every return on the camera frame
// and from above with its Doppler, and fuse the Doppler into the speed of the session's 3D tracks. Event
// cameras: import a recording and accumulate it into frames that are labelled in the normal editor.
// There is no radar or event data on this server yet: both paths are tested on generated files in the
// published formats, and the page says so where it is empty.

import { useCallback, useEffect, useMemo, useState } from "react";

import { api, humanizeError, uploadMultipart } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Bar, Button, ErrLine, Field, NumberInput, Panel, Segmented, Stat, TextArea, TextInput } from "@/components/ui/kit";
import { fromNs, num, shortId } from "@/lib/format";
import { useToolJob } from "@/lib/tools/toolJobs";
import {
  radarBev, radarFuse, radarFusion, radarImport, radarOverlay, radarSweeps, sensorsOverview,
  type RadarBev, type RadarFusionTrack, type RadarOverlay, type RadarSweeps, type SensorsOverview,
} from "@/lib/tools/sensorsApi";
import EventsPanel from "./EventsPanel";
import { DopplerLegend, RadarFromAbove, RadarOnFrame } from "./RadarViews";

type Tab = "radar" | "events";

function parseExtrinsics(text: string): Record<string, number[][]> | null {
  if (!text.trim()) return null;
  const v = JSON.parse(text);
  if (typeof v !== "object" || Array.isArray(v)) throw new Error("extrinsics: an object of topic -> 4x4 matrix");
  return v as Record<string, number[][]>;
}

export default function SensorsPage() {
  const [tab, setTab] = useState<Tab>("radar");
  const [ov, setOv] = useState<SensorsOverview | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [allSessions, setAllSessions] = useState<{ session_id: string; label: string }[]>([]);

  const load = useCallback(() => {
    sensorsOverview().then(setOv).catch((e) => setLoadErr(humanizeError(e)));
  }, []);
  useEffect(() => {
    load();
    api.sessionsPage({ limit: 200 }).then((p) => setAllSessions(p.sessions.map((s) => ({
      session_id: s.session_id, label: `${s.route ?? s.vehicle_id}${s.city ? ` / ${s.city}` : ""} / ${s.session_id.slice(0, 8)}`,
    })))).catch(() => {});
  }, [load]);

  return (
    <PageShell active="Radar and events">
      <div className="page-wrap flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <Segmented value={tab} onChange={setTab} options={[
            { value: "radar", label: `Radar${ov ? ` (${ov.radar.length})` : ""}` },
            { value: "events", label: `Event cameras${ov ? ` (${ov.events.length})` : ""}` }]} />
          {loadErr && <span className="text-[12px] text-block">{loadErr}</span>}
        </div>
        {tab === "radar"
          ? <RadarTab ov={ov} failed={!!loadErr} sessions={allSessions} onChanged={load} />
          : <EventsPanel streams={ov?.events ?? []} sessions={allSessions} onChanged={load} />}
      </div>
    </PageShell>
  );
}

function RadarTab({ ov, failed, sessions, onChanged }: { ov: SensorsOverview | null; failed: boolean;
  sessions: { session_id: string; label: string }[]; onChanged: () => void }) {
  const [sid, setSid] = useState<string>("");
  const [sw, setSw] = useState<RadarSweeps | null>(null);
  const [fi, setFi] = useState(0);
  const [overlay, setOverlay] = useState<RadarOverlay | null>(null);
  const [bev, setBev] = useState<RadarBev | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [fusion, setFusion] = useState<{ tracks_3d: number; fused_tracks: RadarFusionTrack[] } | null>(null);
  const [tol, setTol] = useState(3);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // import form
  const [target, setTarget] = useState("");
  const [fmt, setFmt] = useState<"nuscenes" | "mcap" | "pcd">("nuscenes");
  const [file, setFile] = useState<File | null>(null);
  const [path, setPath] = useState("");
  const [ext, setExt] = useState("");
  const [pcdTs, setPcdTs] = useState("");
  const [up, setUp] = useState<number | null>(null);

  useEffect(() => { if (!sid && ov?.radar.length) setSid(ov.radar[0].session_id); }, [ov, sid]);

  const loadSession = useCallback(async (id: string) => {
    setSw(null); setOverlay(null); setBev(null); setFusion(null);
    if (!id) return;
    try {
      const [s, f] = await Promise.all([radarSweeps(id), radarFusion(id)]);
      setSw(s);
      setFusion(f);
      const first = s.frames.findIndex((x) => x.radar_dt_ms != null && Math.abs(x.radar_dt_ms) <= 75);
      setFi(Math.max(0, first));
    } catch (e) { setErr(humanizeError(e)); }
  }, []);
  useEffect(() => { void loadSession(sid); }, [sid, loadSession]);

  const frame = sw?.frames[fi];
  useEffect(() => {
    if (!frame || !sid) return;
    let stop = false;
    Promise.all([radarOverlay(frame.frame_id), radarBev(sid, frame.ts_ns)]).then(([o, b]) => {
      if (!stop) { setOverlay(o); setBev(b); }
    }).catch((e) => { if (!stop) setErr(humanizeError(e)); });
    return () => { stop = true; };
  }, [frame, sid]);

  const { job } = useToolJob(jobId, {
    onDone: (j) => { onChanged(); void loadSession(sid); if (j.status === "error") setErr(j.error); },
  });
  const busy = job && !["done", "error", "canceled"].includes(job.status);

  const doImport = async () => {
    setErr(null);
    try {
      if (!target) { setErr("Choose the session the radar belongs to"); return; }
      let uri = path.trim();
      if (file) { setUp(0); uri = await uploadMultipart(file, setUp); }
      if (!uri) { setErr("Choose a file or give a server path"); return; }
      const j = await radarImport({ session_id: target, source_uri: uri, format: fmt,
        extrinsics: fmt === "mcap" ? parseExtrinsics(ext) : null, ts_ns: fmt === "pcd" && pcdTs ? Number(pcdTs) : null });
      setJobId(j.job_id);
      setSid(target);
    } catch (e) { setErr(humanizeError(e)); } finally { setUp(null); }
  };

  const fuse = async () => {
    setErr(null);
    try { setJobId((await radarFuse(sid, { tol_mps: tol, apply: true })).job_id); } catch (e) { setErr(humanizeError(e)); }
  };

  const nIn = useMemo(() => (overlay?.sensors ?? []).filter((s) => !hidden.has(s.sensor)).reduce((a, s) => a + s.in_image, 0), [overlay, hidden]);
  const row = ov?.radar.find((r) => r.session_id === sid);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[340px_1fr] gap-4">
      <div className="flex flex-col gap-4 min-w-0">
        <Panel title="Import radar" desc="Sweeps outside the session's time span are skipped, which also picks one nuScenes scene out of a directory.">
          <div className="flex flex-col gap-1">
            <Field label="Session">
              <select className="field w-44" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">Choose a session</option>
                {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.label}</option>)}
              </select>
            </Field>
            <Field label="Format">
              <Segmented value={fmt} onChange={setFmt} options={[
                { value: "nuscenes", label: "nuScenes", tip: "A .zip of the dataset directory with its tables and RADAR_* .pcd files" },
                { value: "mcap", label: "MCAP", tip: "radar_msgs/RadarScan or PointCloud2 with velocity fields" },
                { value: "pcd", label: "One .pcd" }]} />
            </Field>
            <input type="file" accept={fmt === "nuscenes" ? ".zip" : fmt === "mcap" ? ".mcap" : ".pcd"} className="text-[12px]"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)} aria-label="Radar file" />
            <Field label="or server path"><TextInput value={path} onChange={setPath} placeholder="/data/nuscenes" className="w-44" /></Field>
            {fmt === "mcap" && (
              <div className="mt-1">
                <div className="text-[12px] text-ink-2 mb-1">Sensor to ego per topic (JSON, 4x4); identity when empty</div>
                <TextArea value={ext} onChange={setExt} rows={3} />
              </div>
            )}
            {fmt === "pcd" && <Field label="Capture time (ns)"><TextInput value={pcdTs} onChange={setPcdTs} className="w-44" /></Field>}
            <div className="flex items-center gap-2 mt-2">
              <Button variant="primary" onClick={doImport} disabled={!!busy} tip="Needs the reviewer role">Import</Button>
              {up != null && <span className="flex-1 flex items-center gap-2 text-[12px]"><Bar frac={up} />uploading</span>}
              {busy && <span className="flex-1 flex items-center gap-2 text-[12px]"><Bar frac={job!.progress} />{job!.stage ?? job!.status}</span>}
            </div>
          </div>
        </Panel>
        <Panel title="Sessions with radar">
          {!ov ? <p className="text-[12px] text-ink-3">{failed ? "Unavailable." : "Loading..."}</p> : !ov.radar.length ? <p className="text-[12px] text-ink-3">None yet.</p> : (
            <ul className="flex flex-col gap-1">
              {ov.radar.map((r) => (
                <li key={r.session_id}>
                  <button type="button" onClick={() => setSid(r.session_id)}
                    className={`w-full text-left px-2 py-1.5 rounded border text-[12px] ${sid === r.session_id ? "border-accent bg-accent/10" : "border-line hover:bg-head"}`}>
                    <div className="text-ink">{r.name ?? shortId(r.session_id)}</div>
                    <div className="text-ink-3">{num(r.sweeps)} sweeps from {r.sensors} sensor{r.sensors === 1 ? "" : "s"}, {num(r.points)} returns</div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="flex flex-col gap-4 min-w-0">
        {!sid ? (
          <div className="panel"><EmptyState icon="activity" title="No radar yet">There is no radar data on this server. Import nuScenes radar sweeps or an MCAP with radar topics on the left; each sweep is placed on the session&apos;s camera frames by its time.</EmptyState></div>
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Sweeps" value={num(row?.sweeps ?? sw?.sweeps.length)} sub={sw?.sensors.join(", ")} />
              <Stat label="Returns" value={num(row?.points)} />
              <Stat label="On this frame" value={num(nIn)} sub={overlay ? `calibration: ${overlay.calibration}` : undefined} />
              <Stat label="3D tracks fused" value={`${fusion?.fused_tracks.length ?? 0} of ${fusion?.tracks_3d ?? 0}`} />
            </div>
            <Panel title="On the camera" right={<DopplerLegend />}
              desc="Each return projected with the session's calibration at the frame's instant.">
              {sw && sw.frames.length > 0 && frame ? (
                <>
                  <div className="flex items-center gap-3 mb-2 text-[12px]">
                    <input type="range" min={0} max={sw.frames.length - 1} value={fi} onChange={(e) => setFi(Number(e.target.value))} className="flex-1" aria-label="Frame" />
                    <span className="tabular-nums text-ink-2">{fromNs(frame.ts_ns)} · {frame.cam_id}</span>
                    {frame.radar_dt_ms != null && <Badge state={Math.abs(frame.radar_dt_ms) <= 75 ? "pass" : "muted"}>radar {frame.radar_dt_ms} ms</Badge>}
                  </div>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {sw.sensors.map((s) => (
                      <button key={s} type="button" onClick={() => setHidden((h) => { const n = new Set(h); if (n.has(s)) n.delete(s); else n.add(s); return n; })}
                        className={`chip ${hidden.has(s) ? "" : "chip-on"}`}>{s}</button>
                    ))}
                  </div>
                  <div className="grid grid-cols-1 2xl:grid-cols-[1fr_360px] gap-4">
                    <RadarOnFrame frameId={frame.frame_id} overlay={overlay} hidden={hidden} />
                    <div>
                      <div className="label-caps mb-1">From above (ego frame, forward is up)</div>
                      <RadarFromAbove bev={bev} hidden={hidden} />
                    </div>
                  </div>
                </>
              ) : <p className="text-[12px] text-ink-3">This session has no camera frames to overlay on.</p>}
            </Panel>
            <Panel title="Speed from Doppler" desc="For each 3D box, the radar returns inside it replace the radial part of the track's velocity, if they agree with it within the tolerance; a reading that disagrees is rejected, not averaged.">
              <div className="flex items-center gap-3 mb-3">
                <Field label="Tolerance (m/s)"><NumberInput value={tol} onChange={(n) => setTol(Math.max(0.1, Math.min(30, n || 3)))} w="w-20" /></Field>
                <Button variant="primary" onClick={fuse} disabled={!!busy} tip="Writes onto the session's 3D tracks; needs the reviewer role">Fuse into tracks</Button>
              </div>
              {!fusion?.fused_tracks.length ? <p className="text-[12px] text-ink-3">{fusion?.tracks_3d ? "Not fused yet." : "This session has no 3D tracks to fuse into."}</p> : (
                <table className="data-table w-full">
                  <thead><tr><th>Track</th><th>Class</th><th>Median speed</th><th>Radar</th><th>Rejected</th><th>No return</th></tr></thead>
                  <tbody>
                    {fusion.fused_tracks.map((t) => (
                      <tr key={t.track_3d_id}>
                        <td className="font-mono">{shortId(t.track_3d_id)}</td><td>{t.class_name.replace(/_/g, " ")}</td>
                        <td className="tabular-nums">{t.median_speed_mps != null ? `${t.median_speed_mps.toFixed(1)} m/s` : "n/a"}</td>
                        <td className="tabular-nums">{t.fused}</td><td className="tabular-nums">{t.rejected}</td><td className="tabular-nums">{t.no_radar}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Panel>
          </>
        )}
        <ErrLine err={err} />
      </div>
    </div>
  );
}
