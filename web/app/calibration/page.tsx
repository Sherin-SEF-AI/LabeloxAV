"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { CalibDetail, CalibResolved, CalibSession, SessionRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import Inspector from "@/components/shell/Inspector";
import { StateBadge } from "@/components/StateBadge";

// M3.0 calibration validation report viewer: per-camera reprojection + FOV check + time offset, with a
// clear pass/fail. A failing session is excluded from 3D and multi-camera work.

export default function CalibrationPage() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [validated, setValidated] = useState<CalibSession[]>([]);
  const [pick, setPick] = useState("");
  const [detail, setDetail] = useState<CalibDetail | null>(null);
  const [busy, setBusy] = useState(false);
  // M-CAL.3 real-calibration ingestion + trust surface
  const [resolved, setResolved] = useState<CalibResolved | null>(null);
  const [specCam, setSpecCam] = useState("");
  const [focalMode, setFocalMode] = useState<"fov" | "fx">("fov");
  const [focalVal, setFocalVal] = useState("37");
  const [heightVal, setHeightVal] = useState("1.5");
  const [pitchVal, setPitchVal] = useState("0");
  const [impCam, setImpCam] = useState("cam_f");
  const [impText, setImpText] = useState("");
  const [ing, setIng] = useState("");

  const load = useCallback(async () => {
    setValidated(await api.calibrationSessions().catch(() => []));
  }, []);

  const loadResolved = useCallback(async (sid: string) => {
    if (!sid) return;
    const r = await api.calibrationResolved(sid).catch(() => null);
    setResolved(r);
    if (r && r.cameras[0]) setSpecCam((c) => c || r.cameras[0].cam_id);
  }, []);

  useEffect(() => {
    load();
    api.sessions().then((s) => { setSessions(s); if (s[0]) setPick(s[0].session_id); });
  }, [load]);

  useEffect(() => { setSpecCam(""); loadResolved(pick); }, [pick, loadResolved]);

  const ingest = async (fn: () => Promise<unknown>, label: string) => {
    setIng(label + "...");
    try { await fn(); await loadResolved(pick); setIng(label + " done"); }
    catch (e) { setIng(label + " failed: " + String(e)); }
  };
  const applySpec = () => ingest(() => api.calibrationSetSpec(pick, { [specCam]: {
    ...(focalMode === "fov" ? { hfov_deg: Number(focalVal) } : { fx: Number(focalVal) }),
    height_m: Number(heightVal), pitch_deg: Number(pitchVal) } }, "measured"), "set spec");
  const runEstimate = () => ingest(() => api.calibrationEstimate(pick), "estimate");
  const runImport = () => ingest(() => api.calibrationImport(pick, { cam_id: impCam, format: "kitti", calib_text: impText }), "import");

  const SRC_DOT: Record<string, string> = { measured: "bg-pass", dataset: "bg-info", estimated: "bg-warn", nominal: "bg-ink-3" };

  const validate = async () => {
    if (!pick) return;
    setBusy(true);
    try {
      setDetail(await api.calibrationValidate(pick));
      await load();
    } finally {
      setBusy(false);
    }
  };

  const open = async (sid: string) => setDetail(await api.calibrationDetail(sid));

  const validateAction = (
    <button onClick={validate} disabled={busy || !pick} className="border border-accent text-accent px-2 py-1 font-mono text-[11px] hover:bg-accent/10 disabled:opacity-50">{busy ? "..." : "validate"}</button>
  );

  return (
    <PageShell active="CALIBRATION" title="Calibration" primaryAction={validateAction}>
      <div className="flex h-full min-h-0">
        <Inspector side="left" title="Sessions">
          <div className="p-2 space-y-3 font-mono text-[11px]">
            <div className="space-y-1">
              <span className="block text-ink-3">validate calibration on</span>
              <select value={pick} onChange={(e) => setPick(e.target.value)} className="w-full bg-bg border border-line px-2 py-1 text-ink">
                {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.vehicle_id} / {s.session_id.slice(0, 8)}</option>)}
              </select>
              <span className="block text-ink-3">a failing session is excluded from 3D + multi-camera work</span>
            </div>
            <div className="space-y-1">
              <div className="uppercase text-ink-3 border-b hairline pb-1">validated sessions</div>
              {validated.length ? validated.map((s) => (
                <button key={s.session_id} onClick={() => open(s.session_id)} className="w-full flex items-center gap-2 px-1 py-0.5 hover:bg-line text-left">
                  <span className={`w-2 h-2 rounded-full ${s.overall === "fail" ? "bg-block" : "bg-pass"}`} />
                  <span className="text-ink-2 truncate flex-1">{s.vehicle_id} / {s.session_id.slice(0, 8)}</span>
                  <span className="text-ink-3">{s.cameras} cam</span>
                </button>
              )) : <div className="text-ink-3 text-center py-4">none validated yet</div>}
            </div>
          </div>
        </Inspector>

        <section className="flex-1 min-h-0 overflow-auto p-4">
          <div className="panel">
            <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2 flex items-center gap-2">
              report {detail && <StateBadge state={detail.overall} />}
            </div>
            {detail ? (
              <table className="w-full font-mono text-[11px]">
                <thead><tr className="text-ink-3 text-left border-b hairline"><th className="px-3 py-1">camera</th><th>model</th><th>reproj px</th><th>implied FOV</th><th>expected</th><th>time offset</th><th className="text-right pr-3">status</th></tr></thead>
                <tbody>
                  {(detail.validations ?? []).map((c) => (
                    <tr key={c.cam_id} className="border-b hairline">
                      <td className="px-3 py-1 text-ink-2">{c.cam_id}</td>
                      <td className="text-ink-3">{c.model}</td>
                      <td className="text-ink-3">{c.reproj_error_px != null ? c.reproj_error_px.toFixed(2) : "-"}</td>
                      <td className={c.fov_check.ok ? "text-ink-3" : "text-block"}>{c.fov_check.implied_fov_deg}&deg;</td>
                      <td className="text-ink-3">{c.fov_check.expected_fov_deg ?? "-"}&deg;</td>
                      <td className="text-ink-3">{c.time_offset_ns != null ? `${(c.time_offset_ns / 1e6).toFixed(2)} ms` : "-"}</td>
                      <td className="text-right pr-3"><StateBadge state={c.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="font-mono text-xs text-ink-3 py-10 text-center">validate a session, or pick one from the list.</div>
            )}
          </div>
        </section>

        <Inspector side="right" title="Real calibration">
          <div className="p-2 space-y-3 font-mono text-[11px]">
            {resolved && (
              <div className="flex items-center gap-2">
                <span className="text-ink-3">trust</span>
                <span className={`w-2 h-2 rounded-full ${SRC_DOT[resolved.trust.level] ?? "bg-ink-3"}`} />
                <span className="text-ink-2">{resolved.trust.level}</span>
                <span className="ml-auto text-ink-3">mean q {resolved.trust.mean_quality}</span>
              </div>
            )}
            <div className="space-y-1">
              <div className="uppercase text-ink-3 border-b hairline pb-1">resolved cameras</div>
              {resolved?.cameras.length ? resolved.cameras.map((c) => (
                <div key={c.cam_id} className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${SRC_DOT[c.source] ?? "bg-ink-3"}`} />
                  <span className="text-ink-2 w-12 truncate">{c.cam_id}</span>
                  <span className="text-ink-3">{c.source}</span>
                  <span className="ml-auto text-ink-3">fx {c.fx} / {c.pitch_deg}&deg;</span>
                </div>
              )) : <div className="text-ink-3 text-center py-2">pick a session</div>}
            </div>
            <div className="space-y-1.5 border-t hairline pt-2">
              <div className="uppercase text-ink-3">set rig spec (measured)</div>
              <select value={specCam} onChange={(e) => setSpecCam(e.target.value)} className="w-full bg-bg border border-line px-2 py-1 text-ink">
                {resolved?.cameras.map((c) => <option key={c.cam_id} value={c.cam_id}>{c.cam_id}</option>)}
              </select>
              <div className="flex gap-1">
                <select value={focalMode} onChange={(e) => setFocalMode(e.target.value as "fov" | "fx")} className="bg-bg border border-line px-1 text-ink">
                  <option value="fov">FOV&deg;</option><option value="fx">fx px</option>
                </select>
                <input value={focalVal} onChange={(e) => setFocalVal(e.target.value)} className="flex-1 bg-bg border border-line px-2 py-1 text-ink" />
              </div>
              <div className="flex gap-1">
                <label className="flex-1 flex items-center gap-1"><span className="text-ink-3">h m</span><input value={heightVal} onChange={(e) => setHeightVal(e.target.value)} className="w-full bg-bg border border-line px-1 py-1 text-ink" /></label>
                <label className="flex-1 flex items-center gap-1"><span className="text-ink-3">pitch&deg;</span><input value={pitchVal} onChange={(e) => setPitchVal(e.target.value)} className="w-full bg-bg border border-line px-1 py-1 text-ink" /></label>
              </div>
              <button onClick={applySpec} disabled={!pick || !specCam} className="w-full border border-accent text-accent py-1 hover:bg-accent/10 disabled:opacity-50">apply to session</button>
            </div>
            <button onClick={runEstimate} disabled={!pick} className="w-full border border-line text-ink-2 py-1 hover:border-accent disabled:opacity-50">estimate from road lines</button>
            <div className="space-y-1.5 border-t hairline pt-2">
              <div className="uppercase text-ink-3">import KITTI calib (dataset)</div>
              <input value={impCam} onChange={(e) => setImpCam(e.target.value)} placeholder="cam_id" className="w-full bg-bg border border-line px-2 py-1 text-ink" />
              <textarea value={impText} onChange={(e) => setImpText(e.target.value)} placeholder="paste calib.txt (P2: fx 0 cx ...)" rows={3} className="w-full bg-bg border border-line px-2 py-1 text-ink resize-none" />
              <button onClick={runImport} disabled={!pick || !impText} className="w-full border border-line text-ink-2 py-1 hover:border-accent disabled:opacity-50">import</button>
            </div>
            {ing && <div className="text-ink-3">{ing}</div>}
          </div>
        </Inspector>
      </div>
    </PageShell>
  );
}
