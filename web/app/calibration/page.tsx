"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { CalibDetail, CalibSession, SessionRow } from "@/lib/types";
import TopNav from "@/components/TopNav";

// M3.0 calibration validation report viewer: per-camera reprojection + FOV check + time offset, with a
// clear pass/fail. A failing session is excluded from 3D and multi-camera work.

const STATUS: Record<string, string> = { pass: "text-pass", warn: "text-warn", fail: "text-block" };

export default function CalibrationPage() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [validated, setValidated] = useState<CalibSession[]>([]);
  const [pick, setPick] = useState("");
  const [detail, setDetail] = useState<CalibDetail | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setValidated(await api.calibrationSessions().catch(() => []));
  }, []);

  useEffect(() => {
    load();
    api.sessions().then((s) => { setSessions(s); if (s[0]) setPick(s[0].session_id); });
  }, [load]);

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

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="CALIBRATION" />
      <main className="flex-1 overflow-auto p-4 space-y-4">
        <div className="panel p-3 flex items-center gap-2 flex-wrap font-mono text-[11px]">
          <span className="text-ink-3">validate calibration on</span>
          <select value={pick} onChange={(e) => setPick(e.target.value)} className="bg-bg border border-line px-2 py-1 text-ink max-w-xs">
            {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.vehicle_id} / {s.session_id.slice(0, 8)}</option>)}
          </select>
          <button onClick={validate} disabled={busy || !pick} className="border border-accent text-accent px-2 py-1 hover:bg-accent/10 disabled:opacity-50">{busy ? "..." : "validate"}</button>
          <span className="ml-auto text-ink-3">a failing session is excluded from 3D + multi-camera work</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <section className="panel">
            <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">validated sessions</div>
            <div className="p-2 font-mono text-[11px] space-y-1">
              {validated.length ? validated.map((s) => (
                <button key={s.session_id} onClick={() => open(s.session_id)} className="w-full flex items-center gap-2 px-1 py-0.5 hover:bg-line text-left">
                  <span className={`w-2 h-2 rounded-full ${s.overall === "fail" ? "bg-block" : "bg-pass"}`} />
                  <span className="text-ink-2 truncate flex-1">{s.vehicle_id} / {s.session_id.slice(0, 8)}</span>
                  <span className="text-ink-3">{s.cameras} cam</span>
                </button>
              )) : <div className="text-ink-3 text-center py-4">none validated yet</div>}
            </div>
          </section>

          <section className="panel lg:col-span-2">
            <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
              report {detail && <span className={STATUS[detail.overall] || "text-ink-2"}>· {detail.overall.toUpperCase()}</span>}
            </div>
            {detail ? (
              <table className="w-full font-mono text-[11px]">
                <thead><tr className="text-ink-3 text-left border-b hairline"><th className="px-3 py-1">camera</th><th>model</th><th>reproj px</th><th>implied FOV</th><th>expected</th><th>time offset</th><th className="text-right pr-3">status</th></tr></thead>
                <tbody>
                  {detail.validations.map((c) => (
                    <tr key={c.cam_id} className="border-b hairline">
                      <td className="px-3 py-1 text-ink-2">{c.cam_id}</td>
                      <td className="text-ink-3">{c.model}</td>
                      <td className="text-ink-3">{c.reproj_error_px != null ? c.reproj_error_px.toFixed(2) : "-"}</td>
                      <td className={c.fov_check.ok ? "text-ink-3" : "text-block"}>{c.fov_check.implied_fov_deg}&deg;</td>
                      <td className="text-ink-3">{c.fov_check.expected_fov_deg ?? "-"}&deg;</td>
                      <td className="text-ink-3">{c.time_offset_ns != null ? `${(c.time_offset_ns / 1e6).toFixed(2)} ms` : "-"}</td>
                      <td className={`text-right pr-3 ${STATUS[c.status] || "text-ink-2"}`}>{c.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="font-mono text-xs text-ink-3 py-10 text-center">validate a session, or pick one from the list.</div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
