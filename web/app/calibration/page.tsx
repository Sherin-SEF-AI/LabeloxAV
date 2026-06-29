"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { CalibDetail, CalibSession, SessionRow } from "@/lib/types";
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
                  {detail.validations.map((c) => (
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
      </div>
    </PageShell>
  );
}
