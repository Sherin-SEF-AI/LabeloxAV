"use client";

import { useEffect, useState } from "react";
import { T } from "@/lib/i18n";
import { api , humanizeError } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";
import type { CalibDetail, CalibResolved, CalibSession, SessionRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import Inspector from "@/components/shell/Inspector";
import { StateBadge } from "@/components/StateBadge";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, shortId } from "@/lib/format";

// M3.0 calibration validation report viewer: per-camera reprojection + FOV check + time offset, with a
// clear pass/fail. A failing session is excluded from 3D and multi-camera work.

const NO_SESSIONS: SessionRow[] = [];
const NO_VALIDATED: CalibSession[] = [];

export default function CalibrationPage() {
  const sessionsQ = useQuery(["sessions"], () => api.sessions());
  const sessions: SessionRow[] = sessionsQ.data ?? NO_SESSIONS;
  // An unreadable list reads as none validated, as before.
  const validatedQ = useQuery(["calibrationSessions"], () => api.calibrationSessions());
  const validated: CalibSession[] = validatedQ.error ? NO_VALIDATED : validatedQ.data ?? NO_VALIDATED;
  const [pick, setPick] = useState("");
  // Start on the newest session, without overriding one already picked when the list refreshes.
  useEffect(() => {
    if (sessions[0]) setPick((cur) => cur || sessions[0].session_id);
  }, [sessions]);
  const [detail, setDetail] = useState<CalibDetail | null>(null);
  const [busy, setBusy] = useState(false);
  // M-CAL.3 real-calibration ingestion + trust surface
  // What this session's calibration resolves to, absent when it cannot be read. keepPrevious: the last
  // session's stays up while the next one loads.
  const resolvedQ = useQuery(pick ? ["calibrationResolved", pick] : null, () => api.calibrationResolved(pick),
    { keepPrevious: true });
  const resolved: CalibResolved | null = resolvedQ.error ? null : resolvedQ.data ?? null;
  const [specCam, setSpecCam] = useState("");
  const [focalMode, setFocalMode] = useState<"fov" | "fx">("fov");
  const [focalVal, setFocalVal] = useState("37");
  const [heightVal, setHeightVal] = useState("1.5");
  const [pitchVal, setPitchVal] = useState("0");
  const [impCam, setImpCam] = useState("cam_f");
  const [impText, setImpText] = useState("");
  const [ing, setIng] = useState("");

  const load = validatedQ.reload;
  const loadResolved = async (_sid: string) => { await resolvedQ.reload(); };

  // Declared in this order on purpose: a new session clears the chosen camera, then the calibration that
  // arrives for it (at once, when cached) picks its first camera unless one has been chosen since.
  useEffect(() => { setSpecCam(""); }, [pick]);
  const loaded = resolvedQ.data;
  useEffect(() => {
    if (loaded && loaded.cameras[0]) setSpecCam((c) => c || loaded.cameras[0].cam_id);
  }, [loaded]);

  const ingest = async (fn: () => Promise<unknown>, label: string) => {
    setIng(label + "...");
    try { await fn(); await loadResolved(pick); setIng(label + ": done"); }
    catch (e) { setIng(label + " failed: " + humanizeError(e)); }
  };
  const applySpec = () => ingest(() => api.calibrationSetSpec(pick, { [specCam]: {
    ...(focalMode === "fov" ? { hfov_deg: Number(focalVal) } : { fx: Number(focalVal) }),
    height_m: Number(heightVal), pitch_deg: Number(pitchVal) } }, "measured"), "Measured values");
  const runEstimate = () => ingest(() => api.calibrationEstimate(pick), "Estimate from road lines");
  const runExtrinsics = async () => {
    setIng("Checking cameras against each other...");
    try {
      const r = await api.calibrationExtrinsics(pick);
      setIng(r.checked ? `Cameras agree to within ${r.worst_sampson_px ?? "-"} px (worst pair)` : `Could not check: ${r.reason}`);
    } catch (e) { setIng("Camera check failed: " + humanizeError(e)); }
  };
  const runImport = () => ingest(() => api.calibrationImport(pick, { cam_id: impCam, format: "kitti", calib_text: impText }), "Import");

  const SRC_DOT: Record<string, string> = { measured: "bg-pass", dataset: "bg-info", estimated: "bg-warn", nominal: "bg-ink-3" };
  // Where a camera's numbers came from, best first. "Nominal" is a default nobody measured.
  const SRC_LABEL: Record<string, string> = { measured: "Measured", dataset: "From the dataset", estimated: "Estimated", nominal: "Default guess" };

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
    <button type="button" onClick={validate} disabled={busy || !pick} className="btn btn-primary">{busy ? "Validating..." : "Validate session"}</button>
  );

  return (
    <PageShell active="CALIBRATION" title="Calibration" primaryAction={validateAction}
      subtitle="Check each camera's calibration before a session is used for 3D or multi-camera work">
      <div className="flex h-full min-h-0">
        <Inspector side="left" title="Sessions">
          <div className="p-3 space-y-4 text-[12.5px]">
            <label className="block space-y-1.5">
              <span className="block text-ink-2">Session to validate</span>
              <select value={pick} onChange={(e) => setPick(e.target.value)} className="field w-full">
                {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.vehicle_id} · {shortId(s.session_id)}</option>)}
              </select>
              <span className="block text-[12px] text-ink-3">A session that fails is left out of 3D and multi-camera work.</span>
            </label>
            <div className="space-y-1">
              <h2 className="section-title border-b hairline pb-1.5">Already validated</h2>
              {validated.length ? validated.map((s) => (
                <button key={s.session_id} type="button" onClick={() => open(s.session_id)}
                  className={`w-full flex items-center gap-2 px-1.5 py-1.5 rounded text-left ${detail?.session_id === s.session_id ? "bg-accent/15" : "hover:bg-head"}`}>
                  <span className={`w-2 h-2 rounded-full shrink-0 ${s.overall === "fail" ? "bg-block" : "bg-pass"}`}
                    aria-label={s.overall === "fail" ? "failed" : "passed"} />
                  <span className="text-ink-2 truncate flex-1">{s.vehicle_id} <span className="font-mono text-[11px] text-ink-3">{shortId(s.session_id)}</span></span>
                  <span className="text-ink-3 text-[12px] shrink-0">{s.cameras === 1 ? "1 camera" : `${s.cameras} cameras`}</span>
                </button>
              )) : <div className="text-ink-3 py-3">None validated yet.</div>}
            </div>
          </div>
        </Inspector>

        <section className="flex-1 min-w-0 min-h-0 overflow-auto">
          <div className="page-wrap">
          <div className="panel">
            <div className="panel-head">
              <h2 className="section-title">Validation report</h2>
              {detail && <StateBadge state={detail.overall} />}
            </div>
            {detail ? (
              <table className="data-table">
                <thead><tr><th>Camera</th><th>Lens model</th><th className="num">Reprojection error</th><th className="num">Field of view</th><th className="num">Expected</th><th className="num">Time offset</th><th className="text-right">Result</th></tr></thead>
                <tbody>
                  {(detail.validations ?? []).map((c) => (
                    <tr key={c.cam_id}>
                      <td className="text-ink">{c.cam_id}</td>
                      <td>{humanize(c.model)}</td>
                      <td className="num">{c.reproj_error_px != null ? `${c.reproj_error_px.toFixed(2)} px` : "-"}</td>
                      <td className={`num ${c.fov_check.ok ? "" : "text-block"}`}>{c.fov_check.implied_fov_deg}&deg;</td>
                      <td className="num">{c.fov_check.expected_fov_deg != null ? <>{c.fov_check.expected_fov_deg}&deg;</> : "-"}</td>
                      <td className="num">{c.time_offset_ns != null ? `${(c.time_offset_ns / 1e6).toFixed(2)} ms` : "-"}</td>
                      <td className="text-right"><StateBadge state={c.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <EmptyState icon="ruler" title="No report open"
                action={<button type="button" onClick={validate} disabled={busy || !pick} className="btn btn-primary">{busy ? "Validating..." : "Validate session"}</button>}>
                Validate the session picked on the left, or open one that was already validated. The report
                checks each camera&apos;s lens fit, field of view and clock offset.
              </EmptyState>
            )}
          </div>
          </div>
        </section>

        <Inspector side="right" title="Camera settings" width="w-80">
          <div className="p-3 space-y-4 text-[12.5px]">
            {resolved && (
              <div className="flex items-center gap-2" data-tip="How much the current numbers can be trusted, from where they came from">
                <span className="text-ink-3">Trust</span>
                <span className={`w-2 h-2 rounded-full ${SRC_DOT[resolved.trust.level] ?? "bg-ink-3"}`} />
                <span className="text-ink-2">{SRC_LABEL[resolved.trust.level] ?? humanize(resolved.trust.level)}</span>
                <span className="ml-auto text-ink-3">quality {resolved.trust.mean_quality}</span>
              </div>
            )}
            <div className="space-y-1.5">
              <h2 className="section-title border-b hairline pb-1.5">Cameras in this session</h2>
              {resolved?.cameras.length ? resolved.cameras.map((c) => (
                <div key={c.cam_id} className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full shrink-0 ${SRC_DOT[c.source] ?? "bg-ink-3"}`} />
                  <span className="text-ink-2 truncate min-w-0" data-tip={c.cam_id}>{c.cam_id}</span>
                  <span className="ml-auto shrink-0 text-ink-3 text-[12px]" data-tip={`Focal length ${c.fx} px, pitch ${c.pitch_deg}°`}>
                    {SRC_LABEL[c.source] ?? humanize(c.source)}
                  </span>
                </div>
              )) : <div className="text-ink-3">Pick a session to see its cameras.</div>}
            </div>
            <div className="space-y-2 border-t hairline pt-3">
              <h2 className="section-title">Enter measured values</h2>
              <label className="block space-y-1">
                <span className="block text-[12px] text-ink-3">Camera</span>
                <select value={specCam} onChange={(e) => setSpecCam(e.target.value)} className="field w-full">
                  {resolved?.cameras.map((c) => <option key={c.cam_id} value={c.cam_id}>{c.cam_id}</option>)}
                </select>
              </label>
              <div className="flex gap-2 items-end">
                <label className="block space-y-1">
                  <span className="block text-[12px] text-ink-3">Measure as</span>
                  <select value={focalMode} onChange={(e) => setFocalMode(e.target.value as "fov" | "fx")} className="field">
                    <option value="fov">Field of view (&deg;)</option><option value="fx">Focal length (px)</option>
                  </select>
                </label>
                <label className="block flex-1 space-y-1">
                  <span className="block text-[12px] text-ink-3">Value</span>
                  <input value={focalVal} onChange={(e) => setFocalVal(e.target.value)} className="field w-full" />
                </label>
              </div>
              <div className="flex gap-2">
                <label className="flex-1 block space-y-1"><span className="block text-[12px] text-ink-3">Height (m)</span><input value={heightVal} onChange={(e) => setHeightVal(e.target.value)} className="field w-full" /></label>
                <label className="flex-1 block space-y-1"><span className="block text-[12px] text-ink-3">Pitch (&deg;)</span><input value={pitchVal} onChange={(e) => setPitchVal(e.target.value)} className="field w-full" /></label>
              </div>
              <button type="button" onClick={applySpec} disabled={!pick || !specCam} className="btn w-full">Save to session</button>
            </div>
            <div className="space-y-2 border-t hairline pt-3">
              <h2 className="section-title">Work it out automatically</h2>
              <button type="button" onClick={runEstimate} disabled={!pick} className="btn w-full">Estimate from road lines</button>
              <button type="button" onClick={runExtrinsics} disabled={!pick} className="btn w-full"
                data-tip="Checks that the cameras of a multi-camera rig agree about where things are">Check cameras against each other</button>
            </div>
            <div className="space-y-2 border-t hairline pt-3">
              <h2 className="section-title">Import a KITTI calibration file</h2>
              <label className="block space-y-1">
                <span className="block text-[12px] text-ink-3">Camera id</span>
                <input value={impCam} onChange={(e) => setImpCam(e.target.value)} placeholder="cam_id" className="field w-full" />
              </label>
              <label className="block space-y-1">
                <span className="block text-[12px] text-ink-3">Contents of calib.txt</span>
                <textarea value={impText} onChange={(e) => setImpText(e.target.value)} placeholder="P2: fx 0 cx ..." rows={3} className="field w-full resize-none font-mono text-[12px]" />
              </label>
              <button type="button" onClick={runImport} disabled={!pick || !impText} className="btn w-full"><T k="action.import">Import</T></button>
            </div>
            {ing && <div role="status" className="text-ink-2 text-[12px] border-t hairline pt-2">{ing}</div>}
          </div>
        </Inspector>
      </div>
    </PageShell>
  );
}
