"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import type { EdgeDeviceRow, EdgeFieldReport, EdgeFleet } from "@/lib/types";

// What the field says about what the bench passed.
//
// FORGYX gates artifacts on latency and accuracy measured on a bench. A bench does not thermally throttle
// after twenty minutes in a parked vehicle, does not share its GPU with a video encoder, and does not see
// the input distribution the field sees. So artifacts have been passing on figures that are true in a room
// nobody deploys in, and the first anyone learns of the gap is a device dropping frames.
//
// The comparison is the product here. Either number alone is uninteresting; the ratio between them is the
// finding, so it is what the table leads with.

const WINDOWS: [string, number][] = [["24h", 24], ["week", 168], ["month", 720]];

function ratioTone(r: number | null): string {
  if (r == null) return "text-ink-3";
  if (r > 1.35) return "text-block";
  if (r > 1.15) return "text-warn";
  return "text-pass";
}

export default function FieldPage() {
  const [fleet, setFleet] = useState<EdgeFleet | null>(null);
  const [devices, setDevices] = useState<EdgeDeviceRow[]>([]);
  const [hours, setHours] = useState(168);
  const [open, setOpen] = useState<string | null>(null);
  const [report, setReport] = useState<(EdgeFieldReport & { verdict?: string; detail?: string }) | null>(null);

  const load = useCallback(async () => {
    try {
      const [f, d] = await Promise.all([api.edgeFleet(hours), api.edgeDevices()]);
      setFleet(f);
      setDevices(d.devices);
    } catch (e) { toast(humanizeError(e), "error"); }
  }, [hours]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (!open) { setReport(null); return; }
    api.edgeFieldGate(open, hours).then(setReport).catch((e) => toast(humanizeError(e), "error"));
  }, [open, hours]);

  return (
    <PageShell active="FIELD" title="Field performance"
      subtitle="what deployed devices report, next to what the bench measured"
      filters={
        <>
          {WINDOWS.map(([label, h]) => (
            <button key={h} onClick={() => setHours(h)}
              className={`px-2 py-0.5 border ${
                hours === h ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
              {label}
            </button>
          ))}
        </>
      }>
      <div className="p-4 space-y-4 max-w-6xl">
        {fleet && (
          <div className="flex gap-2 flex-wrap">
            <div className="panel px-3 py-2 min-w-[110px]">
              <div className="font-mono text-[10px] uppercase text-ink-3">devices</div>
              <div className="font-mono text-[18px] text-ink tabular-nums">{fleet.devices}</div>
            </div>
            <div className="panel px-3 py-2 min-w-[110px]">
              <div className="font-mono text-[10px] uppercase text-ink-3">silent</div>
              {/* Surfaced, because a fleet whose devices have gone quiet looks identical to a healthy one
                  in every average computed over the devices still talking. */}
              <div className={`font-mono text-[18px] tabular-nums ${
                fleet.silent_devices ? "text-warn" : "text-ink"}`}>{fleet.silent_devices}</div>
            </div>
            <div className="panel px-3 py-2 min-w-[110px]">
              <div className="font-mono text-[10px] uppercase text-ink-3">artifacts</div>
              <div className="font-mono text-[18px] text-ink tabular-nums">{fleet.artifacts.length}</div>
            </div>
          </div>
        )}

        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            artifacts in the field
          </div>
          {!fleet?.artifacts.length ? (
            <div className="p-4 font-mono text-[11px] text-ink-3">
              No telemetry yet. A device posts a reporting window rather than every inference: p50, p95 and
              the thermal ceiling reached are properties of a window, and a device posting each inference
              would spend its uplink on telemetry.
            </div>
          ) : (
            <table className="w-full font-mono text-[11px]">
              <thead>
                <tr className="text-ink-3 text-left border-b hairline">
                  <th className="py-1">artifact</th><th>devices</th><th>windows</th>
                  <th>field p95</th><th>vs bench</th><th>findings</th>
                </tr>
              </thead>
              <tbody>
                {fleet.artifacts.map((a) => (
                  <tr key={a.artifact_id} className="border-b hairline">
                    <td className="py-1">
                      <button onClick={() => setOpen(open === a.artifact_id ? null : a.artifact_id)}
                        className="text-ink hover:text-accent">{a.artifact_id.slice(0, 14)}</button>
                    </td>
                    <td className="text-ink-3 tabular-nums">
                      {a.devices}
                      {!a.fleet_significant && <span className="text-warn"> (too few)</span>}
                    </td>
                    <td className="text-ink-3 tabular-nums">{a.windows}</td>
                    <td className="text-ink-2 tabular-nums">
                      {a.latency_p95_ms != null ? `${a.latency_p95_ms}ms` : "-"}
                    </td>
                    <td className={`tabular-nums ${ratioTone(a.latency_ratio)}`}>
                      {a.latency_ratio != null ? `${a.latency_ratio}x` : "-"}
                    </td>
                    <td className={a.findings ? "text-warn" : "text-pass"}>
                      {a.findings || "none"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {report && (
          <section className="panel">
            <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3
                            border-b hairline px-3 py-2">
              <span>{report.artifact_id.slice(0, 20)}</span>
              <span className={`normal-case ${
                report.verdict === "pass" ? "text-pass"
                  : report.verdict === "field_regression" ? "text-block" : "text-ink-3"}`}>
                {report.verdict}
              </span>
              {/* Advisory, and it says so. Telemetry comes from devices, which are outside the trust
                  boundary, so a single misconfigured unit must not be able to demote a champion. */}
              <span className="ml-auto normal-case text-ink-3">advisory: a device cannot demote</span>
            </div>
            <div className="p-3 space-y-2">
              <div className="font-mono text-[11px] text-ink-2">{report.detail}</div>
              <div className="flex gap-2 flex-wrap">
                {[["field p50", report.field.latency_p50_ms, "ms"],
                  ["field p95", report.field.latency_p95_ms, "ms"],
                  ["bench p95", report.bench?.latency_p95_ms as number | undefined, "ms"],
                  ["worst throttle", report.field.worst_throttled_fraction, ""],
                  ["max temp", report.field.temp_c_max, "C"],
                  ["drift", report.confidence_drift, ""]].map(([label, value, unit]) => (
                    <div key={String(label)} className="panel px-3 py-2 min-w-[110px]">
                      <div className="font-mono text-[10px] uppercase text-ink-3">{String(label)}</div>
                      <div className="font-mono text-[15px] text-ink tabular-nums">
                        {value == null ? "-" : `${value}${unit}`}
                      </div>
                    </div>
                  ))}
              </div>
              {report.findings.length > 0 && (
                <ul className="space-y-0.5">
                  {report.findings.map((f, i) => (
                    <li key={i} className="font-mono text-[11px] text-warn">
                      {f.kind.replace(/_/g, " ")}: <span className="text-ink-2">{f.detail}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            devices ({devices.length})
          </div>
          <div className="p-3">
            {devices.length === 0 ? (
              <div className="font-mono text-[11px] text-ink-3">No devices registered.</div>
            ) : (
              <table className="w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">device</th><th>hardware</th><th>runtime</th>
                    <th>artifact</th><th>fleet</th><th>state</th>
                  </tr>
                </thead>
                <tbody>
                  {devices.map((d) => (
                    <tr key={d.device_id} className="border-b hairline">
                      <td className="py-1 text-ink-2">{d.name || d.device_id}</td>
                      <td className="text-ink-3">{d.hardware ?? "-"}</td>
                      <td className="text-ink-3">{d.runtime ?? "-"}</td>
                      <td className="text-ink-3">{d.artifact_id?.slice(0, 12) ?? "-"}</td>
                      <td className="text-ink-3">{d.fleet ?? "-"}</td>
                      <td className={d.live ? "text-pass" : "text-warn"}>
                        {d.live ? "live" : "silent"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      </div>
    </PageShell>
  );
}
