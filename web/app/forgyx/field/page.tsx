"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import type { EdgeDeviceRow, EdgeFieldReport, EdgeFleet } from "@/lib/types";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Stat } from "@/components/ui/kit";
import { ago, humanize, num, plural, shortId } from "@/lib/format";

// What the field says about what the bench passed.
//
// Edge deployment gates artifacts on latency and accuracy measured on a bench. A bench does not thermally throttle
// after twenty minutes in a parked vehicle, does not share its GPU with a video encoder, and does not see
// the input distribution the field sees. So artifacts have been passing on figures that are true in a room
// nobody deploys in, and the first anyone learns of the gap is a device dropping frames.
//
// The comparison is the product here. Either number alone is uninteresting; the ratio between them is the
// finding, so it is what the table leads with.

const WINDOWS: [string, number][] = [["24 hours", 24], ["Week", 168], ["Month", 720]];

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

  const verdictState = report?.verdict === "pass" ? "pass" : report?.verdict === "field_regression" ? "block" : "muted";
  return (
    <PageShell active="Edge deployment" title="Field performance"
      subtitle="What deployed devices report, next to what the bench measured."
      filters={
        <>
          <span className="text-ink-3">Period</span>
          {WINDOWS.map(([label, h]) => (
            <button key={h} type="button" onClick={() => setHours(h)} aria-pressed={hours === h}
              className={`chip ${hours === h ? "chip-on" : ""}`}>
              {label}
            </button>
          ))}
        </>
      }>
      <div className="page-wrap space-y-4">
        {fleet ? (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <Stat label="Devices" value={num(fleet.devices)} />
            {/* Surfaced, because a fleet whose devices have gone quiet looks identical to a healthy one
                in every average computed over the devices still talking. */}
            <Stat label="Silent devices" value={num(fleet.silent_devices)} tone={fleet.silent_devices ? "warn" : undefined}
              sub="have stopped reporting" />
            <Stat label="Models in the field" value={num(fleet.artifacts.length)} />
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[0, 1, 2].map((i) => <div key={i} className="skeleton h-[74px]" />)}
          </div>
        )}

        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Models in the field</h2>
            {fleet && fleet.artifacts.length > 0 && (
              <span className="ml-auto text-[12px] text-ink-3">Select a model to compare its field and bench figures.</span>
            )}
          </div>
          {!fleet?.artifacts.length ? (
            <EmptyState compact icon="server" title="No field reports yet">
              Devices running a deployed model report their latency and temperature every few minutes. Their
              reports, and how they compare with the bench, appear here.
            </EmptyState>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model package</th><th className="num">Devices</th><th className="num">Reports</th>
                  <th className="num">Field p95</th><th className="num">Field vs bench</th><th>Findings</th>
                </tr>
              </thead>
              <tbody>
                {fleet.artifacts.map((a) => (
                  <tr key={a.artifact_id} className={open === a.artifact_id ? "bg-head" : ""}>
                    <td>
                      <button type="button" onClick={() => setOpen(open === a.artifact_id ? null : a.artifact_id)}
                        aria-expanded={open === a.artifact_id}
                        className="link font-mono text-[12px]">{a.artifact_id.slice(0, 14)}</button>
                    </td>
                    <td className="num">
                      {num(a.devices)}
                      {!a.fleet_significant && <span className="text-warn" data-tip="Too few devices to judge the fleet"> (too few)</span>}
                    </td>
                    <td className="num">{num(a.windows)}</td>
                    <td className="num">{a.latency_p95_ms != null ? `${a.latency_p95_ms} ms` : "-"}</td>
                    <td className={`num ${ratioTone(a.latency_ratio)}`}>
                      {a.latency_ratio != null ? `${a.latency_ratio}x slower` : "-"}
                    </td>
                    <td className={a.findings ? "text-warn" : "text-pass"}>
                      {a.findings ? plural(a.findings, "finding") : "None"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {report && (
          <section className="panel">
            <div className="panel-head">
              <h2 className="section-title font-mono">{report.artifact_id.slice(0, 20)}</h2>
              <Badge state={verdictState}>{report.verdict ? humanize(report.verdict) : "No verdict"}</Badge>
              {/* Advisory, and it says so. Telemetry comes from devices, which are outside the trust
                  boundary, so a single misconfigured unit must not be able to demote a champion. */}
              <span className="ml-auto text-[12px] text-ink-3">Advisory only: a device cannot demote a model</span>
            </div>
            <div className="p-3.5 space-y-3">
              {report.detail && <div className="text-[12.5px] text-ink-2">{report.detail}</div>}
              <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
                {([["Field p50", report.field.latency_p50_ms, " ms"],
                  ["Field p95", report.field.latency_p95_ms, " ms"],
                  ["Bench p95", report.bench?.latency_p95_ms as number | undefined, " ms"],
                  ["Worst throttling", report.field.worst_throttled_fraction, ""],
                  ["Hottest", report.field.temp_c_max, " °C"],
                  ["Confidence drift", report.confidence_drift, ""]] as [string, number | null | undefined, string][]).map(([label, value, unit]) => (
                    <Stat key={label} label={label} value={value == null ? "-" : `${value}${unit}`} />
                  ))}
              </div>
              {report.findings.length > 0 && (
                <ul className="space-y-1">
                  {report.findings.map((f, i) => (
                    <li key={i} className="text-[12.5px] text-warn">
                      {humanize(f.kind)}: <span className="text-ink-2">{f.detail}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Devices</h2>
            <span className="text-[12px] text-ink-3">{num(devices.length)}</span>
          </div>
          {devices.length === 0 ? (
            <EmptyState compact icon="chip" title="No devices registered">
              A vehicle computer registers itself the first time it reports. Until then there is nothing to
              compare against the bench.
            </EmptyState>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Device</th><th>Hardware</th><th>Runtime</th>
                  <th>Model</th><th>Fleet</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {devices.map((d) => (
                  <tr key={d.device_id}>
                    <td className="text-ink">{d.name || <span className="font-mono text-[12px]">{shortId(d.device_id)}</span>}</td>
                    <td>{d.hardware ?? "-"}</td>
                    <td>{d.runtime ?? "-"}</td>
                    <td>{d.model_version ?? (d.artifact_id ? <span className="font-mono text-[12px]">{d.artifact_id.slice(0, 12)}</span> : "-")}</td>
                    <td>{d.fleet ?? "-"}</td>
                    <td>
                      <Badge state={d.live ? "pass" : "warn"}>{d.live ? "Live" : "Silent"}</Badge>
                      {d.last_seen_at && <span className="ml-2 text-[12px] text-ink-3">{ago(d.last_seen_at)}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </PageShell>
  );
}
