"use client";

import PackGate from "@/components/shell/PackGate";
import { T } from "@/lib/i18n";
import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
import type { CameraZone, PersonIdentityRow, SecurityIncident } from "@/lib/types";
import EmptyState from "@/components/ui/EmptyState";
import { duration, fromNs, humanize, num, shortId } from "@/lib/format";

// LabeloxSec v2: zones, incidents, and cross-camera identity.
//
// A plate read, a zone crossing and a person track were three unrelated rows about the same van arriving
// at the same gate at the same moment, and an operator assembled the event in their head from three
// screens. An incident is that assembly, made once and kept.
//
// The identity panel deliberately shows no names and no faces. A signature links tracks to each other and
// can never say who anybody is; that boundary is the difference between re-identification for an
// authorised security deployment and building a face database, and it is enforced by there being nowhere
// to put a name.

const SEV_TONE: Record<string, string> = {
  info: "text-ink-2", warn: "text-warn", critical: "text-block",
};

const WINDOWS: [string, number][] = [["24 hours", 24], ["3 days", 72], ["Week", 168]];
const STATUSES: [string, string][] = [["open", "Open"], ["ack", "Acknowledged"], ["closed", "Closed"], ["", "All"]];

const NO_INCIDENTS: SecurityIncident[] = [];
const NO_ZONES: CameraZone[] = [];
const NO_IDENTITIES: PersonIdentityRow[] = [];

function IncidentsPage() {
  const [hours, setHours] = useState(168);
  const [status, setStatus] = useState("open");
  const [busy, setBusy] = useState(false);
  const [denied, setDenied] = useState(false);

  // keepPrevious: changing a filter leaves the last incidents up until the new ones arrive.
  const incArgs = { since_hours: hours, status: status || undefined, limit: 200 };
  const incidentsQ = useQuery(["secIncidents", incArgs], () => api.secIncidents(incArgs), { keepPrevious: true });
  const zonesQ = useQuery(["secZones"], () => api.secZones());
  const identitiesQ = useQuery(["secIdentities", 2], () => api.secIdentities(2));
  const incidents: SecurityIncident[] = incidentsQ.data?.incidents ?? NO_INCIDENTS;
  const zones: CameraZone[] = zonesQ.data?.zones ?? NO_ZONES;
  const identities: PersonIdentityRow[] = identitiesQ.data?.identities ?? NO_IDENTITIES;

  // A refusal is a state of the page, anything else is announced. `denied` clears only when the reads
  // succeed, so it does not flicker off while a changed filter is being refused again.
  const failure = incidentsQ.error ?? zonesQ.error ?? identitiesQ.error;
  useEffect(() => {
    if (!failure) return;
    const msg = humanizeError(failure);
    if (msg.toLowerCase().includes("authorised") || msg.includes("403")) setDenied(true);
    else toast(msg, "error");
  }, [failure]);
  const settled = !failure && !incidentsQ.validating && !zonesQ.validating && !identitiesQ.validating
    && Boolean(incidentsQ.data) && Boolean(zonesQ.data) && Boolean(identitiesQ.data);
  useEffect(() => { if (settled) setDenied(false); }, [settled]);

  const load = useCallback(() => invalidate("secIncidents", "secZones", "secIdentities"), []);

  const ack = (id: string, close: boolean) => async () => {
    setBusy(true);
    try { await api.acknowledgeIncident(id, close); await load(); }
    catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const forget = (id: string) => async () => {
    setBusy(true);
    try {
      await api.forgetIdentity(id);
      toast("Signature erased", "success");
      await load();
    } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  return (
    <PageShell active="Incidents" title="Incidents"
      subtitle="Zone crossings, watchlist hits and cross-camera sightings, grouped into events."
      filters={
        <>
          <span className="text-ink-3">Period</span>
          {WINDOWS.map(([label, h]) => (
            <button key={h} type="button" onClick={() => setHours(h)} aria-pressed={hours === h}
              className={`chip ${hours === h ? "chip-on" : ""}`}>
              {label}
            </button>
          ))}
          <span className="text-ink-3 ml-3">Status</span>
          {STATUSES.map(([s, label]) => (
            <button key={s || "all"} type="button" onClick={() => setStatus(s)} aria-pressed={status === s}
              className={`chip ${status === s ? "chip-on" : ""}`}>
              {label}
            </button>
          ))}
        </>
      }>
      <div className="page-wrap space-y-4">
        {denied ? (
          <div role="alert" className="panel border-block/40 bg-block/5 p-4 space-y-1.5">
            <div className="text-[13px] font-medium text-block">
              Scene analytics is not allowed on this deployment
            </div>
            <p className="text-[12.5px] text-ink-2 max-w-[75ch]">
              Drawing a tripwire on a camera and following a person between cameras are only for authorised
              security deployments. Here they are exactly what the privacy rules prevent, so the server refuses
              them rather than this page showing an empty list.
            </p>
          </div>
        ) : (
          <>
            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Incidents</h2>
                <span className="text-[12px] text-ink-3 tabular-nums">{num(incidents.length)}</span>
              </div>
              {incidents.length === 0 ? (
                <EmptyState compact icon="flag" title={incidentsQ.loading ? "Loading incidents..." : "No incidents in this period"}>
                  An incident is raised when a zone rule fires, a watched plate is read, or the same person is seen
                  on a second camera.
                </EmptyState>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>When</th><th>Kind</th><th>What happened</th>
                      <th>Camera</th><th>Plate</th><th className="num">Lasted</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {incidents.map((i) => (
                      <tr key={i.incident_id} className={i.severity === "critical" ? "bg-block/10" : ""}>
                        <td className="whitespace-nowrap">{fromNs(i.start_ts_ns)}</td>
                        <td className={SEV_TONE[i.severity] ?? "text-ink-2"}>{humanize(i.kind)}</td>
                        <td className="text-ink">
                          {i.title}
                          {Number(i.evidence?.events ?? 1) > 1 && (
                            // Folded, not repeated: a van sitting across a tripwire is one incident that
                            // grows rather than one per frame.
                            <span className="text-ink-3"> ×{String(i.evidence.events)}</span>
                          )}
                        </td>
                        <td>{i.camera_id ?? "-"}</td>
                        <td className={`font-mono ${i.plate ? "text-warn" : ""}`}>{i.plate ?? "-"}</td>
                        <td className="num">{duration(i.duration_s)}</td>
                        <td className="text-right whitespace-nowrap">
                          <span className="inline-flex gap-2">
                            {i.status === "open" && (
                              <button type="button" onClick={ack(i.incident_id, false)} disabled={busy} className="btn">
                                Acknowledge
                              </button>
                            )}
                            {i.status !== "closed" && (
                              <button type="button" onClick={ack(i.incident_id, true)} disabled={busy} className="btn">
                                <T k="action.close">Close</T>
                              </button>
                            )}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>

            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Zones</h2>
                <span className="text-[12px] text-ink-3 tabular-nums">{num(zones.length)}</span>
              </div>
              {zones.length === 0 ? (
                <EmptyState compact icon="polygon" title="No zones">
                  A zone is an area or a line drawn on a fixed camera, such as a loading bay or a gate. Crossing
                  or lingering in one raises an incident.
                </EmptyState>
              ) : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Name</th><th>Camera</th><th>Kind</th>
                      <th>Rule</th><th>Classes</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {zones.map((z) => (
                      <tr key={z.zone_id}>
                        <td className="text-ink">{z.name}</td>
                        <td>{z.camera_id}</td>
                        <td>{humanize(z.kind)}</td>
                        <td className={SEV_TONE[z.severity] ?? "text-ink-2"}>
                          {humanize(z.rule)}{z.dwell_seconds ? ` after ${duration(z.dwell_seconds)}` : ""}
                        </td>
                        <td>{z.classes.length ? z.classes.map(humanize).join(", ") : "Every class"}</td>
                        <td className="text-right">
                          <button type="button" onClick={async () => {
                            try { await api.deleteSecZone(z.zone_id); await load(); }
                            catch (e) { toast(humanizeError(e), "error"); }
                          }} className="btn btn-danger">Deactivate</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>

            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Seen on more than one camera</h2>
                <span className="text-[12px] text-ink-3 tabular-nums">{num(identities.length)}</span>
              </div>
              <p className="section-desc px-3.5 pt-3">
                Appearance signatures, never names. This can say the same person appeared at two gates, and can
                never say who they are: there is nowhere in the record to put a name.
              </p>
              {identities.length === 0 ? (
                <EmptyState compact icon="person" title="Nobody has been seen on a second camera" />
              ) : (
                <table className="data-table mt-2">
                  <thead>
                    <tr><th>Signature</th><th>Cameras</th><th className="num">Tracks</th><th></th></tr>
                  </thead>
                  <tbody>
                    {identities.map((p) => (
                      <tr key={p.identity_id}>
                        <td className="font-mono text-[12px]">{shortId(p.identity_id)}</td>
                        <td>{p.cameras.join(", ")}</td>
                        <td className="num">{num(p.n_tracks)}</td>
                        <td className="text-right">
                          {/* Present because it must be: a signature is derived from a person's
                              appearance, so it is personal data whether or not a name is attached. */}
                          <button type="button" onClick={forget(p.identity_id)} disabled={busy} className="btn btn-danger"
                            data-tip="Deletes this appearance signature; it is personal data">Erase</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
          </>
        )}
      </div>
    </PageShell>
  );
}

// LabeloxSec is an optional package (settings.packs.optional on the server); without it these routes do not exist.
export default function IncidentsPageRoute() {
  return (
    <PackGate pack="sec" title="Incidents"
      what="LabeloxSec, for authorised security deployments: number-plate reading, watchlists, zones and following a person across cameras">
      <IncidentsPage />
    </PackGate>
  );
}
