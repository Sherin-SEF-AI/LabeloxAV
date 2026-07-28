"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import type { CameraZone, PersonIdentityRow, SecurityIncident } from "@/lib/types";

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

const WINDOWS: [string, number][] = [["24h", 24], ["3 days", 72], ["week", 168]];

function when(ns: number): string {
  return new Date(ns / 1e6).toISOString().slice(0, 19).replace("T", " ");
}

export default function IncidentsPage() {
  const [incidents, setIncidents] = useState<SecurityIncident[]>([]);
  const [zones, setZones] = useState<CameraZone[]>([]);
  const [identities, setIdentities] = useState<PersonIdentityRow[]>([]);
  const [hours, setHours] = useState(168);
  const [status, setStatus] = useState("open");
  const [busy, setBusy] = useState(false);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    try {
      const [inc, z, ids] = await Promise.all([
        api.secIncidents({ since_hours: hours, status: status || undefined, limit: 200 }),
        api.secZones(),
        api.secIdentities(2),
      ]);
      setIncidents(inc.incidents);
      setZones(z.zones);
      setIdentities(ids.identities);
      setDenied(false);
    } catch (e) {
      const msg = humanizeError(e);
      if (msg.toLowerCase().includes("authorised") || msg.includes("403")) setDenied(true);
      else toast(msg, "error");
    }
  }, [hours, status]);

  useEffect(() => { load(); }, [load]);

  const ack = (id: string, close: boolean) => async () => {
    setBusy(true);
    try { await api.acknowledgeIncident(id, close); await load(); }
    catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const forget = (id: string) => async () => {
    setBusy(true);
    try {
      await api.forgetIdentity(id);
      toast("signature erased", "success");
      await load();
    } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  return (
    <PageShell active="INCIDENTS" title="Incidents"
      subtitle="zone crossings, watchlist hits, and cross-camera sightings, stitched into events"
      filters={
        <>
          {WINDOWS.map(([label, h]) => (
            <button key={h} onClick={() => setHours(h)}
              className={`px-2 py-0.5 border ${
                hours === h ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
              {label}
            </button>
          ))}
          <span className="text-ink-4">|</span>
          {["open", "ack", "closed", ""].map((s) => (
            <button key={s || "all"} onClick={() => setStatus(s)}
              className={`px-2 py-0.5 border ${
                status === s ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
              {s || "all"}
            </button>
          ))}
        </>
      }>
      <div className="p-4 space-y-4 max-w-6xl">
        {denied ? (
          <div className="panel border border-block p-3 space-y-1">
            <div className="font-mono text-[11px] text-block">
              scene analytics is not authorised under this pack
            </div>
            <div className="font-mono text-[10.5px] text-ink-3">
              Drawing a tripwire on a camera and following a person between cameras are security-domain
              capabilities. Under the AV pack they are exactly what the privacy plane exists to prevent, so
              the server refuses rather than this page showing an empty list.
            </div>
          </div>
        ) : (
          <>
            <section className="panel">
              <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                incidents ({incidents.length})
              </div>
              {incidents.length === 0 ? (
                <div className="p-4 font-mono text-[11px] text-ink-3">
                  Nothing in this window. An incident is raised when a zone rule fires, a watched plate is
                  read, or the same person is seen on a second camera.
                </div>
              ) : (
                <table className="w-full font-mono text-[11px]">
                  <thead>
                    <tr className="text-ink-3 text-left border-b hairline">
                      <th className="py-1">when</th><th>kind</th><th>what</th>
                      <th>camera</th><th>plate</th><th>for</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {incidents.map((i) => (
                      <tr key={i.incident_id}
                        className={`border-b hairline ${i.severity === "critical" ? "bg-block/10" : ""}`}>
                        <td className="py-1 text-ink-3">{when(i.start_ts_ns)}</td>
                        <td className={SEV_TONE[i.severity] ?? "text-ink-2"}>
                          {i.kind.replace(/_/g, " ")}
                        </td>
                        <td className="text-ink">
                          {i.title}
                          {Number(i.evidence?.events ?? 1) > 1 && (
                            // Folded, not repeated: a van sitting across a tripwire is one incident that
                            // grows rather than one per frame.
                            <span className="text-ink-4"> ×{String(i.evidence.events)}</span>
                          )}
                        </td>
                        <td className="text-ink-3">{i.camera_id ?? "-"}</td>
                        <td className={i.plate ? "text-warn" : "text-ink-4"}>{i.plate ?? "-"}</td>
                        <td className="text-ink-3 tabular-nums">{i.duration_s}s</td>
                        <td className="text-right space-x-2">
                          {i.status === "open" && (
                            <button onClick={ack(i.incident_id, false)} disabled={busy}
                              className="text-ink-3 hover:text-accent disabled:opacity-40">ack</button>
                          )}
                          {i.status !== "closed" && (
                            <button onClick={ack(i.incident_id, true)} disabled={busy}
                              className="text-ink-3 hover:text-pass disabled:opacity-40">close</button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>

            <section className="panel">
              <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                zones ({zones.length})
              </div>
              <div className="p-3">
                {zones.length === 0 ? (
                  <div className="font-mono text-[11px] text-ink-3">
                    No zones. A fixed camera&apos;s frame does not move, which makes a polygon on it a
                    permanent statement about the world: this quadrilateral is the loading bay, this
                    segment is the gate line.
                  </div>
                ) : (
                  <table className="w-full font-mono text-[11px]">
                    <thead>
                      <tr className="text-ink-3 text-left border-b hairline">
                        <th className="py-1">name</th><th>camera</th><th>kind</th>
                        <th>rule</th><th>classes</th><th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {zones.map((z) => (
                        <tr key={z.zone_id} className="border-b hairline">
                          <td className="py-1 text-ink">{z.name}</td>
                          <td className="text-ink-3">{z.camera_id}</td>
                          <td className="text-ink-3">{z.kind}</td>
                          <td className={SEV_TONE[z.severity] ?? "text-ink-2"}>
                            {z.rule}{z.dwell_seconds ? ` ${z.dwell_seconds}s` : ""}
                          </td>
                          <td className="text-ink-3">
                            {z.classes.length ? z.classes.join(", ") : "every class"}
                          </td>
                          <td className="text-right">
                            <button onClick={async () => {
                              try { await api.deleteSecZone(z.zone_id); await load(); }
                              catch (e) { toast(humanizeError(e), "error"); }
                            }} className="text-ink-3 hover:text-block">deactivate</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </section>

            <section className="panel">
              <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                cross-camera sightings ({identities.length})
              </div>
              <div className="p-3 space-y-2">
                <div className="font-mono text-[10px] text-ink-3">
                  Appearance signatures, never names. This can say the same person appeared at two gates
                  and can never say who they are: there is nowhere in the record to put a name.
                </div>
                {identities.length === 0 ? (
                  <div className="font-mono text-[11px] text-ink-3">
                    Nobody has been seen on more than one camera.
                  </div>
                ) : (
                  <table className="w-full font-mono text-[11px]">
                    <thead>
                      <tr className="text-ink-3 text-left border-b hairline">
                        <th className="py-1">signature</th><th>cameras</th><th>tracks</th><th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {identities.map((p) => (
                        <tr key={p.identity_id} className="border-b hairline">
                          <td className="py-1 text-ink-2">{p.identity_id.slice(0, 8)}</td>
                          <td className="text-ink-3">{p.cameras.join(", ")}</td>
                          <td className="text-ink-3 tabular-nums">{p.n_tracks}</td>
                          <td className="text-right">
                            {/* Present because it must be: a signature is derived from a person's
                                appearance, so it is personal data whether or not a name is attached. */}
                            <button onClick={forget(p.identity_id)} disabled={busy}
                              className="text-ink-3 hover:text-block disabled:opacity-40">erase</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </PageShell>
  );
}
