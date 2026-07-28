"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";
import type { SecPack, SecSession, SecStats, WatchlistEntry, PlateReadRow } from "@/lib/types";

// The LabeloxSec console: the security domain's own surface.
//
// The Sec pack shipped with an ontology, a static-camera scene model, a tested India plate-format kernel and a
// recogniser, and nothing outside the Python package could reach any of it. This page is that reach.
//
// The pack banner is deliberately the first thing rendered and the gate on everything below it. Reading a
// registration mark is lawful for an authorised security deployment and is exactly what the AV pack must never
// do, because under DPDPA a plate is personal data the privacy plane blurs. A deployment running the AV pack is
// told that in one sentence instead of being handed a console whose every button 403s.

const SEVERITIES = ["info", "warn", "critical"] as const;

// Severity earns colour because it is live operational state, not decoration. Everything else stays graphite.
const SEV_CLASS: Record<string, string> = {
  info: "text-ink-2",
  warn: "text-warn",
  critical: "text-block",
};

function Section({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        <span>{title}</span>
        {right && <span className="ml-auto normal-case">{right}</span>}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: number | string; hint?: string }) {
  return (
    <div className="panel px-3 py-2 min-w-[120px]">
      <div className="font-mono text-[10px] uppercase text-ink-3">{label}</div>
      <div className="font-mono text-[18px] text-ink tabular-nums">{value}</div>
      {hint && <div className="font-mono text-[10px] text-ink-3">{hint}</div>}
    </div>
  );
}

export default function LabeloxSecPage() {
  // useSearchParams forces a client-side-render bailout, which Next requires be under a Suspense boundary or
  // the static prerender of this route fails the build. Match the pattern used by integrations and search.
  return <Suspense fallback={null}><LabeloxSecBody /></Suspense>;
}

function LabeloxSecBody() {
  const params = useSearchParams();
  const [pack, setPack] = useState<SecPack | null>(null);
  const [packErr, setPackErr] = useState<string | null>(null);
  const [stats, setStats] = useState<SecStats | null>(null);
  const [reads, setReads] = useState<PlateReadRow[]>([]);
  const [readTotal, setReadTotal] = useState(0);
  const [watchlist, setWatchlist] = useState<WatchlistEntry[]>([]);
  const [sessions, setSessions] = useState<SecSession[]>([]);
  const [busy, setBusy] = useState(false);
  const confirm = useConfirm();

  // read filters
  const [plateFilter, setPlateFilter] = useState("");
  const [cameraFilter, setCameraFilter] = useState("");
  // The menu's "Watchlist hits" entry lands here with the filter already applied, so it is a destination
  // rather than a link that drops you on an unfiltered feed and expects you to tick a box.
  const [hitsOnly, setHitsOnly] = useState(params.get("hits") === "1");
  const [stateFilter, setStateFilter] = useState("");

  // watchlist form
  const [newPlate, setNewPlate] = useState("");
  const [newReason, setNewReason] = useState("");
  const [newSeverity, setNewSeverity] = useState<string>("warn");

  const loadPack = useCallback(async () => {
    try {
      setPack(await api.secPack("sec"));
      setPackErr(null);
    } catch (e) {
      setPackErr(humanizeError(e));
    }
  }, []);

  const loadReads = useCallback(async () => {
    try {
      const r = await api.secReads({
        plate: plateFilter.trim() || undefined,
        camera_id: cameraFilter.trim() || undefined,
        state_code: stateFilter || undefined,
        hits_only: hitsOnly || undefined,
        limit: 200,
      });
      setReads(r.reads);
      setReadTotal(r.total);
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, [plateFilter, cameraFilter, stateFilter, hitsOnly]);

  const loadRest = useCallback(async () => {
    try {
      const [s, w, ss] = await Promise.all([api.secStats(), api.secWatchlist(true), api.secSessions(50)]);
      setStats(s);
      setWatchlist(w.entries);
      setSessions(ss.sessions);
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, []);

  useEffect(() => { loadPack(); loadRest(); }, [loadPack, loadRest]);
  useEffect(() => { loadReads(); }, [loadReads]);

  const addWatch = async () => {
    if (!newPlate.trim()) { toast("enter a registration mark", "error"); return; }
    setBusy(true);
    try {
      const e = await api.secAddWatch(newPlate.trim(), newReason.trim() || null, newSeverity);
      toast(`watching ${e.plate}`, "success");
      setNewPlate(""); setNewReason("");
      await Promise.all([loadRest(), loadReads()]);
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally { setBusy(false); }
  };

  const removeWatch = async (e: WatchlistEntry) => {
    if (!(await confirm({
      title: "Stop watching this plate?",
      body: `${e.plate} stays on record as deactivated, so a hit already recorded against it remains explainable.`,
      confirmLabel: "Stop watching",
    }))) return;
    try {
      await api.secRemoveWatch(e.entry_id);
      toast(`${e.plate} deactivated`, "success");
      await loadRest();
    } catch (err) { toast(humanizeError(err), "error"); }
  };

  // A session has no detail route of its own; the corpus opens one at its first frame, the same way the
  // annotations list does. Resolving it server side avoids a second list request just to find that frame.
  const openSession = async (sessionId: string) => {
    try {
      const { frame_id } = await api.firstFrame(sessionId);
      window.location.href = `/frame/${frame_id}`;
    } catch (e) { toast(humanizeError(e), "error"); }
  };

  // The whole console is gated on the pack, because a plate console under a pack that refuses ANPR is a lie.
  const authorised = pack?.anpr_authorised === true;

  return (
    <PageShell
      active="LABELOXSEC"
      title="LabeloxSec"
      subtitle="security domain: plate reads, watchlist, static-camera sessions"
      right={pack ? (
        <span className="font-mono text-[11px] text-ink-3">
          pack <span className={authorised ? "text-pass" : "text-block"}>{pack.pack_id}</span>
        </span>
      ) : null}
    >
      <div className="p-4 space-y-4 max-w-6xl">
        {/* the pack gate */}
        {packErr && (
          <div className="panel border border-block p-3 font-mono text-[11px] text-block">{packErr}</div>
        )}
        {pack && !authorised && (
          <div className="panel border border-block p-3 space-y-1">
            <div className="font-mono text-[11px] text-block">
              ANPR is not authorised under the {pack.pack_id} pack
            </div>
            <div className="font-mono text-[10.5px] text-ink-3">
              Reading a registration mark is a security-domain capability. Under the AV pack a plate is personal
              data that the privacy plane blurs and never reads, so every action below would be refused by the
              server. Packs available on this deployment: {pack.available_packs.join(", ")}.
            </div>
          </div>
        )}
        {pack && authorised && (
          <div className="panel p-3 flex items-start gap-4 flex-wrap">
            <div>
              <div className="font-mono text-[11px] text-ink">{pack.name}</div>
              <div className="font-mono text-[10px] text-ink-3">
                capabilities: {pack.capabilities.join(", ")}
              </div>
            </div>
            {pack.safety_classes.length > 0 && (
              <div className="ml-auto">
                <div className="font-mono text-[10px] uppercase text-ink-3">safety-critical classes</div>
                <div className="font-mono text-[10.5px] text-ink-2">{pack.safety_classes.join(", ")}</div>
              </div>
            )}
          </div>
        )}

        {stats && (
          <div className="flex gap-2 flex-wrap">
            <Stat label="reads" value={stats.reads} />
            <Stat label="watchlist hits" value={stats.watchlist_hits} />
            <Stat label="valid format" value={stats.valid_format}
              hint={stats.reads ? `${Math.round((stats.valid_format / stats.reads) * 100)}% of reads` : undefined} />
            <Stat label="watching" value={stats.watchlist_size} />
            {/* Surfaced rather than hidden: it tells the operator a confidence filter is meaningless on those
                rows, because the local reader exposes no calibrated score. */}
            <Stat label="unscored" value={stats.unscored_reads} hint="no OCR confidence measured" />
          </div>
        )}

        {stats && Object.keys(stats.top_states).length > 0 && (
          <Section title="reads by issuing state">
            <div className="flex gap-1 flex-wrap">
              {/* A facet, not a legend: clicking a state filters the feed below, and clicking it again clears
                  the filter. A count that looks clickable and is not is worse than no affordance at all. */}
              {Object.entries(stats.top_states).map(([code, n]) => (
                <button key={code}
                  onClick={() => setStateFilter((s) => (s === code ? "" : code))}
                  className={`border px-1.5 py-0.5 font-mono text-[10.5px] ${
                    stateFilter === code ? "border-accent text-ink" : "border-line text-ink-2 hover:text-ink"}`}>
                  {code} <span className="text-ink-3">{n}</span>
                </button>
              ))}
              {stateFilter && (
                <button onClick={() => setStateFilter("")}
                  className="px-1.5 py-0.5 font-mono text-[10.5px] text-ink-3 hover:text-ink">clear</button>
              )}
            </div>
          </Section>
        )}

        <Section title={`watchlist (${watchlist.length} active)`}>
          <div className="space-y-2">
            <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
              <input value={newPlate} onChange={(e) => setNewPlate(e.target.value)}
                placeholder="KA 01 AB 1234"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-40" />
              <input value={newReason} onChange={(e) => setNewReason(e.target.value)}
                placeholder="reason (optional)"
                className="flex-1 min-w-[200px] bg-bg border border-line px-1.5 py-0.5 text-ink" />
              <select value={newSeverity} onChange={(e) => setNewSeverity(e.target.value)}
                className="bg-bg border border-line px-1.5 py-0.5 text-ink">
                {SEVERITIES.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
              <button onClick={addWatch} disabled={busy || !authorised}
                className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
                watch
              </button>
            </div>
            <div className="font-mono text-[10px] text-ink-3">
              stored on the normalised mark, so KA 01 AB 1234, ka-01-ab-1234 and KA01AB1234 are one entry
            </div>

            {watchlist.length > 0 && (
              <table className="w-full font-mono text-[11px] mt-2">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">plate</th><th>severity</th><th>reason</th><th>added by</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {watchlist.map((w) => (
                    <tr key={w.entry_id} className="border-b hairline">
                      <td className="py-1 text-ink">{w.plate}</td>
                      <td className={SEV_CLASS[w.severity] ?? "text-ink-2"}>{w.severity}</td>
                      <td className="text-ink-3 truncate max-w-[320px]">{w.reason ?? "-"}</td>
                      <td className="text-ink-3">{w.added_by ?? "-"}</td>
                      <td className="text-right">
                        <button onClick={() => removeWatch(w)} className="text-ink-3 hover:text-block">
                          stop watching
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Section>

        <Section title={`plate reads (${reads.length} of ${readTotal})`}
          right={
            <span className="flex items-center gap-2 font-mono text-[11px]">
              <input value={plateFilter} onChange={(e) => setPlateFilter(e.target.value)}
                placeholder="plate"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-32" />
              <input value={cameraFilter} onChange={(e) => setCameraFilter(e.target.value)}
                placeholder="camera"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-28" />
              <label className="flex items-center gap-1 text-ink-3">
                <input type="checkbox" checked={hitsOnly} onChange={(e) => setHitsOnly(e.target.checked)} />
                hits only
              </label>
            </span>
          }>
          {reads.length === 0 ? (
            <div className="font-mono text-[11px] text-ink-3">
              no reads recorded. A read is produced by POST /api/security/recognize over a frame and its plate
              regions, and every one is stored against its session so an erasure request takes the plate text
              with it.
            </div>
          ) : (
            <table className="w-full font-mono text-[11px]">
              <thead>
                <tr className="text-ink-3 text-left border-b hairline">
                  <th className="py-1">plate</th><th>type</th><th>state</th><th>camera</th>
                  <th>det</th><th>ocr</th><th>format</th><th>when</th>
                </tr>
              </thead>
              <tbody>
                {reads.map((r) => (
                  <tr key={r.read_id}
                    className={`border-b hairline ${r.watchlist_hit ? "bg-block/10" : ""}`}>
                    <td className={`py-1 ${r.watchlist_hit ? SEV_CLASS[r.watchlist_severity ?? "warn"] : "text-ink"}`}>
                      {r.plate || r.plate_raw}
                      {r.watchlist_hit && (
                        <span className="ml-1 text-[10px] uppercase">
                          {r.watchlist_severity ?? "hit"}
                        </span>
                      )}
                    </td>
                    <td className="text-ink-3">{r.plate_type}</td>
                    <td className="text-ink-3">{r.state_code ?? "-"}</td>
                    <td className="text-ink-3">{r.camera_id ?? "-"}</td>
                    <td className="text-ink-3 tabular-nums">{r.det_conf.toFixed(2)}</td>
                    {/* A dash, not a zero: the reader measured nothing and inventing a number here would make
                        the column look sortable in a way it is not. */}
                    <td className="text-ink-3 tabular-nums">
                      {r.ocr_conf == null ? "-" : r.ocr_conf.toFixed(2)}
                    </td>
                    <td className={r.valid ? "text-pass tabular-nums" : "text-ink-3 tabular-nums"}>
                      {r.format_confidence.toFixed(2)}
                    </td>
                    <td className="text-ink-3">{r.created_at?.slice(0, 19).replace("T", " ") ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>

        <Section title={`security sessions (${sessions.length})`}>
          {sessions.length === 0 ? (
            <div className="font-mono text-[11px] text-ink-3">
              no sessions are routed to the sec pack. A session records the pack it belongs to at ingest, which
              is what keeps a dashcam drive out of this list and out of the ANPR path.
            </div>
          ) : (
            <table className="w-full font-mono text-[11px]">
              <thead>
                <tr className="text-ink-3 text-left border-b hairline">
                  <th className="py-1">session</th><th>camera</th><th>city</th><th>reads</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((s) => (
                  <tr key={s.session_id} className="border-b hairline">
                    <td className="py-1 text-ink-2">
                      <button onClick={() => openSession(s.session_id)}
                        className="hover:text-accent">{s.session_id.slice(0, 8)}</button>
                    </td>
                    <td className="text-ink-3">{s.camera_id ?? "-"}</td>
                    <td className="text-ink-3">{s.city ?? "-"}</td>
                    <td className="text-ink-3 tabular-nums">{s.plate_reads}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </div>
    </PageShell>
  );
}
