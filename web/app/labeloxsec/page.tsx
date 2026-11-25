"use client";

import PackGate from "@/components/shell/PackGate";
import { T } from "@/lib/i18n";
import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
import type { SecPack, SecSession, SecStats, WatchlistEntry, PlateReadRow } from "@/lib/types";
import EmptyState from "@/components/ui/EmptyState";
import LoadState from "@/components/shell/LoadState";
import { Badge, Stat } from "@/components/ui/kit";
import { ago, humanize, num, pct, shortId } from "@/lib/format";

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
const SEV_BADGE: Record<string, "muted" | "warn" | "block"> = { info: "muted", warn: "warn", critical: "block" };
const SEV_CLASS: Record<string, string> = {
  info: "text-ink-2",
  warn: "text-warn",
  critical: "text-block",
};

function Section({ title, count, children, right, flush = false }: {
  title: string; count?: number; children: React.ReactNode; right?: React.ReactNode; flush?: boolean;
}) {
  return (
    <section className="panel">
      <div className="panel-head flex-wrap">
        <h2 className="section-title">{title}</h2>
        {count != null && <span className="text-[12px] text-ink-3 tabular-nums">{num(count)}</span>}
        {right && <div className="ml-auto flex items-center gap-2">{right}</div>}
      </div>
      <div className={flush ? "" : "p-3.5"}>{children}</div>
    </section>
  );
}

function LabeloxSecPage() {
  // useSearchParams forces a client-side-render bailout, which Next requires be under a Suspense boundary or
  // the static prerender of this route fails the build. Match the pattern used by integrations and search.
  return <Suspense fallback={null}><LabeloxSecBody /></Suspense>;
}

const NO_WATCH: WatchlistEntry[] = [];
const NO_SEC_SESSIONS: SecSession[] = [];
const NO_READS: PlateReadRow[] = [];

function LabeloxSecBody() {
  const params = useSearchParams();
  const packQ = useQuery(["secPack", "sec"], () => api.secPack("sec"));
  const pack: SecPack | null = packQ.data ?? null;
  const packErr: string | null = packQ.error ? humanizeError(packQ.error) : null;
  const statsQ = useQuery(["secStats"], () => api.secStats());
  const watchlistQ = useQuery(["secWatchlist", true], () => api.secWatchlist(true));
  const sessionsQ = useQuery(["secSessions", 50], () => api.secSessions(50));
  const stats: SecStats | null = statsQ.data ?? null;
  const watchlist: WatchlistEntry[] = watchlistQ.data?.entries ?? NO_WATCH;
  const sessions: SecSession[] = sessionsQ.data?.sessions ?? NO_SEC_SESSIONS;
  const restErr = statsQ.error ?? watchlistQ.error ?? sessionsQ.error;
  useEffect(() => { if (restErr) toast(humanizeError(restErr), "error"); }, [restErr]);
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

  // The feed follows the filters, one cached answer per combination. keepPrevious: the last rows stay up
  // while the next ones load.
  const readArgs = {
    plate: plateFilter.trim() || undefined,
    camera_id: cameraFilter.trim() || undefined,
    state_code: stateFilter || undefined,
    hits_only: hitsOnly || undefined,
    limit: 200,
  };
  const readsQ = useQuery(["secReads", readArgs], () => api.secReads(readArgs), { keepPrevious: true });
  const reads: PlateReadRow[] = readsQ.data?.reads ?? NO_READS;
  const readTotal = readsQ.data?.total ?? 0;
  const readsErr = readsQ.error;
  useEffect(() => { if (readsErr) toast(humanizeError(readsErr), "error"); }, [readsErr]);

  // Every filter's feed: a new watch entry turns existing reads into hits whichever filter they sit under.
  const loadReads = useCallback(() => invalidate("secReads"), []);
  const loadRest = useCallback(() => invalidate("secStats", "secWatchlist", "secSessions"), []);

  const addWatch = async () => {
    if (!newPlate.trim()) { toast("Enter a registration number", "error"); return; }
    setBusy(true);
    try {
      const e = await api.secAddWatch(newPlate.trim(), newReason.trim() || null, newSeverity);
      toast(`Watching ${e.plate}`, "success");
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
      toast(`Stopped watching ${e.plate}`, "success");
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
      active="Security console"
      title="Security console"
      subtitle="Number-plate reads, the watchlist and fixed-camera sessions."
      meta={pack ? <Badge state={authorised ? "pass" : "block"}>{authorised ? "Plate reading on" : "Plate reading off"}</Badge> : undefined}
    >
      <div className="page-wrap space-y-4">
        {/* the pack gate */}
        {packErr && <LoadState error={packQ.error} onRetry={() => void packQ.reload()} />}
        {pack && !authorised && (
          <div role="alert" className="panel border-block/40 bg-block/5 p-4 space-y-1.5">
            <div className="text-[13px] font-medium text-block">
              Plate reading is not allowed on this deployment
            </div>
            <p className="text-[12.5px] text-ink-2 max-w-[75ch]">
              Reading number plates is only for authorised security deployments. Here a plate is personal data
              that is blurred and never read, so the server will refuse every action below.
              Packages available: {pack.available_packs.join(", ")}.
            </p>
          </div>
        )}
        {pack && authorised && (
          <div className="panel p-3.5 flex items-start gap-6 flex-wrap">
            <div className="min-w-0">
              <div className="text-[13px] text-ink font-medium">{pack.name}</div>
              <div className="text-[12px] text-ink-3 mt-0.5">
                Can do: {pack.capabilities.map((c) => humanize(c).toLowerCase()).join(", ")}
              </div>
            </div>
            {pack.safety_classes.length > 0 && (
              <div className="ml-auto">
                <div className="label-caps">Safety-critical classes</div>
                <div className="text-[12.5px] text-ink-2 mt-0.5">{pack.safety_classes.map(humanize).join(", ")}</div>
              </div>
            )}
          </div>
        )}

        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
            <Stat label="Plate reads" value={num(stats.reads)} />
            <Stat label="Watchlist hits" value={num(stats.watchlist_hits)} tone={stats.watchlist_hits ? "warn" : undefined} />
            <Stat label="Valid format" value={num(stats.valid_format)}
              sub={stats.reads ? `${pct(stats.valid_format / stats.reads)} of reads` : undefined} />
            <Stat label="Plates watched" value={num(stats.watchlist_size)} />
            {/* Surfaced rather than hidden: it tells the operator a confidence filter is meaningless on those
                rows, because the local reader exposes no calibrated score. */}
            <Stat label="Without a confidence" value={num(stats.unscored_reads)} sub="the reader gave no score" />
          </div>
        )}

        {stats && Object.keys(stats.top_states).length > 0 && (
          <Section title="Reads by state" right={<span className="text-[12px] text-ink-3">Select a state to filter the reads below</span>}>
            <div className="flex gap-1.5 flex-wrap">
              {/* A facet, not a legend: clicking a state filters the feed below, and clicking it again clears
                  the filter. A count that looks clickable and is not is worse than no affordance at all. */}
              {Object.entries(stats.top_states).map(([code, n]) => (
                <button key={code} type="button" aria-pressed={stateFilter === code}
                  onClick={() => setStateFilter((s) => (s === code ? "" : code))}
                  className={`chip ${stateFilter === code ? "chip-on" : ""}`}>
                  {code} <span className="text-ink-3 tabular-nums">{num(n)}</span>
                </button>
              ))}
              {stateFilter && (
                <button type="button" onClick={() => setStateFilter("")} className="link text-[12px] px-1">
                  <T k="action.clear">Clear</T>
                </button>
              )}
            </div>
          </Section>
        )}

        <Section title="Watchlist" count={watchlist.length} flush>
          <form className="p-3.5 space-y-2" onSubmit={(e) => { e.preventDefault(); void addWatch(); }}>
            <div className="flex items-end gap-2 text-[12.5px] flex-wrap">
              <label className="flex flex-col gap-1.5"><span className="text-[12px] text-ink-2">Registration number</span>
                <input value={newPlate} onChange={(e) => setNewPlate(e.target.value)}
                  placeholder="KA 01 AB 1234" className="field w-44 font-mono" />
              </label>
              <label className="flex flex-col gap-1.5 flex-1 min-w-[220px]"><span className="text-[12px] text-ink-2">Reason (optional)</span>
                <input value={newReason} onChange={(e) => setNewReason(e.target.value)}
                  placeholder="Why this plate is watched" className="field w-full" />
              </label>
              <label className="flex flex-col gap-1.5"><span className="text-[12px] text-ink-2">Severity</span>
                <select value={newSeverity} onChange={(e) => setNewSeverity(e.target.value)} className="field">
                  {SEVERITIES.map((s) => <option key={s} value={s}>{humanize(s)}</option>)}
                </select>
              </label>
              <button type="submit" disabled={busy || !authorised} className="btn btn-primary">
                Watch plate
              </button>
            </div>
            <p className="section-desc">
              Spacing and dashes do not matter: KA 01 AB 1234, ka-01-ab-1234 and KA01AB1234 are the same entry.
            </p>
          </form>

          {watchlist.length > 0 ? (
            <table className="data-table border-t hairline">
              <thead>
                <tr><th>Plate</th><th>Severity</th><th>Reason</th><th>Added by</th><th></th></tr>
              </thead>
              <tbody>
                {watchlist.map((w) => (
                  <tr key={w.entry_id}>
                    <td className="text-ink font-mono">{w.plate}</td>
                    <td><Badge state={SEV_BADGE[w.severity] ?? "muted"}>{humanize(w.severity)}</Badge></td>
                    <td className="truncate max-w-[320px]">{w.reason ?? "-"}</td>
                    <td>{w.added_by ?? "-"}</td>
                    <td className="text-right">
                      <button type="button" onClick={() => removeWatch(w)} className="btn btn-danger">
                        Stop watching
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="px-3.5 pb-3.5 text-[12.5px] text-ink-3">No plates are being watched.</p>
          )}
        </Section>

        <Section title="Plate reads" count={readTotal} flush
          right={
            <>
              <input value={plateFilter} onChange={(e) => setPlateFilter(e.target.value)}
                placeholder="Plate" aria-label="Filter by plate" className="field w-36 font-mono" />
              <input value={cameraFilter} onChange={(e) => setCameraFilter(e.target.value)}
                placeholder="Camera" aria-label="Filter by camera" className="field w-32" />
              <label className="flex items-center gap-1.5 text-[12.5px] text-ink-2">
                <input type="checkbox" checked={hitsOnly} onChange={(e) => setHitsOnly(e.target.checked)} />
                Watchlist hits only
              </label>
            </>
          }>
          {reads.length === 0 ? (
            <EmptyState compact icon="eye" title={readsQ.loading ? "Loading reads..." : "No plate reads"}>
              {plateFilter || cameraFilter || hitsOnly || stateFilter
                ? "Nothing matches these filters. Clear a filter to see more."
                : "Reads appear here once plates are recognised in a security session's frames. Each read is stored with its session, so deleting a session removes its plates too."}
            </EmptyState>
          ) : (
            <>
              {reads.length < readTotal && (
                <p className="section-desc px-3.5 pt-2.5">Showing the newest {num(reads.length)} of {num(readTotal)}.</p>
              )}
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Plate</th><th>Type</th><th>State</th><th>Camera</th>
                    <th className="num" data-tip="How sure the detector was that this is a plate">Detection</th>
                    <th className="num" data-tip="How sure the reader was of the characters">Reading</th>
                    <th className="num" data-tip="How well the text matches an Indian plate format">Format</th>
                    <th>When</th>
                  </tr>
                </thead>
                <tbody>
                  {reads.map((r) => (
                    <tr key={r.read_id} className={r.watchlist_hit ? "bg-block/10" : ""}>
                      <td className={`font-mono ${r.watchlist_hit ? SEV_CLASS[r.watchlist_severity ?? "warn"] : "text-ink"}`}>
                        {r.plate || r.plate_raw}
                        {r.watchlist_hit && (
                          <span className="ml-2 font-body"><Badge state={SEV_BADGE[r.watchlist_severity ?? "warn"] ?? "warn"}>
                            {r.watchlist_severity ? `Watched: ${r.watchlist_severity}` : "Watched"}
                          </Badge></span>
                        )}
                      </td>
                      <td>{humanize(r.plate_type)}</td>
                      <td>{r.state_code ?? "-"}</td>
                      <td>{r.camera_id ?? "-"}</td>
                      <td className="num">{r.det_conf.toFixed(2)}</td>
                      {/* A dash, not a zero: the reader measured nothing and inventing a number here would make
                          the column look sortable in a way it is not. */}
                      <td className="num">{r.ocr_conf == null ? "-" : r.ocr_conf.toFixed(2)}</td>
                      <td className={`num ${r.valid ? "text-pass" : ""}`}>{r.format_confidence.toFixed(2)}</td>
                      <td className="text-ink-3 whitespace-nowrap" title={r.created_at ?? undefined}>{r.created_at ? ago(r.created_at) : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </Section>

        <Section title="Security sessions" count={sessions.length} flush>
          {sessions.length === 0 ? (
            <EmptyState compact icon="folder" title="No security sessions">
              Recordings from fixed security cameras appear here. A session is marked as security or driving
              footage when it is uploaded, which keeps dashcam drives out of plate reading.
            </EmptyState>
          ) : (
            <table className="data-table">
              <thead>
                <tr><th>Session</th><th>Camera</th><th>City</th><th className="num">Plate reads</th></tr>
              </thead>
              <tbody>
                {sessions.map((s) => (
                  <tr key={s.session_id}>
                    <td>
                      <button type="button" onClick={() => openSession(s.session_id)} className="link font-mono text-[12px]">
                        {shortId(s.session_id)}
                      </button>
                    </td>
                    <td>{s.camera_id ?? "-"}</td>
                    <td>{s.city ?? "-"}</td>
                    <td className="num">{num(s.plate_reads)}</td>
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

// LabeloxSec is an optional package (settings.packs.optional on the server); without it these routes do not exist.
export default function LabeloxSecPageRoute() {
  return (
    <PackGate pack="sec" title="Security console"
      what="LabeloxSec, for authorised security deployments: number-plate reading, watchlists, zones and following a person across cameras">
      <LabeloxSecPage />
    </PackGate>
  );
}
