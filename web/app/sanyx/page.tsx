"use client";

// SANYX ingest board: every session with its latest health score and decision, colored only by state
// (green pass, amber degraded, red quarantine). Run the six checks on demand, expand a row for per-check
// evidence, and override a decision with a captured reason. Operational Materialism throughout.

import { Fragment, useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { apiGet, apiPost, humanizeError } from "@/lib/api";
import { ago, humanize, num, shortId } from "@/lib/format";

type Check = { name: string; status: string; score: number | null; detail: string; evidence: Record<string, unknown> };
type Row = {
  session_id: string; vehicle_id: string | null; city: string | null; created_at: string | null;
  score: number | null; decision: string | null; root_cause: string | null; n_checks: number;
};

const DECISION: Record<string, string> = { pass: "text-pass", degraded: "text-warn", quarantine: "text-block" };
const DECISION_LABEL: Record<string, string> = { pass: "Healthy", degraded: "Degraded", quarantine: "Quarantined" };

export default function SanyxBoard() {
  const [rows, setRows] = useState<Row[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [report, setReport] = useState<Record<string, Check[]>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadErr, setLoadErr] = useState<unknown>(null);
  const [runErr, setRunErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoadErr(null);
    try {
      const r = await apiGet<{ sessions?: Row[] }>("/api/sanyx/board?limit=200");
      setRows(r.sessions ?? []);
    } catch (e) { setLoadErr(e); }
    finally { setLoaded(true); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const run = async (id: string) => {
    setBusy(id); setRunErr(null);
    try {
      const rep = await apiPost<{ checks?: Check[] }>(`/api/sanyx/session/${id}/run`, {});
      setReport((m) => ({ ...m, [id]: rep.checks ?? [] }));
      setOpen(id);
      await load();
    } catch (e) { setRunErr(`Could not check ${shortId(id)}: ${humanizeError(e)}`); }
    finally { setBusy(null); }
  };

  const expand = async (id: string) => {
    if (open === id) { setOpen(null); return; }
    setOpen(id);
    if (!report[id]) {
      const rep = await apiGet<{ checks?: Check[] }>(`/api/sanyx/session/${id}`).catch(() => null);
      if (rep) setReport((m) => ({ ...m, [id]: rep.checks ?? [] }));
    }
  };

  const counts = rows.reduce((a, r) => { const k = r.decision ?? "unrun"; a[k] = (a[k] ?? 0) + 1; return a; }, {} as Record<string, number>);

  return (
    <PageShell
      active="SANYX"
      title="Health board"
      subtitle="Six quality checks per recording: is the data good enough to label and train on?"
      right={
        <span className="text-[12.5px] text-ink-3">
          <span className="text-pass">{num(counts.pass ?? 0)} healthy</span> ·{" "}
          <span className="text-warn">{num(counts.degraded ?? 0)} degraded</span> ·{" "}
          <span className="text-block">{num(counts.quarantine ?? 0)} quarantined</span> · {num(counts.unrun ?? 0)} not checked
        </span>
      }
    >
      <div className="page-wrap space-y-3">
        {loadErr != null && <LoadState error={loadErr} onRetry={() => void load()} />}
        {runErr && <div role="alert" className="panel border-block/40 px-3.5 py-2.5 text-[12.5px] text-block">{runErr}</div>}
        {!loaded ? <LoadState loading rows={8} /> : !rows.length && loadErr == null ? (
          <LoadState empty emptyIcon="eye" emptyTitle="No recordings yet"
            emptyText="Upload or import a drive and it appears here, ready for its health checks." />
        ) : rows.length > 0 && (
          <section className="panel overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Recording</th><th>City</th><th>Recorded</th>
                  <th className="num">Score</th><th>Result</th><th>Main problem</th><th className="num">Checks</th><th></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <Fragment key={r.session_id}>
                    <tr>
                      <td>
                        <span className="text-ink">{r.vehicle_id ?? "Unknown vehicle"}</span>
                        <span className="ml-2 font-mono text-[11px] text-ink-3">{shortId(r.session_id)}</span>
                      </td>
                      <td>{r.city ?? "-"}</td>
                      <td className="text-ink-3">{r.created_at ? ago(r.created_at) : "-"}</td>
                      <td className="num text-ink">{r.score != null ? r.score.toFixed(0) : "-"}</td>
                      <td className={DECISION[r.decision ?? ""] ?? "text-ink-3"}>
                        {r.decision ? (DECISION_LABEL[r.decision] ?? humanize(r.decision)) : "Not checked"}
                      </td>
                      <td className="text-ink-3">{r.root_cause ? humanize(r.root_cause) : "-"}</td>
                      <td className="num text-ink-3">{r.n_checks || "-"}</td>
                      <td className="text-right whitespace-nowrap space-x-3">
                        <button type="button" onClick={() => expand(r.session_id)} aria-expanded={open === r.session_id}
                          className="link">{open === r.session_id ? "Hide details" : "Details"}</button>
                        <button type="button" onClick={() => run(r.session_id)} disabled={busy === r.session_id}
                          className="link disabled:opacity-50">
                          {busy === r.session_id ? "Checking..." : r.decision ? "Re-check" : "Check"}
                        </button>
                      </td>
                    </tr>
                    {open === r.session_id && (
                      <tr><td colSpan={8} className="bg-bg-2">
                        {(report[r.session_id] ?? []).length ? (
                          <div className="space-y-1.5 py-1">
                            {(report[r.session_id] ?? []).map((c, i) => (
                              <div key={i} className="flex items-start gap-3">
                                <span className={`w-36 shrink-0 ${DECISION[c.status] ?? "text-ink-3"}`}>{humanize(c.name)}</span>
                                <span className="w-20 shrink-0 text-ink-3 tabular-nums">
                                  {c.score != null ? c.score.toFixed(2) : humanize(c.status)}
                                </span>
                                <span className="text-ink-2 leading-snug">{c.detail}</span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <span className="text-ink-3">Not checked yet. Press Check to run the six checks on this recording.</span>
                        )}
                      </td></tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>
    </PageShell>
  );
}
