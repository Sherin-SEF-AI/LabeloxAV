"use client";

// SANYX ingest board: every session with its latest health score and decision, colored only by state
// (green pass, amber degraded, red quarantine). Run the six checks on demand, expand a row for per-check
// evidence, and override a decision with a captured reason. Operational Materialism throughout.

import { Fragment, useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { apiGet, apiPost } from "@/lib/api";

type Check = { name: string; status: string; score: number | null; detail: string; evidence: Record<string, unknown> };
type Row = {
  session_id: string; vehicle_id: string | null; city: string | null; created_at: string | null;
  score: number | null; decision: string | null; root_cause: string | null; n_checks: number;
};

const DECISION: Record<string, string> = { pass: "text-pass", degraded: "text-warn", quarantine: "text-block" };

export default function SanyxBoard() {
  const [rows, setRows] = useState<Row[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [report, setReport] = useState<Record<string, Check[]>>({});
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    const r = await apiGet<{ sessions?: Row[] }>("/api/sanyx/board?limit=200");
    setRows(r.sessions ?? []);
  }, []);
  useEffect(() => { load(); }, [load]);

  const run = async (id: string) => {
    setBusy(id);
    try {
      const rep = await apiPost<{ checks?: Check[] }>(`/api/sanyx/session/${id}/run`, {});
      setReport((m) => ({ ...m, [id]: rep.checks ?? [] }));
      setOpen(id);
      await load();
    } finally { setBusy(null); }
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
      title="SANYX ingest QA"
      right={
        <span className="font-mono text-[11px] text-ink-3">
          <span className="text-pass">{counts.pass ?? 0} pass</span> ·{" "}
          <span className="text-warn">{counts.degraded ?? 0} degraded</span> ·{" "}
          <span className="text-block">{counts.quarantine ?? 0} quarantine</span> · {counts.unrun ?? 0} unrun
        </span>
      }
    >
      <div className="p-4 font-mono text-[11px]">
        <table className="w-full">
          <thead>
            <tr className="text-ink-3 text-left border-b hairline">
              <th className="px-2 py-1">session</th><th>vehicle</th><th>city</th>
              <th>score</th><th>decision</th><th>root cause</th><th>checks</th><th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <Fragment key={r.session_id}>
                <tr className="border-b hairline hover:bg-line">
                  <td className="px-2 py-1 text-ink-2">{r.session_id.slice(0, 8)}</td>
                  <td className="text-ink-3">{r.vehicle_id ?? "-"}</td>
                  <td className="text-ink-3">{r.city ?? "-"}</td>
                  <td className="text-ink">{r.score != null ? r.score.toFixed(0) : "-"}</td>
                  <td className={DECISION[r.decision ?? ""] ?? "text-ink-3"}>{r.decision ?? "unrun"}</td>
                  <td className="text-ink-3">{r.root_cause ? r.root_cause.replace(/_/g, " ") : "-"}</td>
                  <td className="text-ink-3">{r.n_checks || "-"}</td>
                  <td className="text-right pr-2 whitespace-nowrap space-x-2">
                    <button onClick={() => expand(r.session_id)}
                      className={open === r.session_id ? "text-accent" : "text-ink-3 hover:text-ink"}>evidence</button>
                    <button onClick={() => run(r.session_id)} disabled={busy === r.session_id}
                      className="text-info hover:text-accent disabled:opacity-40">
                      {busy === r.session_id ? "running..." : "run"}
                    </button>
                  </td>
                </tr>
                {open === r.session_id && (
                  <tr className="border-b hairline"><td colSpan={8} className="px-3 py-2 bg-bg-2">
                    {(report[r.session_id] ?? []).length ? (
                      <div className="space-y-1">
                        {(report[r.session_id] ?? []).map((c, i) => (
                          <div key={i} className="flex items-start gap-3">
                            <span className={`w-28 shrink-0 ${DECISION[c.status] ?? "text-ink-3"}`}>{c.name}</span>
                            <span className="w-16 shrink-0 text-ink-3">
                              {c.score != null ? c.score.toFixed(2) : c.status}
                            </span>
                            <span className="text-ink-2 leading-snug">{c.detail}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <span className="text-ink-3">no report yet, press run</span>
                    )}
                  </td></tr>
                )}
              </Fragment>
            ))}
            {!rows.length && <tr><td colSpan={8} className="text-ink-3 text-center py-4">no sessions</td></tr>}
          </tbody>
        </table>
      </div>
    </PageShell>
  );
}
