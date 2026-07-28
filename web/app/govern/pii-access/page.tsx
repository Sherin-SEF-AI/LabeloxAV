"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import type { PiiAccessRow } from "@/lib/types";

// Who looked at personal data.
//
// The privacy plane already recorded what the redactor found in each frame. Nothing recorded what a human
// then viewed, so the system could describe its personal data and could not say who had seen it, which is
// the half of the question a DPDPA or GDPR enquiry actually turns on.
//
// Admin-only, and the page says why: this is a record about employees as much as about data subjects, and a
// reviewer being able to browse colleagues' viewing history would be surveillance rather than compliance.

const WINDOWS: [string, number][] = [["24h", 24], ["week", 168], ["month", 720], ["quarter", 2160]];

export default function PiiAccessPage() {
  const [rows, setRows] = useState<PiiAccessRow[]>([]);
  const [total, setTotal] = useState(0);
  const [summary, setSummary] = useState<Awaited<ReturnType<typeof api.piiAccessSummary>> | null>(null);
  const [hours, setHours] = useState(168);
  const [action, setAction] = useState("");
  const [subject, setSubject] = useState("");
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    try {
      const [list, sum] = await Promise.all([
        api.piiAccess({ since_hours: hours, action: action || undefined,
                        subject_id: subject.trim() || undefined, limit: 300 }),
        api.piiAccessSummary(hours),
      ]);
      setRows(list.accesses);
      setTotal(list.total);
      setSummary(sum);
      setDenied(false);
    } catch (e) {
      const msg = humanizeError(e);
      if (msg.toLowerCase().includes("admin") || msg.includes("403")) setDenied(true);
      else toast(msg, "error");
    }
  }, [hours, action, subject]);

  useEffect(() => { load(); }, [load]);

  return (
    <PageShell active="PII ACCESS" title="PII access log"
      subtitle="who viewed personal data, and whether it was redacted"
      filters={
        <>
          <div className="flex items-center gap-1">
            {WINDOWS.map(([label, h]) => (
              <button key={h} onClick={() => setHours(h)}
                className={`px-2 py-0.5 border ${
                  hours === h ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
                {label}
              </button>
            ))}
          </div>
          <span className="text-ink-4">|</span>
          {["", "view", "read_plate", "download", "export"].map((a) => (
            <button key={a || "all"} onClick={() => setAction(a)}
              className={`px-2 py-0.5 border ${
                action === a ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
              {a || "all actions"}
            </button>
          ))}
          <input value={subject} onChange={(e) => setSubject(e.target.value)}
            placeholder="subject id"
            className="ml-auto bg-bg border border-line px-1.5 py-0.5 text-ink w-52" />
        </>
      }
    >
      <div className="p-4 space-y-4 max-w-6xl">
        {denied ? (
          <div className="panel border border-block p-3 space-y-1">
            <div className="font-mono text-[11px] text-block">this log is administrator-only</div>
            <div className="font-mono text-[10.5px] text-ink-3">
              It records who viewed personal data, which makes it a record about people who work here as much
              as about the subjects in the corpus. Browsing a colleague&apos;s viewing history is surveillance
              rather than compliance, so the floor is deliberately high.
            </div>
          </div>
        ) : (
          <>
            {summary && (
              <div className="flex gap-2 flex-wrap">
                <div className="panel px-3 py-2 min-w-[120px]">
                  <div className="font-mono text-[10px] uppercase text-ink-3">accesses</div>
                  <div className="font-mono text-[18px] text-ink tabular-nums">{summary.accesses}</div>
                </div>
                <div className="panel px-3 py-2 min-w-[120px]">
                  <div className="font-mono text-[10px] uppercase text-ink-3">unredacted</div>
                  {/* The number a policy is actually written about: viewing a blurred frame is ordinary
                      work, viewing the original is the governed act. */}
                  <div className={`font-mono text-[18px] tabular-nums ${
                    summary.unredacted ? "text-warn" : "text-ink"}`}>{summary.unredacted}</div>
                  <div className="font-mono text-[10px] text-ink-3">the governed act</div>
                </div>
                {Object.entries(summary.by_action).map(([a, n]) => (
                  <div key={a} className="panel px-3 py-2 min-w-[110px]">
                    <div className="font-mono text-[10px] uppercase text-ink-3">{a}</div>
                    <div className="font-mono text-[18px] text-ink tabular-nums">{n}</div>
                  </div>
                ))}
              </div>
            )}

            {summary && Object.keys(summary.by_user).length > 0 && (
              <section className="panel">
                <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                  by person
                </div>
                <div className="p-3 flex gap-1 flex-wrap">
                  {Object.entries(summary.by_user).sort((a, b) => b[1] - a[1]).map(([u, n]) => (
                    <span key={u} className="border border-line px-1.5 py-0.5 font-mono text-[10.5px] text-ink-2">
                      {u} <span className="text-ink-3">{n}</span>
                    </span>
                  ))}
                </div>
              </section>
            )}

            <section className="panel">
              <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                accesses ({rows.length} of {total})
              </div>
              <div className="p-3">
                {rows.length === 0 ? (
                  <div className="font-mono text-[11px] text-ink-3">
                    Nothing recorded in this window. An access is written when a frame known to contain a
                    face or a plate is served, and when the security console reads a registration mark.
                  </div>
                ) : (
                  <table className="w-full font-mono text-[11px]">
                    <thead>
                      <tr className="text-ink-3 text-left border-b hairline">
                        <th className="py-1">when</th><th>who</th><th>action</th>
                        <th>subject</th><th>kinds</th><th>state</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={r.access_id} className="border-b hairline">
                          <td className="py-1 text-ink-3">
                            {r.created_at?.slice(0, 19).replace("T", " ") ?? "-"}
                          </td>
                          <td className="text-ink-2">{r.user_name ?? "unknown"}</td>
                          <td className="text-ink-3">{r.action}</td>
                          <td className="text-ink-3">
                            {r.subject_type} {r.subject_id.slice(0, 8)}
                          </td>
                          <td className="text-ink-3">{r.pii_kinds.join(", ") || "-"}</td>
                          <td className={r.redacted ? "text-ink-3" : "text-warn"}>
                            {r.redacted ? "redacted" : "unredacted"}
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
