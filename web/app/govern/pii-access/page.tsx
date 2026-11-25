"use client";

import { useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import { useQuery } from "@/lib/useQuery";
import type { PiiAccessRow } from "@/lib/types";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Segmented, Stat } from "@/components/ui/kit";
import { ago, humanize, num, plural, shortId } from "@/lib/format";

// Who looked at personal data.
//
// The privacy plane already recorded what the redactor found in each frame. Nothing recorded what a human
// then viewed, so the system could describe its personal data and could not say who had seen it, which is
// the half of the question a DPDPA or GDPR enquiry actually turns on.
//
// Admin-only, and the page says why: this is a record about employees as much as about data subjects, and a
// reviewer being able to browse colleagues' viewing history would be surveillance rather than compliance.

const WINDOWS: [string, number][] = [["24 hours", 24], ["Week", 168], ["Month", 720], ["Quarter", 2160]];

const NO_ROWS: PiiAccessRow[] = [];

const ACTION_WORDS: Record<string, string> = {
  view: "Viewed", read_plate: "Read a plate", download: "Downloaded", export: "Exported",
};

export default function PiiAccessPage() {
  const [hours, setHours] = useState(168);
  const [action, setAction] = useState("");
  const [subject, setSubject] = useState("");
  const [denied, setDenied] = useState(false);

  // keepPrevious: changing a filter leaves the last rows up until the new ones arrive.
  const listArgs = { since_hours: hours, action: action || undefined,
                     subject_id: subject.trim() || undefined, limit: 300 };
  const listQ = useQuery(["piiAccess", listArgs], () => api.piiAccess(listArgs), { keepPrevious: true });
  const sumQ = useQuery(["piiAccessSummary", hours], () => api.piiAccessSummary(hours), { keepPrevious: true });
  const rows: PiiAccessRow[] = listQ.data?.accesses ?? NO_ROWS;
  const total = listQ.data?.total ?? 0;
  const summary = sumQ.data ?? null;

  // A refusal is a state of the page, anything else is announced. `denied` clears only when a read
  // succeeds, so it does not flicker off while a changed filter is being refused again.
  const failure = listQ.error ?? sumQ.error;
  useEffect(() => {
    if (!failure) return;
    const msg = humanizeError(failure);
    if (msg.toLowerCase().includes("admin") || msg.includes("403")) setDenied(true);
    else toast(msg, "error");
  }, [failure]);
  const settled = !failure && !listQ.validating && !sumQ.validating && Boolean(listQ.data) && Boolean(sumQ.data);
  useEffect(() => { if (settled) setDenied(false); }, [settled]);

  return (
    <PageShell active="PII ACCESS" title="PII access log"
      subtitle="Who viewed personal data (faces, number plates), and whether it was redacted when they did."
      filters={
        <>
          <span className="text-ink-3">Period</span>
          <Segmented value={String(hours)} onChange={(v) => setHours(Number(v))}
            options={WINDOWS.map(([label, h]) => ({ value: String(h), label }))} />
          <span className="text-ink-3 ml-3">Action</span>
          <Segmented value={action || "all"} onChange={(v) => setAction(v === "all" ? "" : v)}
            options={["all", "view", "read_plate", "download", "export"].map((a) => ({
              value: a, label: a === "all" ? "All" : ACTION_WORDS[a] ?? humanize(a) }))} />
          <input value={subject} onChange={(e) => setSubject(e.target.value)}
            placeholder="Filter by subject id" aria-label="Filter by subject id"
            className="field ml-auto text-[12px] w-56" />
        </>
      }
    >
      <div className="page-wrap space-y-4">
        {denied ? (
          <div className="panel">
            <EmptyState icon="shield" title="This log is for administrators only">
              It records who viewed personal data, which makes it a record about the people who work here as much
              as about the people in the recordings. Browsing a colleague&apos;s viewing history would be
              surveillance rather than compliance, so access is deliberately restricted.
            </EmptyState>
          </div>
        ) : (
          <>
            {summary && (
              <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
                <Stat label="Accesses" value={num(summary.accesses)} />
                {/* The number a policy is actually written about: viewing a blurred frame is ordinary
                    work, viewing the original is the governed act. */}
                <Stat label="Unredacted" value={num(summary.unredacted)} tone={summary.unredacted ? "warn" : undefined}
                  sub="views of the original, unblurred" />
                {Object.entries(summary.by_action).map(([a, n]) => (
                  <Stat key={a} label={ACTION_WORDS[a] ?? humanize(a)} value={num(n)} />
                ))}
              </div>
            )}

            {summary && Object.keys(summary.by_user).length > 0 && (
              <section className="panel">
                <div className="panel-head"><h2 className="section-title">By person</h2></div>
                <div className="p-3 flex gap-1.5 flex-wrap">
                  {Object.entries(summary.by_user).sort((a, b) => b[1] - a[1]).map(([u, n]) => (
                    <span key={u} className="chip cursor-default">
                      {u} <span className="text-ink-3 tabular-nums">{num(n)}</span>
                    </span>
                  ))}
                </div>
              </section>
            )}

            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Accesses</h2>
                <span className="ml-auto text-[12px] text-ink-3 font-normal">
                  {rows.length < total ? `Showing ${num(rows.length)} of ${num(total)}` : plural(total, "access", "accesses")}
                </span>
              </div>
              {listQ.loading && !listQ.data ? (
                <div className="p-3"><LoadState loading rows={5} /></div>
              ) : rows.length === 0 ? (
                <EmptyState compact icon="shield" title="Nothing recorded in this period">
                  An access is written when a frame known to contain a face or a plate is served, and when the
                  security console reads a registration number.
                </EmptyState>
              ) : (
                <div className="overflow-x-auto">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>When</th><th>Who</th><th>Action</th>
                        <th>What</th><th>Contains</th><th>Shown</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={r.access_id}>
                          <td className="text-ink-3 whitespace-nowrap" title={r.created_at ?? undefined}>
                            {ago(r.created_at)}
                          </td>
                          <td className="text-ink">{r.user_name ?? "Unknown"}</td>
                          <td>{ACTION_WORDS[r.action] ?? humanize(r.action)}</td>
                          <td>
                            {humanize(r.subject_type)}{" "}
                            {r.subject_type === "frame"
                              ? <a href={`/frame/${r.subject_id}`} className="link font-mono text-[11.5px]">{shortId(r.subject_id)}</a>
                              : <span className="font-mono text-[11.5px] text-ink-3">{shortId(r.subject_id)}</span>}
                          </td>
                          <td className="text-ink-3">{r.pii_kinds.map(humanize).join(", ") || "-"}</td>
                          <td className={r.redacted ? "text-ink-3" : "text-warn"}>
                            {r.redacted ? "Redacted" : "Unredacted"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </PageShell>
  );
}
