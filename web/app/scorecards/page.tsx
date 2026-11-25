"use client";

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { api } from "@/lib/api";
import type { ClassRate, Rate, ScorecardsFull } from "@/lib/types";
import { useQuery } from "@/lib/useQuery";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Stat } from "@/components/ui/kit";
import { humanize, num, plural } from "@/lib/format";

// How good each labeller is, per class, whether they are a person or a vendor.
//
// Four scores existed on four surfaces and none of them answered the question you have to answer before
// you can price work or route it. The vendor half of this page shows `workforce_rating`, which has been
// computed since it was written and rendered on no page in this application: the routing weight that
// decides who gets the next batch was invisible to the people deciding.
//
// Every rate here shows its interval, and an unproven one says so rather than showing a number. Three
// right out of three is 1.0 and means very little; a page that printed 100% next to it would be lying by
// omission to whoever is about to send that person more work.

function RateCell({ r, width = 120 }: { r: Rate; width?: number }) {
  if (r.p == null) {
    return <span className="text-[12.5px] text-ink-3">Not checked yet</span>;
  }
  const pct = Math.round(r.p * 100);
  const tone = !r.proven ? "bg-ink-3" : r.lo >= 0.9 ? "bg-pass" : r.lo >= 0.7 ? "bg-warn" : "bg-block";
  return (
    <span className="flex items-center gap-2" title={r.note ?? `${r.n} judged, 95% interval ${(r.lo * 100).toFixed(0)}-${(r.hi * 100).toFixed(0)}%`}>
      <span className="relative block h-1.5 bg-line rounded overflow-hidden" style={{ width }}>
        {/* The interval, not just the estimate: the bar is the range, the tick is the point. */}
        <span className="absolute h-full bg-ink-3/40" style={{ left: `${r.lo * 100}%`, width: `${(r.hi - r.lo) * 100}%` }} />
        <span className={`absolute h-full w-[2px] ${tone}`} style={{ left: `${r.p * 100}%` }} />
      </span>
      <span className={`text-[12.5px] tabular-nums ${r.proven ? "text-ink-2" : "text-ink-3"}`}>
        {pct}%{r.proven ? "" : "?"}
      </span>
    </span>
  );
}

function ClassTable({ rows }: { rows: ClassRate[] }) {
  if (!rows.length) {
    return <div className="px-4 py-3 text-[12.5px] text-ink-3">Nothing this labeller made has been checked by a reviewer yet.</div>;
  }
  return (
    <div className="px-4 pb-3 pt-1 grid gap-x-8 gap-y-1.5 md:grid-cols-2 bg-bg-2/40">
      {rows.map((c) => (
        <div key={c.class_name} className="flex items-center gap-3 text-[12.5px]">
          <span className="w-44 truncate text-ink-2" title={c.class_name}>{humanize(c.class_name)}</span>
          <RateCell r={c} width={90} />
          <span className="text-ink-3 tabular-nums">{c.correct} of {c.judged} right</span>
        </div>
      ))}
    </div>
  );
}

export default function ScorecardsPage() {
  const q = useQuery(["lopScorecardsFull"], () => api.lopScorecardsFull());
  const data: ScorecardsFull | null = q.data ?? null;
  const err: string | null = q.error ? (q.error instanceof Error ? q.error.message : String(q.error)) : null;
  const [open, setOpen] = useState<string | null>(null);

  const rows = [...(data?.people ?? []), ...(data?.vendors ?? [])];

  return (
    <PageShell
      active="Labeller scorecards"
      title="Labeller scorecards"
      subtitle="How often each person's and vendor's labels survive review, per class."
    >
      <div className="page-wrap space-y-4">
        {err && <LoadState error={q.error} onRetry={() => void q.reload()} />}

        {data && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <Stat label="People" value={num(data.people.length)} />
            <Stat label="Vendors" value={num(data.vendors.length)} />
            <Stat label="Labels checked by a reviewer" value={num(data.judged_total)} />
          </div>
        )}

        {/* The caveat travels with the numbers rather than sitting in a footnote nobody reads. */}
        {data && (
          <p className="section-desc">
            {data.caveat.charAt(0).toUpperCase() + data.caveat.slice(1)}. A rate counts as proven after {data.min_judged} checked labels; below that it shows
            a question mark, because unchecked is not the same as good. The bar is the likely range, the tick
            is the estimate.
          </p>
        )}

        {q.loading && !data ? <LoadState loading rows={4} /> : !err && (
        <div className="panel">
          <div className="grid grid-cols-[1fr_auto_auto_auto] gap-3 px-4 py-2.5 border-b hairline label-caps">
            <span>Labeller</span><span className="w-[170px]">Accuracy</span><span className="w-16 text-right">Checked</span><span className="w-16 text-right">Classes</span>
          </div>
          {rows.length === 0 && (
            <EmptyState compact icon="person" title="No scorecards yet">
              A scorecard appears once a reviewer has accepted or corrected someone&apos;s labels. Review some
              labelled work and each labeller&apos;s accuracy shows up here.
            </EmptyState>
          )}
          {rows.map((r) => {
            const id = r.kind === "person" ? r.user_id : r.workforce_id;
            const isOpen = open === id;
            return (
              <div key={id} className="border-b hairline last:border-0">
                <button type="button" onClick={() => setOpen(isOpen ? null : id)} aria-expanded={isOpen}
                  className="grid grid-cols-[1fr_auto_auto_auto] gap-3 items-center w-full px-4 py-2.5 text-left hover:bg-head/60">
                  <span className="flex items-center gap-2 min-w-0">
                    <span className="text-ink-3 text-[11px] w-3" aria-hidden>{isOpen ? "▾" : "▸"}</span>
                    <span className="text-ink text-[13px] truncate">{r.name}</span>
                    <Badge state={r.kind === "vendor" ? "accent" : "muted"}>{r.kind === "vendor" ? "Vendor" : "Person"}</Badge>
                    {r.kind === "vendor" && r.batch.routing_weight != null && (
                      <span className="text-[12px] text-ink-3"
                        data-tip="How much of the next batch this vendor is routed">
                        routing weight {r.batch.routing_weight.toFixed(2)}
                        {r.batch.decided ? ` · ${r.batch.accepted} of ${r.batch.decided} batches accepted` : ""}
                      </span>
                    )}
                    {r.kind === "person" && r.agreement && (
                      <span className="text-[12px] text-ink-3"
                        title="frames where another annotator labelled the same frame independently">
                        {plural(r.agreement.frames_compared, "frame")} compared with others
                      </span>
                    )}
                  </span>
                  <span className="w-[170px]"><RateCell r={r.accuracy} /></span>
                  <span className="text-[12.5px] text-ink-3 w-16 text-right tabular-nums">{num(r.judged)}</span>
                  <span className="text-[12.5px] text-ink-3 w-16 text-right tabular-nums">{num(r.per_class.length)}</span>
                </button>
                {isOpen && <ClassTable rows={r.per_class} />}
              </div>
            );
          })}
        </div>
        )}
      </div>
    </PageShell>
  );
}
