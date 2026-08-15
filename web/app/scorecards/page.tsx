"use client";

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { api } from "@/lib/api";
import type { ClassRate, Rate, ScorecardsFull } from "@/lib/types";

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
    return <span className="font-mono text-[11px] text-ink-3">not judged</span>;
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
      <span className={`font-mono text-[11px] ${r.proven ? "text-ink-2" : "text-ink-3"}`}>
        {pct}%{r.proven ? "" : "?"}
      </span>
    </span>
  );
}

function ClassTable({ rows }: { rows: ClassRate[] }) {
  if (!rows.length) {
    return <div className="px-4 py-3 font-mono text-[11px] text-ink-3">nothing this labeller made has been ruled on yet.</div>;
  }
  return (
    <div className="px-4 py-2 grid gap-1 md:grid-cols-2">
      {rows.map((c) => (
        <div key={c.class_name} className="flex items-center gap-3 font-mono text-[11px]">
          <span className="w-40 truncate text-ink-2">{c.class_name}</span>
          <RateCell r={c} width={90} />
          <span className="text-ink-3">{c.correct}/{c.judged}</span>
        </div>
      ))}
    </div>
  );
}

export default function ScorecardsPage() {
  const [data, setData] = useState<ScorecardsFull | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    api.lopScorecardsFull()
      .then(setData)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  const rows = [...(data?.people ?? []), ...(data?.vendors ?? [])];

  return (
    <PageShell
      active="SCORECARDS"
      title="Labeller scorecards"
      subtitle={data ? `${data.people.length} people, ${data.vendors.length} vendors, ${data.judged_total} judged labels` : "loading"}
    >
      <div className="p-4 space-y-4">
        {err && <div className="panel p-3 font-mono text-[11px] text-block">{err}</div>}

        {/* The caveat travels with the numbers rather than sitting in a footnote nobody reads. */}
        {data && (
          <div className="panel p-3 font-mono text-[11px] text-ink-3">
            {data.caveat}. A rate is called proven at {data.min_judged} judged labels; below that it shows
            a question mark, because unchecked is not the same as good.
          </div>
        )}

        <div className="panel">
          <div className="grid grid-cols-[1fr_auto_auto_auto] gap-3 px-4 py-2 border-b hairline
                          font-display font-semibold text-[10px] uppercase tracking-wider text-ink-3">
            <span>labeller</span><span>accuracy</span><span>judged</span><span>classes</span>
          </div>
          {rows.length === 0 && !err && (
            <div className="px-4 py-8 text-center font-mono text-[11px] text-ink-3">
              nobody has labelled anything that a second person has ruled on.
            </div>
          )}
          {rows.map((r) => {
            const id = r.kind === "person" ? r.user_id : r.workforce_id;
            const isOpen = open === id;
            return (
              <div key={id} className="border-b hairline last:border-0">
                <button onClick={() => setOpen(isOpen ? null : id)}
                  className="grid grid-cols-[1fr_auto_auto_auto] gap-3 items-center w-full px-4 py-2 text-left hover:bg-line/20">
                  <span className="flex items-center gap-2 min-w-0">
                    <span className={`font-mono text-[9px] px-1 border rounded ${
                      r.kind === "vendor" ? "border-accent text-accent" : "border-line text-ink-3"}`}>
                      {r.kind}
                    </span>
                    <span className="text-ink truncate">{r.name}</span>
                    {r.kind === "vendor" && r.batch.routing_weight != null && (
                      <span className="font-mono text-[10px] text-ink-3">
                        routing {r.batch.routing_weight.toFixed(2)}
                        {r.batch.decided ? ` · ${r.batch.accepted}/${r.batch.decided} batches` : ""}
                      </span>
                    )}
                    {r.kind === "person" && r.agreement && (
                      <span className="font-mono text-[10px] text-ink-3"
                        title="frames where another annotator labelled the same frame independently">
                        {r.agreement.frames_compared} compared
                      </span>
                    )}
                  </span>
                  <RateCell r={r.accuracy} />
                  <span className="font-mono text-[11px] text-ink-3 w-14 text-right">{r.judged}</span>
                  <span className="font-mono text-[11px] text-ink-3 w-14 text-right">{r.per_class.length}</span>
                </button>
                {isOpen && <ClassTable rows={r.per_class} />}
              </div>
            );
          })}
        </div>
      </div>
    </PageShell>
  );
}
