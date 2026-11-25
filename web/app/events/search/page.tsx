"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Stat } from "@/components/ui/kit";
import { humanize, num, pct, plural } from "@/lib/format";
import { toast } from "@/lib/toast";
import type { EventCorpusSummary, EventHit, EventSearchResult, EventTaxonomy } from "@/lib/types";

// Ask the corpus a question about behaviour.
//
// Every other event surface is scoped to one session, which is the right shape for reviewing a drive and the
// wrong shape for the question the events exist to answer. "Show me every illegal lane change where the
// signal was red" is a fact about the fleet, and a per-session page can never state it.
//
// The conjunction is the part that matters. A filtered list of one kind is something the session view could
// nearly do already; "this kind, while that kind was in that state" is a temporal join, and it is what turns
// a table of derived events into a way of interrogating a corpus.

const SEVERITY_TONE: Record<string, string> = {
  info: "text-ink-3", notable: "text-warn", violation: "text-block",
};

// The signal states a conjunction can be conditioned on. R and Y are the ones worth asking about: what an
// actor did while it was legal to proceed is rarely the interesting question.
const SIGNAL_STATES = [
  { key: "R", label: "red" },
  { key: "Y", label: "amber" },
  { key: "R,Y", label: "red or amber" },
  { key: "G", label: "green" },
];

export default function EventSearchPage() {
  const router = useRouter();
  // Both are absent when they cannot be read, as before.
  const taxonomyQ = useQuery(["eventTaxonomy"], () => api.eventTaxonomy());
  const summaryQ = useQuery(["eventCorpusSummary"], () => api.eventCorpusSummary());
  const taxonomy: EventTaxonomy | null = taxonomyQ.error ? null : taxonomyQ.data ?? null;
  const summary: EventCorpusSummary | null = summaryQ.error ? null : summaryQ.data ?? null;

  const [kind, setKind] = useState("");
  const [severity, setSeverity] = useState("");
  const [state, setState] = useState("");
  const [city, setCity] = useState("");
  const [withKind, setWithKind] = useState("");
  const [withState, setWithState] = useState("");
  const [withinMs, setWithinMs] = useState(2000);

  // The search runs whenever a filter changes, and each filter combination is cached. keepPrevious: the
  // last result stays up while the next one loads.
  const searchArgs = {
    kind: kind || undefined, severity: severity || undefined, state: state || undefined,
    city: city || undefined, withKind: withKind || undefined,
    withState: withState || undefined, withinMs: withKind ? withinMs : undefined,
    limit: 200,
  };
  const searchQ = useQuery(["searchEvents", searchArgs], () => api.searchEvents(searchArgs), { keepPrevious: true });
  const res: EventSearchResult | null = searchQ.data ?? null;
  const busy = searchQ.validating;
  // A search in flight is not a failed one: the error shows once the request has settled.
  const err = searchQ.validating ? null : searchQ.error ?? null;
  const reloadSearch = searchQ.reload;
  const run = useCallback(async () => { await reloadSearch(); }, [reloadSearch]);

  // One click for the question this page was built to answer, so it is discoverable rather than something
  // you have to know how to assemble.
  function askTheHardOne() {
    setKind("lane_change,lane_change_illegal");
    setWithKind("signal_phase");
    setWithState("R,Y");
    setWithinMs(2000);
    setSeverity(""); setState(""); setCity("");
    toast("Every lane change while a signal showed red or amber", "success");
  }

  function open(hit: EventHit) {
    if (hit.frame_id) router.push(`/frame/${hit.frame_id}`);
    else toast("This event has no frame to open", "error");
  }

  const kinds = (taxonomy?.kinds ?? []).map((k) => k.kind);
  const kindsLabel = (v: string) => v.split(",").map((k) => humanize(k).toLowerCase()).join(" or ");
  const signalLabel = (v: string) =>
    SIGNAL_STATES.find((s) => s.key === v)?.label ?? v.split(",").join(" or ");

  return (
    <PageShell active="EVENT SEARCH" title="Behaviour search"
      subtitle="Search what happened across every drive, not just one: an event, optionally while another was happening.">
      <div className="page-wrap space-y-4">

        {summary && (
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Events" value={num(summary.total)} />
            <Stat label="Violations" value={num(summary.by_severity.violation ?? 0)}
              tone={(summary.by_severity.violation ?? 0) > 0 ? "block" : undefined} />
            <Stat label="Notable" value={num(summary.by_severity.notable ?? 0)} tone="warn" />
            <Stat label="Sessions with events" value={num(summary.sessions_with_events)} />
            {Object.entries(summary.cities).map(([c, n]) => (
              <Stat key={c} label={c} value={num(n)} sub={n === 1 ? "session" : "sessions"} />
            ))}
          </div>
        )}

        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Find events</h2>
            <div className="ml-auto">
              <button onClick={askTheHardOne} className="btn"
                data-tip="Every lane change while a traffic signal showed red or amber">
                Try: lane change on red
              </button>
            </div>
          </div>
          <div className="p-4 space-y-3">
            <div className="flex flex-wrap items-end gap-3">
              <Field label="Event">
                <select className="field text-[12.5px] min-w-[180px]" value={kind}
                  onChange={(e) => setKind(e.target.value)}>
                  <option value="">Any</option>
                  {/* A preset can set several kinds at once, and without an option carrying that exact value
                      the control falls back to displaying "any" while the query is in fact filtered, which is
                      a control lying about what is being asked. */}
                  {kind && !kinds.includes(kind) && (
                    <option value={kind}>{humanize(kindsLabel(kind))}</option>
                  )}
                  {kinds.map((k) => <option key={k} value={k}>{humanize(k)}</option>)}
                </select>
              </Field>
              <Field label="Severity">
                <select className="field text-[12.5px]" value={severity}
                  onChange={(e) => setSeverity(e.target.value)}>
                  <option value="">Any</option>
                  {(taxonomy?.severities ?? []).map((s) => <option key={s} value={s}>{humanize(s)}</option>)}
                </select>
              </Field>
              <Field label="Review state">
                <select className="field text-[12.5px]" value={state}
                  onChange={(e) => setState(e.target.value)}>
                  <option value="">Any</option>
                  <option value="review">Awaiting review</option>
                  <option value="confirmed">Confirmed</option>
                  <option value="rejected">Rejected</option>
                </select>
              </Field>
              <Field label="City">
                <select className="field text-[12.5px]" value={city}
                  onChange={(e) => setCity(e.target.value)}>
                  <option value="">Any</option>
                  {Object.keys(summary?.cities ?? {}).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </Field>
            </div>

            {/* The conjunction. A filtered list of one kind is what a session view nearly does already; this
                is the query that makes a corpus of events interrogable. */}
            <div className="flex flex-wrap items-end gap-3 border-t hairline pt-3">
              <span className="text-[12.5px] text-ink-2 pb-1.5">While</span>
              <Field label="Another event">
                <select className="field text-[12.5px] min-w-[180px]" value={withKind}
                  onChange={(e) => setWithKind(e.target.value)}>
                  <option value="">Nothing in particular</option>
                  {kinds.map((k) => <option key={k} value={k}>{humanize(k)}</option>)}
                </select>
              </Field>
              {withKind === "signal_phase" && (
                <Field label="Was showing">
                  <select className="field text-[12.5px]" value={withState}
                    onChange={(e) => setWithState(e.target.value)}>
                    <option value="">Any colour</option>
                    {SIGNAL_STATES.map((s) => <option key={s.key} value={s.key}>{humanize(s.label)}</option>)}
                  </select>
                </Field>
              )}
              {withKind && (
                <Field label="Within">
                  <select className="field text-[12.5px]" value={withinMs}
                    onChange={(e) => setWithinMs(Number(e.target.value))}>
                    <option value={0}>Exactly overlapping</option>
                    <option value={1000}>1 s either side</option>
                    <option value={2000}>2 s either side</option>
                    <option value={5000}>5 s either side</option>
                  </select>
                </Field>
              )}
            </div>
          </div>
        </section>

        {err != null && <LoadState error={err} onRetry={() => void run()} />}

        {res && !err && (
          <section className="panel">
            {/* The question, in words. A row of dropdowns states a query; a sentence states what was
                actually asked, which is what somebody reads before trusting a count of zero. */}
            <div className="panel-head flex-wrap">
              <div className="text-[12.5px] text-ink-2 normal-case tracking-normal font-normal">
                <span className="text-ink font-medium">{plural(res.total, "match", "matches")}</span>
                {" for "}
                <span className="text-ink">
                  {kind ? kindsLabel(kind) : "any event"}
                  {severity ? `, severity ${severity}` : ""}
                  {state ? `, ${state === "review" ? "awaiting review" : state}` : ""}
                  {city ? `, in ${city}` : ""}
                  {withKind ? ` while ${humanize(withKind).toLowerCase()}` : ""}
                  {withKind && withState ? ` was ${signalLabel(withState)}` : ""}
                  {withKind && withinMs ? ` (within ${withinMs / 1000} s)` : ""}
                </span>
              </div>
              <div className="ml-auto text-[12px] text-ink-3 normal-case tracking-normal font-normal">
                {res.returned < res.total ? `Showing the first ${num(res.returned)}, highest confidence first` : ""}
                {res.detail ? ` (${res.detail})` : ""}
              </div>
            </div>

            {res.results.length === 0 ? (
              <EmptyState compact icon="search" title="Nothing matches">
                A search that returns nothing is an answer: the corpus holds no case of these things happening
                together. Loosen a filter to widen it.
              </EmptyState>
            ) : (
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Event</th><th>Severity</th><th>Actor</th>
                      <th>City</th><th>Vehicle</th><th className="num">At</th><th className="num">Lasts</th>
                      <th className="num">Confidence</th><th>State</th><th><span className="sr-only">Open</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {res.results.map((h) => (
                      <tr key={h.event_id}>
                        <td className="text-ink">{humanize(h.kind)}</td>
                        <td className={SEVERITY_TONE[h.severity] ?? ""}>{humanize(h.severity)}</td>
                        <td>{h.actor_class ? humanize(h.actor_class) : "-"}</td>
                        <td>{h.city ?? "-"}</td>
                        <td>{h.vehicle_id ?? "-"}</td>
                        <td className="num">{h.at_s == null ? "-" : `${Number(h.at_s).toFixed(1)} s`}</td>
                        <td className="num text-ink-3">{`${Number(h.duration_s).toFixed(1)} s`}</td>
                        <td className="num">{h.conf == null ? "-" : pct(h.conf)}</td>
                        <td className="text-ink-3">{h.state === "review" ? "Awaiting review" : humanize(h.state)}</td>
                        <td className="text-right">
                          <button onClick={() => open(h)} disabled={!h.frame_id}
                            data-tip={h.frame_id ? "Open the frame where this event starts" : "This event has no frame to open"}
                            className="link disabled:opacity-30 whitespace-nowrap">Open frame &rarr;</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {busy && !res && <LoadState loading rows={6} />}
      </div>
    </PageShell>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="label-caps">{label}</span>
      {children}
    </label>
  );
}
