"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
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
  const [taxonomy, setTaxonomy] = useState<EventTaxonomy | null>(null);
  const [summary, setSummary] = useState<EventCorpusSummary | null>(null);
  const [res, setRes] = useState<EventSearchResult | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const [kind, setKind] = useState("");
  const [severity, setSeverity] = useState("");
  const [state, setState] = useState("");
  const [city, setCity] = useState("");
  const [withKind, setWithKind] = useState("");
  const [withState, setWithState] = useState("");
  const [withinMs, setWithinMs] = useState(2000);

  useEffect(() => {
    api.eventTaxonomy().then(setTaxonomy).catch(() => setTaxonomy(null));
    api.eventCorpusSummary().then(setSummary).catch(() => setSummary(null));
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setErr(null);
    try {
      setRes(await api.searchEvents({
        kind: kind || undefined, severity: severity || undefined, state: state || undefined,
        city: city || undefined, withKind: withKind || undefined,
        withState: withState || undefined, withinMs: withKind ? withinMs : undefined,
        limit: 200,
      }));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }, [kind, severity, state, city, withKind, withState, withinMs]);

  useEffect(() => { void run(); }, [run]);

  // One click for the question this page was built to answer, so it is discoverable rather than something
  // you have to know how to assemble.
  function askTheHardOne() {
    setKind("lane_change,lane_change_illegal");
    setWithKind("signal_phase");
    setWithState("R,Y");
    setWithinMs(2000);
    setSeverity(""); setState(""); setCity("");
    toast("every lane change while a signal showed red or amber", "success");
  }

  function open(hit: EventHit) {
    if (hit.frame_id) router.push(`/frame/${hit.frame_id}`);
    else toast("this event has no frame to open", "error");
  }

  const kinds = (taxonomy?.kinds ?? []).map((k) => k.kind);

  return (
    <PageShell active="EVENT SEARCH" title="Behaviour search"
      subtitle="what happened across the corpus, not in one drive">
      <div className="p-4 space-y-3 max-w-6xl">

        {summary && (
          <div className="flex flex-wrap gap-2 font-mono text-[11px]">
            <Stat label="events" value={summary.total} />
            <Stat label="violations" value={summary.by_severity.violation ?? 0}
              tone={(summary.by_severity.violation ?? 0) > 0 ? "text-block" : undefined} />
            <Stat label="notable" value={summary.by_severity.notable ?? 0} tone="text-warn" />
            <Stat label="sessions" value={summary.sessions_with_events} />
            {Object.entries(summary.cities).map(([c, n]) => (
              <Stat key={c} label={c.toLowerCase()} value={n} hint="sessions" />
            ))}
          </div>
        )}

        <div className="panel px-3 py-2 space-y-2 font-mono text-[11px]">
          <div className="flex flex-wrap items-end gap-2">
            <Field label="kind">
              <select className="input font-mono text-[11px]" value={kind}
                onChange={(e) => setKind(e.target.value)}>
                <option value="">any</option>
                {/* A preset can set several kinds at once, and without an option carrying that exact value
                    the control falls back to displaying "any" while the query is in fact filtered, which is
                    a control lying about what is being asked. */}
                {kind && !kinds.includes(kind) && (
                  <option value={kind}>{kind.split(",").join(" or ")}</option>
                )}
                {kinds.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </Field>
            <Field label="severity">
              <select className="input font-mono text-[11px]" value={severity}
                onChange={(e) => setSeverity(e.target.value)}>
                <option value="">any</option>
                {(taxonomy?.severities ?? []).map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="review state">
              <select className="input font-mono text-[11px]" value={state}
                onChange={(e) => setState(e.target.value)}>
                <option value="">any</option>
                <option value="review">review</option>
                <option value="confirmed">confirmed</option>
                <option value="rejected">rejected</option>
              </select>
            </Field>
            <Field label="city">
              <select className="input font-mono text-[11px]" value={city}
                onChange={(e) => setCity(e.target.value)}>
                <option value="">any</option>
                {Object.keys(summary?.cities ?? {}).map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </Field>
          </div>

          {/* The conjunction. A filtered list of one kind is what a session view nearly does already; this
              is the query that makes a corpus of events interrogable. */}
          <div className="flex flex-wrap items-end gap-2 border-t hairline pt-2">
            <span className="text-ink-3 uppercase text-[10px] pb-1.5">while</span>
            <Field label="another event">
              <select className="input font-mono text-[11px]" value={withKind}
                onChange={(e) => setWithKind(e.target.value)}>
                <option value="">nothing in particular</option>
                {kinds.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </Field>
            {withKind === "signal_phase" && (
              <Field label="was showing">
                <select className="input font-mono text-[11px]" value={withState}
                  onChange={(e) => setWithState(e.target.value)}>
                  <option value="">any colour</option>
                  {SIGNAL_STATES.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
                </select>
              </Field>
            )}
            {withKind && (
              <Field label="within">
                <select className="input font-mono text-[11px]" value={withinMs}
                  onChange={(e) => setWithinMs(Number(e.target.value))}>
                  <option value={0}>exactly overlapping</option>
                  <option value={1000}>1s either side</option>
                  <option value={2000}>2s either side</option>
                  <option value={5000}>5s either side</option>
                </select>
              </Field>
            )}
            <button onClick={askTheHardOne}
              className="border border-accent text-accent px-2 py-1 hover:bg-accent/10">
              lane change on red
            </button>
          </div>
        </div>

        {err != null && <LoadState error={err} onRetry={() => void run()} />}

        {res && !err && (
          <>
            {/* The question, in words. A row of dropdowns states a query; a sentence states what was
                actually asked, which is what somebody reads before trusting a count of zero. */}
            <div className="font-mono text-[11px] text-ink-2">
              asking:{" "}
              <span className="text-ink">
                {kind ? kind.split(",").join(" or ") : "any event"}
                {severity ? `, severity ${severity}` : ""}
                {state ? `, ${state}` : ""}
                {city ? `, in ${city}` : ""}
                {withKind ? ` while ${withKind}` : ""}
                {withKind && withState ? ` was ${withState.split(",").join(" or ")}` : ""}
                {withKind && withinMs ? ` (within ${withinMs / 1000}s)` : ""}
              </span>
            </div>
            <div className="font-mono text-[11px] text-ink-3">
              {res.total} match{res.total === 1 ? "" : "es"}
              {res.returned < res.total ? `, showing the first ${res.returned} by confidence` : ""}
              {res.detail ? ` (${res.detail})` : ""}
            </div>

            {res.results.length === 0 ? (
              <div className="panel px-3 py-6 font-mono text-[11px] text-ink-3 text-center">
                Nothing matches. A conjunction that returns nothing is an answer: it means the corpus holds
                no case of the two things happening together, which is worth knowing.
              </div>
            ) : (
              <div className="panel overflow-x-auto">
                <table className="w-full font-mono text-[11px]">
                  <thead className="text-ink-3 text-left">
                    <tr>
                      <th className="px-2 py-1">kind</th><th>severity</th><th>actor</th>
                      <th>city</th><th>vehicle</th><th>at</th><th>for</th><th>conf</th>
                      <th>state</th><th />
                    </tr>
                  </thead>
                  <tbody>
                    {res.results.map((h) => (
                      <tr key={h.event_id} className="border-t border-line hover:bg-line/40">
                        <td className="px-2 py-1">{h.kind}</td>
                        <td className={SEVERITY_TONE[h.severity] ?? ""}>{h.severity}</td>
                        <td className="text-ink-3">{h.actor_class ?? "-"}</td>
                        <td className="text-ink-3">{h.city ?? "-"}</td>
                        <td className="text-ink-3">{h.vehicle_id ?? "-"}</td>
                        <td className="tabular-nums">{h.at_s == null ? "-" : `${h.at_s}s`}</td>
                        <td className="tabular-nums text-ink-3">{h.duration_s}s</td>
                        <td className="tabular-nums">{h.conf == null ? "-" : h.conf.toFixed(2)}</td>
                        <td className="text-ink-3">{h.state}</td>
                        <td className="text-right pr-2">
                          <button onClick={() => open(h)} disabled={!h.frame_id}
                            className="text-info hover:text-accent disabled:opacity-30">open &rarr;</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {busy && !res && <LoadState loading />}
      </div>
    </PageShell>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="font-mono text-[10px] uppercase text-ink-3">{label}</span>
      {children}
    </label>
  );
}

function Stat({ label, value, hint, tone }: {
  label: string; value: number | string; hint?: string; tone?: string;
}) {
  return (
    <div className="panel px-3 py-2 min-w-[110px]">
      <div className="text-[10px] uppercase text-ink-3">{label}</div>
      <div className={`text-[18px] tabular-nums ${tone ?? "text-ink"}`}>{value}</div>
      {hint && <div className="text-[10px] text-ink-3">{hint}</div>}
    </div>
  );
}
