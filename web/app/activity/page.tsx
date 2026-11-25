"use client";

import { useEffect, useMemo, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import { useQuery } from "@/lib/useQuery";
import { getUser } from "@/lib/user";
import type { ActivityEvent, ActivitySummary } from "@/lib/types";
import { groupActivity, verbLabel, windowName } from "@/lib/activityFeed";
import { ago, humanize, num, plural, shortId } from "@/lib/format";
import { Segmented } from "@/components/ui/kit";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";

// The activity feed: "what did I do today", and for a lead, what the team did.
//
// Reviews, drawn objects, jobs, and exports each kept their own history and none of them was a timeline, so
// answering that question meant reading five tables and merging them by hand. This is the merge, recorded at
// the moment each thing happens rather than reconstructed at read time.

const WINDOWS: [string, number][] = [["Today", 24], ["3 days", 72], ["Week", 168], ["Month", 720]];

// Relative time is computed from the clock, so the server renders one string and the client renders a
// different one a moment later, which React reports as a hydration mismatch. `mounted` below defers the
// relative form until after hydration; the absolute time is shown first, which is correct rather than a
// placeholder.

const NO_EVENTS: ActivityEvent[] = [];

export default function ActivityPage() {
  const [mine, setMine] = useState(true);
  const [hours, setHours] = useState(24);
  const [verb, setVerb] = useState("");
  const [mounted, setMounted] = useState(false);
  // getUser reads localStorage, which does not exist during the server render, so reading it in the render
  // body makes the server emit "me" and the client emit the user's name. Deferred to after mount for the
  // same reason the relative times are.
  const me = mounted ? getUser() : null;

  useEffect(() => { setMounted(true); }, []);

  // keepPrevious: changing a filter leaves the last feed up until the new one arrives.
  const feedArgs = { mine, since_hours: hours, verb: verb || undefined, limit: 200 };
  const feedQ = useQuery(["activity", feedArgs], () => api.activity(feedArgs), { keepPrevious: true });
  const sumQ = useQuery(["activitySummary", hours, mine], () => api.activitySummary(hours, mine),
    { keepPrevious: true });
  const events: ActivityEvent[] = feedQ.data?.events ?? NO_EVENTS;
  const total = feedQ.data?.total ?? 0;
  const summary: ActivitySummary | null = sumQ.data ?? null;
  const failure = feedQ.error ?? sumQ.error;
  useEffect(() => { if (failure) toast(humanizeError(failure), "error"); }, [failure]);

  const lines = useMemo(() => groupActivity(events), [events]);
  const when = (iso: string | null) => (mounted ? ago(iso) : (iso?.slice(11, 16) ?? ""));
  const stamp = (iso: string | null) => (iso ? new Date(iso).toLocaleString("en-GB") : "");

  return (
    <PageShell
      active="ACTIVITY"
      title="Activity"
      subtitle={mine ? "Everything you have done, newest first" : "Everything the team has done, newest first"}
      filters={
        <>
          <Segmented
            value={mine ? "mine" : "everyone"}
            onChange={(v) => setMine(v === "mine")}
            options={[
              { value: "mine", label: me?.name ? `Only ${me.name}` : "Only me" },
              { value: "everyone", label: "Everyone" },
            ]} />
          <Segmented
            value={String(hours)}
            onChange={(v) => setHours(Number(v))}
            options={WINDOWS.map(([label, h]) => ({ value: String(h), label }))} />
          {verb && (
            <button type="button" onClick={() => setVerb("")} className="chip chip-on ml-auto"
              aria-label="Clear the action filter">
              {verbLabel(verb, summary?.labels[verb])} <span aria-hidden>×</span>
            </button>
          )}
        </>
      }
    >
      <div className="page-wrap-narrow space-y-4">
        {summary && (
          <section className="panel px-4 py-3.5">
            <div className="flex items-baseline gap-2.5 flex-wrap">
              <span className="text-[24px] leading-none font-semibold text-ink tabular-nums">{num(summary.total)}</span>
              <span className="text-[13px] text-ink-2">
                {mine ? "actions by you" : `actions by ${plural(summary.active_people, "person", "people")}`} in the
                {" "}last {windowName(summary.hours)}
              </span>
            </div>
            {Object.keys(summary.by_verb).length > 0 && (
              // The breakdown doubles as the filter, so a count you find interesting is one click from the
              // list behind it.
              <div className="flex gap-1.5 flex-wrap mt-3">
                {Object.entries(summary.by_verb)
                  .sort((a, b) => b[1] - a[1])
                  .map(([v, n]) => (
                    <button key={v} type="button" onClick={() => setVerb(verb === v ? "" : v)}
                      aria-pressed={verb === v}
                      className={`chip ${verb === v ? "chip-on" : ""}`}>
                      {verbLabel(v, summary.labels[v])} <span className="text-ink-3 tabular-nums">{num(n)}</span>
                    </button>
                  ))}
              </div>
            )}
          </section>
        )}

        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Timeline</h2>
            {events.length > 0 && (
              <span className="ml-auto text-[12px] text-ink-3">
                {events.length < total ? `Latest ${num(events.length)} of ${num(total)}` : plural(total, "action")}
              </span>
            )}
          </div>
          {!feedQ.data || feedQ.error ? (
            <div className="p-3"><LoadState {...feedQ.loadState} rows={6} /></div>
          ) : lines.length === 0 ? (
            <EmptyState compact icon="activity" title={verb ? "Nothing of this kind in this window" : "Nothing in this window"}
              action={hours < 720 ? <button type="button" className="btn" onClick={() => setHours(720)}>Show the last month</button> : undefined}>
              Reviews, drawn objects, jobs, exports and sign-ins land here as they happen.
            </EmptyState>
          ) : (
            <ul className="divide-y divide-line/40">
              {lines.map((l) => (
                <li key={l.key} className="px-4 py-2.5 flex items-start gap-4">
                  <span className="text-[12px] text-ink-3 w-20 shrink-0 pt-px tabular-nums" title={stamp(l.first.created_at)}>
                    {when(l.first.created_at)}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] text-ink">
                      {!mine && l.user && <span className="font-medium text-accent-2">{l.user} </span>}
                      {!mine && l.user ? l.text[0].toLowerCase() + l.text.slice(1) : l.text}
                    </div>
                    <Meta line={l} when={when} />
                  </div>
                  {l.first.href && (
                    <a href={l.first.href} className="link text-[12px] shrink-0 pt-px"
                      data-tip={l.count > 1 ? "Opens the most recent one" : undefined}>
                      Open
                    </a>
                  )}
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </PageShell>
  );
}

// The line under a sentence: what it touched. For a single event the subject and its id; for a run, how many
// and over how long, since the sentence already carries the class.
function Meta({ line, when }: { line: ReturnType<typeof groupActivity>[number]; when: (iso: string | null) => string }) {
  const e = line.first;
  if (line.count > 1) {
    const span = when(line.last.created_at);
    const since = span && span !== when(line.first.created_at) ? `, since ${span}` : "";
    return (
      <div className="text-[11.5px] text-ink-3 mt-0.5">
        {line.frames > 1 ? `Across ${plural(line.frames, "frame")}` : `${plural(line.count, "action")}`}{since}
      </div>
    );
  }
  if (!e.subject_type) return null;
  return (
    <div className="text-[11.5px] text-ink-3 mt-0.5">
      {humanize(e.subject_type)}
      {e.subject_id && <span className="font-mono ml-1.5">{shortId(String(e.subject_id))}</span>}
    </div>
  );
}
