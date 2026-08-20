"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import { getUser } from "@/lib/user";
import type { ActivityEvent, ActivitySummary } from "@/lib/types";

// The activity feed: "what did I do today", and for a lead, what the team did.
//
// Reviews, drawn objects, jobs, and exports each kept their own history and none of them was a timeline, so
// answering that question meant reading five tables and merging them by hand. This is the merge, recorded at
// the moment each thing happens rather than reconstructed at read time.

const WINDOWS: [string, number][] = [["today", 24], ["3 days", 72], ["week", 168], ["month", 720]];

// Relative time is computed from the clock, so the server renders one string and the client renders a
// different one a moment later, which React reports as a hydration mismatch. `mounted` below defers the
// relative form until after hydration; the absolute time is shown first, which is correct rather than a
// placeholder.
function ago(iso: string | null): string {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function ActivityPage() {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [summary, setSummary] = useState<ActivitySummary | null>(null);
  const [mine, setMine] = useState(true);
  const [hours, setHours] = useState(24);
  const [verb, setVerb] = useState("");
  const [mounted, setMounted] = useState(false);
  // getUser reads localStorage, which does not exist during the server render, so reading it in the render
  // body makes the server emit "me" and the client emit the user's name. Deferred to after mount for the
  // same reason the relative times are.
  const me = mounted ? getUser() : null;

  useEffect(() => { setMounted(true); }, []);

  const load = useCallback(async () => {
    try {
      const [feed, sum] = await Promise.all([
        api.activity({ mine, since_hours: hours, verb: verb || undefined, limit: 200 }),
        api.activitySummary(hours, mine),
      ]);
      setEvents(feed.events);
      setTotal(feed.total);
      setSummary(sum);
    } catch (e) { toast(humanizeError(e), "error"); }
  }, [mine, hours, verb]);

  useEffect(() => { load(); }, [load]);

  return (
    <PageShell
      active="ACTIVITY"
      title="Activity"
      subtitle={mine ? "everything you have done, newest first" : "everything the team has done"}
      filters={
        <>
          <div className="flex items-center gap-1">
            {(["mine", "everyone"] as const).map((k) => (
              <button key={k} onClick={() => setMine(k === "mine")}
                className={`px-2 py-0.5 border ${
                  (k === "mine") === mine ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
                {k === "mine" ? (me?.name ? `${me.name}` : "me") : "everyone"}
              </button>
            ))}
          </div>
          <span className="text-ink-3">|</span>
          <div className="flex items-center gap-1">
            {WINDOWS.map(([label, h]) => (
              <button key={h} onClick={() => setHours(h)}
                className={`px-2 py-0.5 border ${
                  hours === h ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
                {label}
              </button>
            ))}
          </div>
          {verb && (
            <button onClick={() => setVerb("")} className="ml-auto text-ink-3 hover:text-accent">
              clear filter: {verb.replace(/_/g, " ")}
            </button>
          )}
        </>
      }
    >
      <div className="p-4 space-y-4 max-w-4xl">
        {summary && (
          <section className="panel p-3">
            <div className="flex items-baseline gap-3 flex-wrap">
              <span className="font-mono text-[22px] text-ink tabular-nums">{summary.total}</span>
              <span className="font-mono text-[11px] text-ink-3">
                {mine ? "things you did" : `things done by ${summary.active_people} people`} in the last
                {" "}{summary.hours >= 24 ? `${Math.round(summary.hours / 24)}d` : `${summary.hours}h`}
              </span>
            </div>
            {Object.keys(summary.by_verb).length > 0 && (
              // The breakdown doubles as the filter, so a count you find interesting is one click from the
              // list behind it.
              <div className="flex gap-1 flex-wrap mt-2">
                {Object.entries(summary.by_verb)
                  .sort((a, b) => b[1] - a[1])
                  .map(([v, n]) => (
                    <button key={v} onClick={() => setVerb(verb === v ? "" : v)}
                      className={`border px-1.5 py-0.5 font-mono text-[10.5px] ${
                        verb === v ? "border-accent text-ink" : "border-line text-ink-2 hover:text-ink"}`}>
                      {summary.labels[v] ?? v} <span className="text-ink-3">{n}</span>
                    </button>
                  ))}
              </div>
            )}
          </section>
        )}

        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            timeline ({events.length} of {total})
          </div>
          {events.length === 0 ? (
            <div className="p-4 font-mono text-[11px] text-ink-3">
              Nothing in this window. Reviews, drawn objects, jobs, exports and sign-ins all land here as
              they happen.
            </div>
          ) : (
            <ul className="divide-y divide-line/40">
              {events.map((e) => (
                <li key={e.event_id} className="px-3 py-2 flex items-start gap-3">
                  <span className="font-mono text-[10px] text-ink-3 w-20 shrink-0 pt-0.5">
                    {mounted ? ago(e.created_at) : (e.created_at?.slice(11, 16) ?? "")}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="font-mono text-[11.5px] text-ink">
                      {!mine && e.user_name && <span className="text-accent">{e.user_name} </span>}
                      {e.summary || e.label}
                    </div>
                    {e.subject_type && (
                      <div className="font-mono text-[9.5px] text-ink-3">
                        {e.subject_type}
                        {e.subject_id ? ` ${String(e.subject_id).slice(0, 8)}` : ""}
                      </div>
                    )}
                  </div>
                  {e.href && (
                    <a href={e.href} className="font-mono text-[10px] text-ink-3 hover:text-accent shrink-0">
                      open
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
