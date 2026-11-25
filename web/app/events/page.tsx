"use client";

import { useCallback, useEffect, useState } from "react";
import { T } from "@/lib/i18n";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import SessionSelect from "@/components/shell/SessionSelect";
import EmptyState from "@/components/ui/EmptyState";
import { Stat } from "@/components/ui/kit";
import { humanize, num, pct } from "@/lib/format";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
import type {
  DrivingEvent,
  DrivingEventSummary,
  EventKindSpec,
  EventTaxonomy,
  LaneLinkResult,
  LaneTypeCoverage,
  LaneTypeResult,
} from "@/lib/types";

// What happened on the road, as opposed to what was in the frame.
//
// Object labels answer "what is here" and cannot answer "what happened", and most of what a planner is
// trained and evaluated against is the second question. This page is where that second layer is derived,
// inspected and ruled on.
//
// Everything here is a candidate until a person says otherwise. A derived event is geometry's opinion, and
// geometry is confidently wrong often enough that auto-accepting behaviour claims would put unfalsifiable
// records into the training set.

const SEVERITY_TONE: Record<string, string> = {
  info: "text-ink-3",
  notable: "text-warn",
  violation: "text-block",
};

const STATE_TONE: Record<string, string> = {
  review: "text-warn",
  confirmed: "text-pass",
  rejected: "text-ink-3",
};

// What a reviewer needs from a payload, in the order they need it. The rest, mostly raw ids, is available
// on the event itself and only crowds out the fields that say what happened.
const DETAIL_KEYS = [
  "direction", "lane_type", "state", "from_state", "to_state", "reverted_to",
  "crossings", "frames", "mean_abs_offset_px", "is_ego_lane",
];

function detailOf(payload: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const k of DETAIL_KEYS) {
    const v = payload[k];
    if (v === undefined || v === null) continue;
    const shown = typeof v === "number" ? (Number.isInteger(v) ? num(v) : v.toFixed(1))
      : typeof v === "boolean" ? (v ? "yes" : "no") : humanize(String(v)).toLowerCase();
    parts.push(`${humanize(k)}: ${shown}`);
  }
  return parts.join(" · ");
}

// Event times are absolute capture time. Shown relative to the session origin, because a reviewer scrubs to
// "12.3 seconds in" and cannot do anything with 1782458496.36.
function at(ns: number | null | undefined, originNs: number | null): string {
  if (ns == null) return "-";
  const t = originNs == null ? ns : ns - originNs;
  return `${(t / 1e9).toFixed(1)} s`;
}

export default function EventsPage() {
  const [sessionId, setSessionId] = useState("");
  const taxonomyQ = useQuery(["eventTaxonomy"], () => api.eventTaxonomy());
  const taxonomy: EventTaxonomy | null = taxonomyQ.data ?? null;
  const taxonomyErr = taxonomyQ.error;
  useEffect(() => { if (taxonomyErr) toast(humanizeError(taxonomyErr), "error"); }, [taxonomyErr]);
  const [summary, setSummary] = useState<DrivingEventSummary | null>(null);
  const [events, setEvents] = useState<DrivingEvent[]>([]);
  const [linkResult, setLinkResult] = useState<LaneLinkResult | null>(null);
  const [originNs, setOriginNs] = useState<number | null>(null);
  const [typeResult, setTypeResult] = useState<LaneTypeResult | null>(null);
  // Absent when it cannot be read, as before: a failure reads as no coverage figure rather than an error.
  const coverageQ = useQuery(["laneTypeCoverage"], () => api.laneTypeCoverage());
  const coverage: LaneTypeCoverage | null = coverageQ.error ? null : coverageQ.data ?? null;
  const setCoverage = coverageQ.setData;
  const [severity, setSeverity] = useState("");
  const [state, setState] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    if (!sessionId.trim()) return;
    setBusy(true);
    try {
      const [s, list] = await Promise.all([
        api.drivingEventSummary(sessionId.trim()),
        api.drivingEvents(sessionId.trim(), {
          severity: severity || undefined,
          state: state || undefined,
          limit: 500,
        }),
      ]);
      setSummary(s);
      setEvents(list.events);
      setOriginNs(list.session_start_ns ?? null);
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  }, [sessionId, severity, state]);

  useEffect(() => {
    if (summary) void load();
    // Refetch when a filter changes, but only once a session has been loaded at least once, so typing an
    // id does not fire a request per keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [severity, state]);

  async function derive() {
    if (!sessionId.trim()) return;
    setBusy(true);
    try {
      const r = await api.deriveDrivingEvents(sessionId.trim());
      toast(`Derived ${r.derived}: ${r.inserted} new, ${r.updated} updated, ${r.pruned_stale} stale removed`,
      );
      await load();
      // Deriving adds and removes events corpus-wide, as far as the event search is concerned.
      void invalidate("searchEvents", "eventCorpusSummary");
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  }

  async function linkLanes() {
    if (!sessionId.trim()) return;
    setBusy(true);
    try {
      const r = await api.linkSessionLanes(sessionId.trim());
      setLinkResult(r);
      toast(`${r.linked} lanes linked into ${r.identities} identities, ${r.multi_frame_identities} spanning more than one frame`,
      );
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  }

  async function classifyLanes() {
    if (!sessionId.trim()) return;
    setBusy(true);
    try {
      const r = await api.classifySessionLanes(sessionId.trim(), { reclassify: true });
      setTypeResult(r);
      const kinds = Object.entries(r.by_type ?? {}).map(([k, n]) => `${humanize(k).toLowerCase()} ${n}`).join(", ");
      toast(`Typed ${r.measured ?? 0} lanes: ${kinds || "nothing measurable"}`, "success");
      setCoverage(await api.laneTypeCoverage());
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  }

  async function rule(ev: DrivingEvent, verdict: "confirm" | "reject") {
    try {
      const updated = await api.ruleOnDrivingEvent(ev.event_id, verdict);
      setEvents((prev) => prev.map((e) => (e.event_id === updated.event_id ? updated : e)));
      // A verdict changes an event's state, which the corpus search and its summary both filter on.
      void invalidate("searchEvents", "eventCorpusSummary");
      if (sessionId.trim()) setSummary(await api.drivingEventSummary(sessionId.trim()));
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }

  const derivedKinds = (taxonomy?.kinds ?? []).filter((k) => k.derived);
  const humanKinds = (taxonomy?.kinds ?? []).filter((k) => !k.derived);
  const ready = Boolean(sessionId.trim());

  return (
    <PageShell active="EVENTS" title="Driving events"
      subtitle="Lane changes, signal phases and the traffic rules they break, derived from a session and ruled on by a person.">
      <div className="page-wrap space-y-4">
      <section className="panel p-4 space-y-3">
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-1.5">
            <span className="label-caps">Session</span>
            <SessionSelect value={sessionId} onChange={setSessionId} className="w-[340px]" />
          </label>
          <button className="btn btn-primary" onClick={() => void load()} disabled={busy || !ready}>
            Load events
          </button>
          <span className="w-px h-6 bg-line mx-1" aria-hidden />
          <button className="btn" onClick={() => void derive()} disabled={busy || !ready}
            data-tip="Find lane crossings and signal changes in this session. Safe to repeat.">
            Derive events
          </button>
          <button className="btn" onClick={() => void linkLanes()} disabled={busy || !ready}
            data-tip="Give each lane one identity across frames, so a crossing can be measured">
            Link lanes
          </button>
          <button className="btn" onClick={() => void classifyLanes()} disabled={busy || !ready}
            data-tip="Read each lane line's paint (solid, dashed, double) off the image">
            Type lanes
          </button>
        </div>
        <details className="text-[12px] text-ink-3">
          <summary className="cursor-pointer text-ink-2 select-none">How this works</summary>
          <div className="mt-2 space-y-1.5 max-w-[80ch] leading-relaxed">
            <p>Deriving is safe to repeat: a matching candidate updates in place rather than duplicating, anything a
              person has ruled on is left alone, and a candidate the geometry no longer implies is removed.</p>
            <p>Lane linking gives a lane an identity across frames, which is what a crossing is measured against;
              without it every lane exists in one frame only and no crossing can be seen.</p>
            <p>Typing reads each line off its own paint, which decides whether crossing it was a manoeuvre or an
              offence. An unmeasured line never makes a crossing a violation.</p>
            <p>Every derived event is a candidate until a person confirms or rejects it.</p>
          </div>
        </details>
      </section>

      {coverage && (
        <section className="space-y-2">
          <h2 className="section-title">Lane lines across all sessions</h2>
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8">
            <Stat label="Lanes" value={num(coverage.total)} />
            <Stat
              label="Type measured"
              value={num(coverage.measured)}
              sub={coverage.total ? `${pct(coverage.measured / coverage.total)} of lanes` : undefined}
              tone={coverage.measured > 0 ? "pass" : "warn"}
            />
            <Stat
              label="Unmeasured"
              value={num(coverage.unmeasured)}
              tone={coverage.unmeasured > 0 ? "warn" : "muted"}
            />
            {Object.entries(coverage.by_type).map(([kind, v]) => (
              <Stat
                key={kind}
                label={humanize(kind)}
                value={num(v.count)}
                sub={v.mean_confidence == null ? "never measured" : `${pct(v.mean_confidence)} mean confidence`}
              />
            ))}
          </div>
        </section>
      )}

      {typeResult && (
        <p className="text-[12px] text-ink-2">
          This session: {num(typeResult.measured ?? 0)} lanes typed over{" "}
          {num(typeResult.frames_decoded ?? 0)} frames, {num(typeResult.changed_type ?? 0)} changed type
          {typeResult.unreadable_frames_lanes
            ? `, ${num(typeResult.unreadable_frames_lanes)} left unmeasured because their frame could not be read`
            : ""}
          .
        </p>
      )}

      {linkResult && (
        <section className="space-y-2">
          <h2 className="section-title">Lane linking, this session</h2>
          <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
            <Stat label="Lanes" value={num(linkResult.lanes)} />
            <Stat label="Linked" value={num(linkResult.linked)} />
            <Stat label="Identities" value={num(linkResult.identities)} />
            <Stat
              label="Multi frame"
              value={num(linkResult.multi_frame_identities)}
              sub="only these can show a crossing"
              tone={linkResult.multi_frame_identities > 0 ? "pass" : "warn"}
            />
          </div>
        </section>
      )}

      {summary && (
        <section className="space-y-2">
          <h2 className="section-title">Events in this session</h2>
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
            <Stat label="Events" value={num(summary.total)} />
            <Stat
              label="Violations"
              value={num(summary.violations)}
              tone={summary.violations > 0 ? "block" : "muted"}
            />
            <Stat label="Awaiting review" value={num(summary.by_state.review ?? 0)} tone="warn" />
            <Stat label="Confirmed" value={num(summary.by_state.confirmed ?? 0)} tone="pass" />
            <Stat label="Rejected" value={num(summary.by_state.rejected ?? 0)} />
          </div>
        </section>
      )}

      {summary ? (
      <section className="panel">
        <div className="panel-head flex-wrap gap-3">
          <h2 className="section-title">Event list</h2>
          <div className="ml-auto flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-2">
              <span className="text-[12px] text-ink-3">Severity</span>
              <select
                className="field text-[12px]"
                value={severity}
                onChange={(e) => setSeverity(e.target.value)}
              >
                <option value="">Any</option>
                {(taxonomy?.severities ?? []).map((s) => (
                  <option key={s} value={s}>
                    {humanize(s)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span className="text-[12px] text-ink-3">State</span>
              <select
                className="field text-[12px]"
                value={state}
                onChange={(e) => setState(e.target.value)}
              >
                <option value="">Any</option>
                <option value="review">Awaiting review</option>
                <option value="confirmed">Confirmed</option>
                <option value="rejected">Rejected</option>
              </select>
            </label>
          </div>
        </div>

      {events.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Event</th>
                <th>Severity</th>
                <th className="num">Start</th>
                <th className="num">End</th>
                <th className="num">Confidence</th>
                <th>Detail</th>
                <th>State</th>
                <th><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => (
                <tr key={ev.event_id}>
                  <td className="text-ink">{humanize(ev.kind)}</td>
                  <td className={SEVERITY_TONE[ev.severity] ?? ""}>{humanize(ev.severity)}</td>
                  <td className="num">{at(ev.t_start_ns, originNs)}</td>
                  <td className="num">{at(ev.t_end_ns, originNs)}</td>
                  <td className="num">
                    {ev.conf == null ? "-" : pct(ev.conf)}
                  </td>
                  <td className="text-ink-3 text-[12px]">{detailOf(ev.payload)}</td>
                  <td className={STATE_TONE[ev.state] ?? ""}>
                    {ev.state === "review" ? "Awaiting review" : humanize(ev.state)}
                    {ev.evidence_changed && (
                      <div className="text-warn text-[11px]" data-tip="The evidence under this ruling changed">
                        Evidence moved: now reads {humanize(ev.evidence_changed.would_now_be)}
                      </div>
                    )}
                  </td>
                  <td className="whitespace-nowrap text-right">
                    {ev.state === "review" && (
                      <span className="inline-flex gap-1.5">
                        <button className="btn" onClick={() => void rule(ev, "confirm")}>
                          <T k="action.confirm_short">Confirm</T>
                        </button>
                        <button className="btn" onClick={() => void rule(ev, "reject")}>
                          <T k="action.reject">Reject</T>
                        </button>
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState compact icon="activity" title="No events match"
          action={<button className="btn" onClick={() => void derive()} disabled={busy || !ready}>Derive events</button>}>
          If this session has never been derived, derive it. If deriving finds nothing, the session most likely has
          lanes and tracked objects on different frames, which makes a crossing impossible to see.
        </EmptyState>
      )}
      </section>
      ) : (
        <div className="panel">
          <EmptyState icon="activity" title="Choose a session to see its driving events">
            Pick a session above and load it. Derive events first if it has never been processed.
          </EmptyState>
        </div>
      )}

      {taxonomy && (
        <details className="panel px-4 py-3">
          <summary className="text-[13px] cursor-pointer text-ink-2 select-none">
            Event vocabulary ({taxonomy.kinds.length} kinds, version {taxonomy.version})
          </summary>
          <div className="mt-3 grid gap-4 md:grid-cols-2">
            <KindList title="Derived by geometry" kinds={derivedKinds} />
            <KindList
              title="Human only"
              kinds={humanKinds}
              note="No geometry we have can see intent, so nothing proposes these."
            />
          </div>
          <div className="mt-4">
            <div className="section-title">Signal phase graph</div>
            <div className="text-[12px] mt-1 space-y-0.5">
              {Object.entries(taxonomy.signal_phase_graph).map(([from, to]) => (
                <div key={from}>
                  {humanize(from)} to {to.map(humanize).join(", ")}
                </div>
              ))}
            </div>
            <p className="section-desc mt-1.5 max-w-[70ch]">
              A transition outside this graph is almost never a broken signal. It is a mislabelled frame, and
              it is invisible frame by frame because each label looks reasonable on its own crop.
            </p>
          </div>
        </details>
      )}
      </div>
    </PageShell>
  );
}

function KindList({ title, kinds, note }: { title: string; kinds: EventKindSpec[]; note?: string }) {
  return (
    <div>
      <div className="section-title">{title}</div>
      {note && <div className="section-desc">{note}</div>}
      <ul className="mt-2 space-y-2">
        {kinds.map((k) => (
          <li key={k.kind} className="text-[12.5px]">
            <span className={SEVERITY_TONE[k.severity] ?? "text-ink"}>{humanize(k.kind)}</span>{" "}
            <span className="text-ink-3">
              {humanize(k.shape).toLowerCase()} on {humanize(k.anchor).toLowerCase()}
            </span>
            <div className="text-ink-3 text-[12px]">{k.description}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
