"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
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
    if (payload[k] !== undefined && payload[k] !== null) parts.push(`${k}=${String(payload[k])}`);
  }
  return parts.join("  ");
}

// Event times are absolute capture time. Shown relative to the session origin, because a reviewer scrubs to
// "12.3 seconds in" and cannot do anything with 1782458496.36.
function at(ns: number | null | undefined, originNs: number | null): string {
  if (ns == null) return "-";
  const t = originNs == null ? ns : ns - originNs;
  return `${(t / 1e9).toFixed(2)}s`;
}

function Stat({ label, value, hint, tone }: {
  label: string; value: string | number; hint?: string; tone?: string;
}) {
  return (
    <div className="panel px-3 py-2 min-w-[130px]">
      <div className="font-mono text-[10px] uppercase text-ink-3">{label}</div>
      <div className={`font-mono text-[18px] tabular-nums ${tone ?? "text-ink"}`}>{value}</div>
      {hint && <div className="font-mono text-[10px] text-ink-3">{hint}</div>}
    </div>
  );
}

export default function EventsPage() {
  const [sessionId, setSessionId] = useState("");
  const [taxonomy, setTaxonomy] = useState<EventTaxonomy | null>(null);
  const [summary, setSummary] = useState<DrivingEventSummary | null>(null);
  const [events, setEvents] = useState<DrivingEvent[]>([]);
  const [linkResult, setLinkResult] = useState<LaneLinkResult | null>(null);
  const [originNs, setOriginNs] = useState<number | null>(null);
  const [typeResult, setTypeResult] = useState<LaneTypeResult | null>(null);
  const [coverage, setCoverage] = useState<LaneTypeCoverage | null>(null);
  const [severity, setSeverity] = useState("");
  const [state, setState] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.eventTaxonomy().then(setTaxonomy).catch((e) => toast(humanizeError(e), "error"));
    api.laneTypeCoverage().then(setCoverage).catch(() => setCoverage(null));
  }, []);

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
      toast(`derived ${r.derived}: ${r.inserted} new, ${r.updated} updated, ${r.pruned_stale} stale removed`,
      );
      await load();
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
      const kinds = Object.entries(r.by_type ?? {}).map(([k, n]) => `${k} ${n}`).join(", ");
      toast(`typed ${r.measured ?? 0} lanes: ${kinds || "nothing measurable"}`, "success");
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
      if (sessionId.trim()) setSummary(await api.drivingEventSummary(sessionId.trim()));
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }

  const derivedKinds = (taxonomy?.kinds ?? []).filter((k) => k.derived);
  const humanKinds = (taxonomy?.kinds ?? []).filter((k) => !k.derived);

  return (
    <PageShell active="EVENTS" title="Driving events"
      subtitle="lane changes, signal phases, and the rules they break">
      <div className="p-4 space-y-4 max-w-6xl">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] uppercase text-ink-3">session id</span>
          <input
            className="input font-mono text-[11px] w-[340px]"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void load();
            }}
            placeholder="paste a session uuid"
          />
        </label>
        <button className="btn" onClick={() => void load()} disabled={busy || !sessionId.trim()}>
          load
        </button>
        <button className="btn" onClick={() => void derive()} disabled={busy || !sessionId.trim()}>
          derive
        </button>
        <button className="btn" onClick={() => void linkLanes()} disabled={busy || !sessionId.trim()}>
          link lanes
        </button>
        <button className="btn" onClick={() => void classifyLanes()} disabled={busy || !sessionId.trim()}>
          type lanes
        </button>
      </div>

      <p className="font-mono text-[10px] text-ink-3 max-w-[76ch]">
        Deriving is safe to repeat: a matching candidate updates in place rather than duplicating, anything a
        person has ruled on is left alone, and a candidate the geometry no longer implies is removed. Lane
        linking gives a lane an identity across frames, which is what a crossing is measured against; without
        it every lane exists in one frame only and no crossing can be seen. Typing reads each line off its
        own paint, which is what decides whether crossing it was a manoeuvre or an offence; an unmeasured
        line never makes a crossing a violation, because a type nobody looked at is not evidence.
      </p>

      {coverage && (
        <div className="flex flex-wrap gap-2">
          <Stat label="lanes" value={coverage.total} />
          <Stat
            label="type measured"
            value={coverage.measured}
            hint="the rest carry the old default"
            tone={coverage.measured > 0 ? "text-pass" : "text-warn"}
          />
          <Stat
            label="unmeasured"
            value={coverage.unmeasured}
            tone={coverage.unmeasured > 0 ? "text-warn" : "text-ink-3"}
          />
          {Object.entries(coverage.by_type).map(([kind, v]) => (
            <Stat
              key={kind}
              label={kind}
              value={v.count}
              hint={v.mean_confidence == null ? "never measured" : `conf ${v.mean_confidence.toFixed(2)}`}
            />
          ))}
        </div>
      )}

      {typeResult && (
        <p className="font-mono text-[10px] text-ink-3">
          this session: {typeResult.measured ?? 0} lanes typed over{" "}
          {typeResult.frames_decoded ?? 0} frames, {typeResult.changed_type ?? 0} changed type
          {typeResult.unreadable_frames_lanes
            ? `, ${typeResult.unreadable_frames_lanes} left unmeasured because their frame could not be read`
            : ""}
          .
        </p>
      )}

      {linkResult && (
        <div className="flex flex-wrap gap-2">
          <Stat label="lanes" value={linkResult.lanes} />
          <Stat label="linked" value={linkResult.linked} />
          <Stat label="identities" value={linkResult.identities} />
          <Stat
            label="multi frame"
            value={linkResult.multi_frame_identities}
            hint="only these can show a crossing"
            tone={linkResult.multi_frame_identities > 0 ? "text-pass" : "text-warn"}
          />
        </div>
      )}

      {summary && (
        <div className="flex flex-wrap gap-2">
          <Stat label="events" value={summary.total} />
          <Stat
            label="violations"
            value={summary.violations}
            tone={summary.violations > 0 ? "text-block" : "text-ink-3"}
          />
          <Stat label="awaiting review" value={summary.by_state.review ?? 0} tone="text-warn" />
          <Stat label="confirmed" value={summary.by_state.confirmed ?? 0} tone="text-pass" />
          <Stat label="rejected" value={summary.by_state.rejected ?? 0} />
        </div>
      )}

      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] uppercase text-ink-3">severity</span>
          <select
            className="input font-mono text-[11px]"
            value={severity}
            onChange={(e) => setSeverity(e.target.value)}
          >
            <option value="">any</option>
            {(taxonomy?.severities ?? []).map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] uppercase text-ink-3">state</span>
          <select
            className="input font-mono text-[11px]"
            value={state}
            onChange={(e) => setState(e.target.value)}
          >
            <option value="">any</option>
            <option value="review">review</option>
            <option value="confirmed">confirmed</option>
            <option value="rejected">rejected</option>
          </select>
        </label>
      </div>

      {events.length > 0 && (
        <div className="panel overflow-x-auto">
          <table className="w-full font-mono text-[11px]">
            <thead className="text-ink-3">
              <tr className="text-left">
                <th className="px-2 py-1">kind</th>
                <th className="px-2 py-1">severity</th>
                <th className="px-2 py-1">start</th>
                <th className="px-2 py-1">end</th>
                <th className="px-2 py-1">conf</th>
                <th className="px-2 py-1">detail</th>
                <th className="px-2 py-1">state</th>
                <th className="px-2 py-1" />
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => (
                <tr key={ev.event_id} className="border-t border-line">
                  <td className="px-2 py-1">{ev.kind}</td>
                  <td className={`px-2 py-1 ${SEVERITY_TONE[ev.severity] ?? ""}`}>{ev.severity}</td>
                  <td className="px-2 py-1 tabular-nums">{at(ev.t_start_ns, originNs)}</td>
                  <td className="px-2 py-1 tabular-nums">{at(ev.t_end_ns, originNs)}</td>
                  <td className="px-2 py-1 tabular-nums">
                    {ev.conf == null ? "-" : ev.conf.toFixed(2)}
                  </td>
                  <td className="px-2 py-1 text-ink-3">{detailOf(ev.payload)}</td>
                  <td className={`px-2 py-1 ${STATE_TONE[ev.state] ?? ""}`}>
                    {ev.state}
                    {ev.evidence_changed && (
                      <div className="text-warn text-[10px]" title="the evidence under this ruling changed">
                        evidence moved: now reads {ev.evidence_changed.would_now_be}
                      </div>
                    )}
                  </td>
                  <td className="px-2 py-1 whitespace-nowrap">
                    {ev.state === "review" && (
                      <>
                        <button className="btn btn-sm" onClick={() => void rule(ev, "confirm")}>
                          confirm
                        </button>{" "}
                        <button className="btn btn-sm" onClick={() => void rule(ev, "reject")}>
                          reject
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {summary && events.length === 0 && (
        <p className="font-mono text-[11px] text-ink-3">
          No events match. If the session has never been derived, press derive. If deriving finds nothing,
          the session most likely has lanes and tracked objects on different frames, which is the one thing
          that makes a crossing unobservable.
        </p>
      )}

      {taxonomy && (
        <details className="panel px-3 py-2">
          <summary className="font-mono text-[11px] cursor-pointer">
            vocabulary ({taxonomy.kinds.length} kinds, version {taxonomy.version})
          </summary>
          <div className="mt-2 grid gap-3 md:grid-cols-2">
            <KindList title="derived by geometry" kinds={derivedKinds} />
            <KindList
              title="human only"
              kinds={humanKinds}
              note="no geometry we have can see intent, so nothing proposes these"
            />
          </div>
          <div className="mt-3">
            <div className="font-mono text-[10px] uppercase text-ink-3">signal phase graph</div>
            <div className="font-mono text-[11px]">
              {Object.entries(taxonomy.signal_phase_graph).map(([from, to]) => (
                <div key={from}>
                  {from} to {to.join(", ")}
                </div>
              ))}
            </div>
            <p className="font-mono text-[10px] text-ink-3 mt-1 max-w-[70ch]">
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
      <div className="font-mono text-[10px] uppercase text-ink-3">{title}</div>
      {note && <div className="font-mono text-[10px] text-ink-3">{note}</div>}
      <ul className="mt-1 space-y-1">
        {kinds.map((k) => (
          <li key={k.kind} className="font-mono text-[11px]">
            <span className={SEVERITY_TONE[k.severity] ?? ""}>{k.kind}</span>{" "}
            <span className="text-ink-3">
              {k.shape} on {k.anchor}
            </span>
            <div className="text-ink-3 text-[10px]">{k.description}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
