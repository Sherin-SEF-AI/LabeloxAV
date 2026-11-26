"use client";

// Paths and intent (#8). Pick a session; see the ego path and every road user's path top-down; correct the ego
// path; draw where a road user goes next, starting from a kinematic baseline (constant turn rate and velocity:
// there is no learned forecaster, and the page says so); tag what they meant to do over a span of frames; export
// the lot as a scenario.
//
// Labels are stored beside the measurement: an edited ego path never overwrites ego_pose, a future path is a
// path_label, and an intent is a track_event (mirrored onto Track.intents for the governed intent words).

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { Badge, Button, KV, Panel } from "@/components/ui/kit";
import { api, humanizeError } from "@/lib/api";
import { fromNs, humanize, plural, shortId } from "@/lib/format";
import { useQuery } from "@/lib/useQuery";
import {
  addTrackIntent, deletePathLabel, deleteTrackIntent, downloadText, pathSession, pathSessions, saveEgoPath, saveFuture,
  scenarioExport, smoothEgo, trackForecast, trackIntents, type Forecast, type PathSession, type PathTrack,
} from "@/lib/tools/drivingApi";
import type { PlotPath } from "./PathPlot";

const PathPlot = dynamic(() => import("./PathPlot"), { ssr: false });

const NO_TRACKS: PathTrack[] = [];
const NO_SESSIONS: PathSession[] = [];

export default function TrajectoriesPage() {
  const sessionsQ = useQuery(["drivingPathSessions"], () => pathSessions());
  const sessions = sessionsQ.data ?? NO_SESSIONS;
  const [sid, setSid] = useState("");
  useEffect(() => {
    // Start on a session with ego poses (a world frame) and a track count a person can read; the busiest
    // sessions hold thousands of tracks and are better picked on purpose.
    if (!sid && sessions.length) {
      setSid((sessions.find((s) => s.ego_poses > 0 && s.tracks <= 1500) ?? sessions.find((s) => s.ego_poses > 0) ?? sessions[0]).session_id);
    }
  }, [sessions, sid]);
  const viewQ = useQuery(sid ? ["drivingPathSession", sid] : null, () => pathSession(sid), { keepPrevious: true });
  const v = viewQ.data && viewQ.data.session_id === sid ? viewQ.data : null;
  const tracks = v?.tracks ?? NO_TRACKS;

  const [tid, setTid] = useState<string | null>(null);
  const track = tracks.find((t) => t.track_id === tid) ?? null;
  const [anchor, setAnchor] = useState(0);
  const [fc, setFc] = useState<Forecast | null>(null);
  const [future, setFuture] = useState<number[][] | null>(null);
  const [futureEdited, setFutureEdited] = useState(false);
  const [egoDraft, setEgoDraft] = useState<number[][] | null>(null);
  const [egoSource, setEgoSource] = useState<"human" | "smoothed">("human");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [frameImg, setFrameImg] = useState<string | null>(null);

  useEffect(() => { setTid(null); setFc(null); setFuture(null); setEgoDraft(null); }, [sid]);
  useEffect(() => {
    setFc(null); setFuture(null); setFutureEdited(false);
    if (track) setAnchor(Math.max(1, Math.floor(track.n * 0.6)));
  }, [tid]); // eslint-disable-line react-hooks/exhaustive-deps

  const anchorFrame = track ? track.frames[Math.min(anchor, track.frames.length - 1)] : null;
  useEffect(() => {
    setFrameImg(null);
    if (!anchorFrame) return;
    let live = true;
    api.frame(anchorFrame).then((m) => { if (live) setFrameImg(m.image_url); })
      .catch((e) => { if (live) setMsg(`Could not load the frame image: ${humanizeError(e)}`); });
    return () => { live = false; };
  }, [anchorFrame]);

  const run = async (what: string, fn: () => Promise<void>) => {
    setBusy(what);
    try { await fn(); } catch (e) { setMsg(`${what} failed: ${humanizeError(e)}`); } finally { setBusy(null); }
  };

  // Times in the plot are seconds from the session's t0; a forecast's are from its anchor.
  const anchorT = track ? track.path[Math.min(anchor, track.path.length - 1)][0] : 0;
  const forecastAbs = (pts: number[][]) => pts.map((p) => [p[0] + anchorT, p[1], p[2]]);
  const forecastRel = (pts: number[][]) => pts.map((p) => [+(p[0] - anchorT).toFixed(4), p[1], p[2]]);

  const getBaseline = () => run("Baseline", async () => {
    if (!track || !anchorFrame) return;
    const f = await trackForecast(track.track_id, anchorFrame, 3, 0.25);
    setFc(f);
    setFuture(forecastAbs(f.forecast));
    setFutureEdited(false);
    setMsg(null);
  });

  const egoPoints = useMemo(() => {
    if (!v) return [];
    if (v.ego.label) {
      const off = (v.ego.label.anchor_ts_ns - v.t0_ns) / 1e9;
      return v.ego.label.points.map((p) => [p[0] + off, p[1], p[2]]);
    }
    return v.ego.poses.map((p) => [(p.ts_ns - v.t0_ns) / 1e9, p.x, p.y]);
  }, [v]);

  const paths: PlotPath[] = useMemo(() => {
    const out: PlotPath[] = [];
    if (!v) return out;
    for (const t of tracks) if (t.track_id !== tid) out.push({ id: t.track_id, points: t.path, kind: "track" });
    if (egoDraft) out.push({ id: "ego-edit", points: egoDraft, kind: "ego-edit" });
    else if (egoPoints.length) out.push({ id: "ego", points: egoPoints, kind: "ego" });
    if (track) {
      out.push({ id: track.track_id, points: track.path.slice(0, anchor + 1), kind: "track-selected" });
      out.push({ id: `${track.track_id}-after`, points: track.path.slice(anchor), kind: "observed-after" });
      for (const f of track.futures) {
        const off = (f.anchor_ts_ns - v.t0_ns) / 1e9;
        out.push({ id: f.path_id, points: f.points.map((p) => [p[0] + off, p[1], p[2]]), kind: "future" });
      }
      if (fc) out.push({ id: "baseline", points: forecastAbs(fc.forecast), kind: "baseline" });
      if (future) out.push({ id: "future-edit", points: future, kind: "future-edit" });
    }
    return out;
  }, [v, tracks, tid, track, anchor, fc, future, egoDraft, egoPoints]); // eslint-disable-line react-hooks/exhaustive-deps

  const fitTo = useMemo(() => {
    if (track) return [track.path, ...(future ? [future] : [])];
    if (egoDraft) return [egoDraft];
    return undefined;
  }, [track, future, egoDraft]);

  const editable = future ? { id: "future", points: future }
    : egoDraft ? { id: "ego", points: egoDraft, every: Math.max(1, Math.ceil(egoDraft.length / 120)) } : null;

  const shown = tracks.filter((t) => !filter || t.class_name.includes(filter.toLowerCase()) || t.track_id.startsWith(filter));

  return (
    <PageShell active="Paths and intent" filters={(
      <>
        <label htmlFor="paths-session" className="text-ink-3">Session</label>
        <select id="paths-session" className="field text-[12.5px] max-w-[26rem]" value={sid} onChange={(e) => setSid(e.target.value)}>
          {sessions.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.route || s.vehicle_id || shortId(s.session_id)} · {plural(s.tracks, "track")} · {s.ego_poses ? `${s.ego_poses} poses` : "no poses"}
            </option>
          ))}
        </select>
        {v && <Badge state={v.coord_frame === "world" ? "pass" : "warn"}>{v.coord_frame === "world" ? "World frame" : "Ego frame"}</Badge>}
        {v && (
          <>
            <Button tip="The labelled paths, futures and intents as a documented JSON scenario" disabled={!!busy}
              onClick={() => run("Export", async () => {
                const r = await scenarioExport(sid);
                downloadText(`scenario_${sid.slice(0, 8)}.json`, JSON.stringify(r.json, null, 2), "application/json");
                setMsg(`Exported ${plural(r.actors, "actor")} and ${plural(r.events, "intent")}`);
              })}>Scenario JSON</Button>
            <Button tip="The observed paths replayed as an ASAM OpenSCENARIO 1.2 file" disabled={!!busy}
              onClick={() => run("Export", async () => {
                const r = await scenarioExport(sid);
                downloadText(`scenario_${sid.slice(0, 8)}.xosc`, r.openscenario);
              })}>OpenSCENARIO</Button>
          </>
        )}
      </>
    )}>
      <div className="page-wrap">
        <LoadState loading={sessionsQ.loading && !sessions.length} error={sessionsQ.error} empty={!sessionsQ.loading && !sessions.length}
          emptyTitle="No tracked sessions" emptyText="Paths come from tracks. Run tracking on a session first." emptyIcon="route">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <div className="space-y-3 min-w-0">
              <Panel title="Top-down" desc={v?.frame_note ?? "Loading session..."}
                right={<span className="text-[11px] text-ink-3">Scroll to zoom, drag to pan, double-click to reset</span>}>
                <LoadState loading={!v && viewQ.loading} error={viewQ.error} onRetry={viewQ.reload}>
                  {v && (
                    <>
                      <PathPlot paths={paths} fitTo={fitTo} editable={editable} onPick={(id) => setTid(id)}
                        marker={track ? track.path[Math.min(anchor, track.path.length - 1)] : null}
                        onEdit={(pts) => {
                          if (future) { setFuture(pts); setFutureEdited(true); } else { setEgoDraft(pts); setEgoSource("human"); }
                        }} />
                      <div className="flex flex-wrap gap-3 mt-2 text-[11.5px] text-ink-3">
                        <span><span className="inline-block w-4 h-0.5 align-middle mr-1" style={{ background: "rgb(var(--c-pass))" }} />Ego</span>
                        <span><span className="inline-block w-4 h-0.5 align-middle mr-1" style={{ background: "rgb(var(--c-accent))" }} />Selected track</span>
                        <span><span className="inline-block w-4 h-0.5 align-middle mr-1" style={{ background: "rgb(var(--c-info))" }} />Saved future</span>
                        <span><span className="inline-block w-4 h-0.5 align-middle mr-1" style={{ background: "rgb(var(--c-warn))" }} />Being edited</span>
                        <span><span className="inline-block w-4 h-0.5 align-middle mr-1" style={{ background: "rgb(var(--c-ink-2))" }} />Kinematic baseline</span>
                        {v.positions_beyond_range > 0 && <span className="text-warn">{v.positions_beyond_range} positions beyond the camera&apos;s ground range are left out</span>}
                        {v.calibration.includes("nominal") && <span className="text-warn">Nominal calibration: distances are approximate</span>}
                      </div>
                    </>
                  )}
                </LoadState>
              </Panel>

              {v && (
                <Panel title="Ego path" desc={v.ego.poses.length
                  ? `${plural(v.ego.poses.length, "pose")} (${v.ego.poses[0].source}${v.ego.poses[0].measured ? ", measured" : ", estimated"}).`
                    + (v.ego.label ? ` Corrected path saved (${humanize(v.ego.label.source)}).` : "")
                  : "This session has no ego poses, so there is no ego path to correct."}>
                  {v.ego.poses.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 items-center">
                      {!egoDraft ? (
                        <>
                          <Button onClick={() => { setTid(null); setEgoDraft(egoPoints.map((p) => [...p])); setEgoSource("human"); }}
                            tip="Drag the handles on the plot to move the path">Edit ego path</Button>
                          <Button disabled={!!busy} tip="Savitzky-Golay smoothing of the measured poses, endpoints fixed"
                            onClick={() => run("Smooth", async () => {
                              const r = await smoothEgo(sid, 7);
                              const off = (r.anchor_ts_ns - v.t0_ns) / 1e9;
                              setTid(null);
                              setEgoDraft(r.points.map((p) => [p[0] + off, p[1], p[2]]));
                              setEgoSource("smoothed");
                            })}>Smooth</Button>
                        </>
                      ) : (
                        <>
                          <Button variant="primary" disabled={!!busy} onClick={() => run("Save", async () => {
                            const t0 = (v.ego.poses[0].ts_ns - v.t0_ns) / 1e9;
                            await saveEgoPath(sid, { points: egoDraft.map((p) => [+(p[0] - t0).toFixed(4), p[1], p[2]]), source: egoSource });
                            setEgoDraft(null);
                            await viewQ.reload();
                            setMsg("Ego path saved beside the measured poses");
                          })}>Save ego path</Button>
                          <Button onClick={() => setEgoDraft(null)}>Discard</Button>
                          <span className="text-[12px] text-ink-3">{egoSource === "smoothed" ? "Smoothed; drag to adjust further." : "Drag the handles."}</span>
                        </>
                      )}
                    </div>
                  )}
                </Panel>
              )}
            </div>

            <div className="space-y-3 min-w-0">
              {track ? (
                <TrackPanel track={track} anchor={anchor} setAnchor={(i) => { setAnchor(i); setFc(null); setFuture(null); }}
                  frameImg={frameImg} fc={fc} future={future} futureEdited={futureEdited} busy={busy}
                  baselineNote={v?.baseline_note ?? ""}
                  onBack={() => setTid(null)} onBaseline={getBaseline}
                  onSaveFuture={() => run("Save", async () => {
                    if (!future || !anchorFrame) return;
                    await saveFuture(track.track_id, { anchor_frame_id: anchorFrame, points: forecastRel(future),
                      source: futureEdited ? "human" : "baseline_ctrv", seed: fc ? { model: "ctrv", state: fc.state } : undefined });
                    setFuture(null); setFc(null);
                    await viewQ.reload();
                    setMsg(futureEdited ? "Future path saved" : "Baseline saved as the future path, unedited");
                  })}
                  onDiscardFuture={() => { setFuture(null); setFutureEdited(false); }}
                  onDeleteFuture={(id) => run("Delete", async () => { await deletePathLabel(id); await viewQ.reload(); })}
                  onEventsChanged={() => { void viewQ.reload(); }} setMsg={setMsg} />
              ) : (
                <Panel title={`Road users${v ? ` (${tracks.length})` : ""}`} desc="Tracks of things that move, with at least three positions on the road.">
                  <input className="field w-full mb-2" placeholder="Filter by class or id" value={filter} onChange={(e) => setFilter(e.target.value)}
                    aria-label="Filter tracks" />
                  {v && !tracks.length && <div className="text-[12px] text-ink-3">No movable track in this session has a path on the road.</div>}
                  <div className="max-h-[60vh] overflow-y-auto">
                    <table className="data-table w-full text-[12px]">
                      <thead><tr><th>Class</th><th className="text-right">Positions</th><th className="text-right">Intents</th><th className="text-right">Futures</th></tr></thead>
                      <tbody>
                        {shown.slice(0, 300).map((t) => (
                          <tr key={t.track_id} className="cursor-pointer hover:bg-head" onClick={() => setTid(t.track_id)}>
                            <td><span className="link">{humanize(t.class_name)}</span> <span className="text-ink-3 font-mono">{shortId(t.track_id)}</span></td>
                            <td className="text-right">{t.n}</td>
                            <td className="text-right">{t.events || ""}</td>
                            <td className="text-right">{t.futures.length || ""}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>
              )}
              {msg && <div role="status" className={`text-[12px] ${/failed/i.test(msg) ? "text-warn" : "text-ink-2"}`}>{msg}</div>}
            </div>
          </div>
        </LoadState>
      </div>
    </PageShell>
  );
}

function TrackPanel({ track, anchor, setAnchor, frameImg, fc, future, futureEdited, busy, baselineNote, onBack, onBaseline,
  onSaveFuture, onDiscardFuture, onDeleteFuture, onEventsChanged, setMsg }: {
  track: PathTrack; anchor: number; setAnchor: (i: number) => void; frameImg: string | null; fc: Forecast | null;
  future: number[][] | null; futureEdited: boolean; busy: string | null; baselineNote: string; onBack: () => void;
  onBaseline: () => void; onSaveFuture: () => void; onDiscardFuture: () => void; onDeleteFuture: (id: string) => void;
  onEventsChanged: () => void; setMsg: (m: string | null) => void;
}) {
  const intentsQ = useQuery(["drivingTrackIntents", track.track_id], () => trackIntents(track.track_id));
  const it = intentsQ.data;
  const [span, setSpan] = useState<[number, number]>([0, Math.max(0, track.n - 1)]);
  const [etype, setEtype] = useState("");
  useEffect(() => { setSpan([0, Math.max(0, track.n - 1)]); setEtype(""); }, [track.track_id, track.n]);
  const applicable = (it?.event_types ?? []).filter((t) => t.applicable);
  const def = applicable.find((t) => t.name === etype)?.definition;

  const add = async (type: string) => {
    try {
      const r = await addTrackIntent(track.track_id, { event_type: type, start_frame_id: track.frames[span[0]], end_frame_id: track.frames[span[1]] });
      setMsg(`${humanize(type)} added (${r.state}${r.mirrored_to_intents ? ", also on the track's intents" : ""})`);
      await intentsQ.reload();
      onEventsChanged();
    } catch (e) {
      setMsg(`Intent failed: ${humanizeError(e)}`);
    }
  };

  return (
    <>
      <Panel title={`${humanize(track.class_name)} ${shortId(track.track_id)}`} desc={`${track.n} positions, ${fromNs(track.first_ts_ns)}`}
        right={<Button variant="ghost" onClick={onBack}>All tracks</Button>}>
        <div className="space-y-2 text-[12px]">
          {frameImg && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={frameImg} alt="the anchor frame" className="w-full rounded border border-line" />
          )}
          <label className="block">
            <span className="label-caps">Anchor: position {anchor + 1} of {track.n}</span>
            <input type="range" min={1} max={track.n - 1} value={anchor} className="w-full" aria-label="Anchor position"
              onChange={(e) => setAnchor(Number(e.target.value))} />
          </label>
          <div className="text-ink-3">{baselineNote}</div>
          {!future ? (
            <Button variant="primary" disabled={!!busy} onClick={onBaseline}>Future from here</Button>
          ) : (
            <div className="flex gap-1.5 flex-wrap">
              <Button variant="primary" disabled={!!busy} onClick={onSaveFuture}>{futureEdited ? "Save future" : "Accept baseline"}</Button>
              <Button onClick={onDiscardFuture}>Discard</Button>
            </div>
          )}
          {fc && (
            <div className="grid grid-cols-2 gap-x-3">
              <KV k="Speed" v={`${(fc.state.speed_mps * 3.6).toFixed(1)} km/h`} />
              <KV k="Turn rate" v={`${((fc.state.yaw_rate_rps * 180) / Math.PI).toFixed(1)} deg/s`} />
            </div>
          )}
          {fc && (
            <div className="text-ink-2">
              Baseline against what the track then did:{" "}
              {fc.error_vs_observed ? `${fc.error_vs_observed.ade_m} m on average, ${fc.error_vs_observed.fde_m} m at ${fc.error_vs_observed.horizon_s} s`
                : "nothing was observed after this anchor"}
            </div>
          )}
          {track.futures.length > 0 && (
            <div className="border-t hairline pt-2 space-y-1">
              <div className="label-caps">Saved futures</div>
              {track.futures.map((f) => (
                <div key={f.path_id} className="flex items-center gap-2">
                  <span className="flex-1">{fromNs(f.anchor_ts_ns)} · {humanize(f.source)} · {f.points.length} points</span>
                  <Button variant="ghost" onClick={() => onDeleteFuture(f.path_id)}>Delete</Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Intent" desc="What this road user meant to do, over which positions. An annotator's tag is proposed for review; a reviewer's is accepted.">
        <LoadState loading={intentsQ.loading && !it} error={intentsQ.error}>
          {it && (
            <div className="space-y-2 text-[12px]">
              <div>
                <span className="label-caps">Span: positions {span[0] + 1} to {span[1] + 1}</span>
                <input type="range" min={0} max={track.n - 1} value={span[0]} className="w-full" aria-label="Span start"
                  onChange={(e) => setSpan([Math.min(Number(e.target.value), span[1]), span[1]])} />
                <input type="range" min={0} max={track.n - 1} value={span[1]} className="w-full" aria-label="Span end"
                  onChange={(e) => setSpan([span[0], Math.max(Number(e.target.value), span[0])])} />
              </div>
              <div className="flex flex-wrap gap-1">
                {it.quick.map((q) => <button key={q} type="button" className="chip" onClick={() => void add(q)}>{humanize(q)}</button>)}
              </div>
              {it.missing_quick.length > 0 && (
                <div className="text-ink-3">Not in this pack&apos;s vocabulary yet: {it.missing_quick.map(humanize).join(", ")}.</div>
              )}
              <div className="flex gap-1.5">
                <select className="field flex-1" value={etype} onChange={(e) => setEtype(e.target.value)} aria-label="Intent type">
                  <option value="">Any other intent...</option>
                  {applicable.map((t) => <option key={t.name} value={t.name}>{humanize(t.name)}</option>)}
                </select>
                <Button disabled={!etype} onClick={() => void add(etype)}>Add</Button>
              </div>
              {def && <div className="text-ink-3">{def}</div>}
              <div className="border-t hairline pt-2 space-y-1">
                {it.events.length === 0 && <div className="text-ink-3">No intents on this track yet.</div>}
                {it.events.map((e) => (
                  <div key={e.event_id} className="flex items-center gap-2">
                    <span className="flex-1">{humanize(e.event_type)} <span className="text-ink-3">{((e.end_ts_ns - e.start_ts_ns) / 1e9).toFixed(1)} s · {e.state} · {e.source}</span></span>
                    <Button variant="ghost" onClick={async () => {
                      try { await deleteTrackIntent(e.event_id); await intentsQ.reload(); onEventsChanged(); }
                      catch (err) { setMsg(`Delete failed: ${humanizeError(err)}`); }
                    }}>Delete</Button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </LoadState>
      </Panel>
    </>
  );
}
