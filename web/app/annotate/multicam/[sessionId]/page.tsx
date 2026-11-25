"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import type { FrameObject, MulticamGroups } from "@/lib/types";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import Inspector from "@/components/shell/Inspector";
import LoadState from "@/components/shell/LoadState";
import { fromNs, shortId } from "@/lib/format";

// Server sentences arrive lowercase ("this session has 1 camera..."); capitalise the first letter only, so
// camera ids and class names inside them keep their case.
const sentence = (t: string) => (t ? t[0].toUpperCase() + t.slice(1) : t);
const views = (n: number) => `${n} view${n === 1 ? "" : "s"}`;

// M3.1 multi-camera synchronized annotation workspace: all rig views at a chosen timestamp, annotate
// once and link across views. Single-camera sessions show one view per instant (degrades gracefully).

export default function MulticamWorkspace() {
  const router = useRouter();
  const sessionId = String(useParams().sessionId);
  const [groups, setGroups] = useState<MulticamGroups | null>(null);
  const [sel, setSel] = useState(0);
  const [msg, setMsg] = useState<string | null>(null);
  // Whether this session can support cross-view linking at all. The machinery is complete and the corpus
  // mostly is not, and a grid that does nothing with no explanation is the worst state for a working tool.
  const [ready, setReady] = useState<Awaited<ReturnType<typeof api.multicamReadiness>> | null>(null);
  // The objects on the camera being worked from, and the candidate this one would produce elsewhere.
  const [srcCam, setSrcCam] = useState<string | null>(null);
  const [objects, setObjects] = useState<FrameObject[]>([]);
  const [pick, setPick] = useState<string | null>(null);
  const [plan, setPlan] = useState<Awaited<ReturnType<typeof api.agentCrossCamPlan>> | null>(null);
  const [busy, setBusy] = useState(false);
  const [loadErr, setLoadErr] = useState<unknown>(null);

  const load = useCallback(async () => {
    try {
      setGroups(await api.multicamGroups(sessionId));
      setLoadErr(null);
    } catch (e) {
      setLoadErr(e);
    }
  }, [sessionId]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    api.multicamReadiness(sessionId).then(setReady)
      .catch((e) => setMsg(`Could not check whether this session is linkable: ${humanizeError(e)}`));
  }, [sessionId]);

  const associate = async () => {
    setBusy(true);
    try {
      const r = await api.multicamAssociate(sessionId);
      setMsg(r.reason ? sentence(r.reason) : `Linked ${r.associated} objects into ${r.rig_tracks} rig tracks across ${r.cameras.join(", ")}`);
    } catch (e) {
      setMsg(`Associate failed: ${humanizeError(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const group = groups?.groups[sel];
  const cams = groups?.cameras ?? [];

  // The objects on whichever camera is being worked from, so one can be picked to carry across.
  useEffect(() => {
    const f = srcCam && group ? group.frames[srcCam] : null;
    if (!f) { setObjects([]); return; }
    let live = true;
    api.frameObjects(f.frame_id)
      .then((o) => { if (live) setObjects(o); })
      .catch((e) => { if (live) setMsg(humanizeError(e)); });
    return () => { live = false; };
  }, [srcCam, group]);

  useEffect(() => { setPick(null); setPlan(null); }, [sel, srcCam]);

  /** What this object would look like in the other cameras. Read-only. */
  const planCross = useCallback(async (objectId: string) => {
    setPick(objectId);
    setPlan(null);
    setBusy(true);
    try {
      const p = await api.agentCrossCamPlan(objectId);
      setPlan(p);
      setMsg(p.reason
        ? sentence(p.reason)
        : `${p.counts.auto_accept + p.counts.review} of ${views(p.counts.targets)} can see it`);
    } catch (e) {
      setMsg(humanizeError(e));
    } finally {
      setBusy(false);
    }
  }, []);

  /** Commit the plan as one reversible run. */
  const commitCross = useCallback(async () => {
    if (!pick) return;
    setBusy(true);
    try {
      const r = await api.agentCrossCam(pick);
      setMsg(r.created
        ? `Placed in ${views(r.created)}. Undo it from agent run ${shortId(r.run_id)}.`
        : "Nothing to place: no other view can see this object");
      setPlan(null);
      setPick(null);
    } catch (e) {
      setMsg(humanizeError(e));
    } finally {
      setBusy(false);
    }
  }, [pick]);

  return (
    <div className="min-h-screen flex flex-col">
      <PageHeaderBar
        title="Multi-camera"
        subtitle={shortId(sessionId)}
        meta={
          <>
            {/* A deep link or a fresh tab has no history, and the default fallback is "/" - which
                drops someone who opened a rig view straight to the triage queue. The session this
                view belongs to is the place they meant to be. */}
            <BackButton fallback={`/annotations?session=${sessionId}`} label="Session" />
            {groups && <span className="text-ink-3">{cams.length} camera{cams.length !== 1 ? "s" : ""} · {groups.n_groups} synchronised instants{groups.multicamera ? "" : " · single camera"}</span>}
          </>
        }
        right={msg && <span role="status" title={msg} className={`max-w-[560px] truncate ${/fail|could not/i.test(msg) ? "text-warn" : "text-ink-2"}`}>{msg}</span>}
        primaryAction={
          <button type="button" onClick={associate} disabled={busy} className="btn btn-primary"
            title="match objects that are the same physical thing across the rig's cameras into rig tracks">
            Associate across views
          </button>
        }
      />

      <div className="flex-1 flex min-h-0">
        <Inspector title="Sync groups" side="left">
          <div className="p-3 space-y-3">
            <div>
              <div className="section-title">Synchronised instants</div>
              <p className="section-desc mt-0.5">Moments where the rig&apos;s cameras fired together. Pick one to see every view at that instant.</p>
            </div>
            <div className="flex flex-wrap items-center gap-1">
              {groups?.groups.map((g, i) => (
                <button key={i} type="button" onClick={() => setSel(i)} aria-pressed={i === sel}
                  aria-label={`instant ${i + 1}: ${fromNs(g.ts_ns)}, ${views(Object.keys(g.frames).length)}`}
                  title={`instant ${i + 1} · ${fromNs(g.ts_ns)} · ${views(Object.keys(g.frames).length)}`}
                  className="w-3 shrink-0 h-9 rounded-sm"
                  style={{ background: i === sel ? "#FF7A2F" : Object.keys(g.frames).length > 1 ? "#56D364" : "#3a3f46" }} />
              ))}
            </div>
            {group && (
              <div className="text-[12.5px] text-ink-2">
                Instant {sel + 1} of {groups?.n_groups ?? groups?.groups.length}
                <div className="text-[12px] text-ink-3 tabular-nums">{fromNs(group.ts_ns)} · {views(Object.keys(group.frames).length)}</div>
              </div>
            )}
            <div className="flex flex-col gap-1 text-[12px] text-ink-3 border-t hairline pt-2">
              <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: "#FF7A2F" }} />Selected</span>
              <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: "#56D364" }} />Seen by more than one camera</span>
              <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: "#3a3f46" }} />One camera only</span>
            </div>
          </div>
        </Inspector>

        <main className="flex-1 overflow-auto p-4 min-w-0 space-y-3">
          {(loadErr || !groups) && (
            <LoadState loading={!groups && !loadErr} error={loadErr} rows={3} onRetry={() => { setLoadErr(null); void load(); }} />
          )}
          {/* Why this session cannot be linked, in the order the fixes have to happen. Labelling objects
              on a session whose calibration has never been validated produces labels that cannot be
              projected anywhere, so the ordering is part of the answer. */}
          {ready && !ready.ready && (
            <div className="panel p-3.5 text-[12.5px] space-y-2.5 border-warn/40">
              <div className="section-title text-warn">
                Cross-view linking is not available on this session
              </div>
              {ready.blockers.map((b, i) => (
                <div key={b.code} className="flex gap-2">
                  <span className="text-ink-3 shrink-0 tabular-nums">{i + 1}.</span>
                  <span className="min-w-0">
                    <span className="block text-ink">{sentence(b.detail)}</span>
                    <span className="block text-ink-3">{sentence(b.fix)}</span>
                  </span>
                </div>
              ))}
              <div className="text-ink-3 border-t hairline pt-2">
                The projection itself is built and tested. What is missing is on this session, not in the
                tool.
              </div>
            </div>
          )}

          {/* Carry one object across the rig. Only offered where it can actually work. */}
          {ready?.ready && group && (
            <div className="panel p-3 text-[12.5px] flex flex-wrap items-center gap-2">
              <span className="text-ink-2">Carry an object across from</span>
              {cams.filter((c) => group.frames[c]).map((c) => (
                <button key={c} type="button" onClick={() => setSrcCam(c === srcCam ? null : c)} aria-pressed={c === srcCam}
                  className={`chip font-mono ${c === srcCam ? "chip-on" : ""}`}>
                  {c}
                </button>
              ))}
              {srcCam && (
                <span className="text-ink-3">
                  {objects.length ? `${objects.length} object${objects.length === 1 ? "" : "s"}: pick one below` : "No objects on this view"}
                </span>
              )}
              {plan && pick && (
                <button type="button" onClick={commitCross} disabled={busy || !plan.counts.targets}
                  className="ml-auto btn btn-primary">
                  Place in {views(plan.counts.auto_accept + plan.counts.review)}
                </button>
              )}
            </div>
          )}

          {/* The objects available to carry, from the chosen source view. */}
          {ready?.ready && srcCam && objects.length > 0 && (
            <div className="panel p-2.5 flex flex-wrap gap-1.5">
              {objects.map((o) => (
                <button key={o.object_id} type="button" onClick={() => void planCross(o.object_id)} disabled={busy}
                  aria-pressed={o.object_id === pick}
                  className={`chip ${o.object_id === pick ? "chip-on" : ""} disabled:opacity-40`}>
                  {o.class_name.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          )}

          {/* rig views at the selected instant */}
          {group && (
            <div className={`grid gap-3 ${cams.length > 2 ? "grid-cols-2 lg:grid-cols-3" : cams.length === 2 ? "grid-cols-2" : "grid-cols-1 max-w-5xl"}`}>
              {cams.map((cam) => {
                const f = group.frames[cam];
                return (
                  <div key={cam} className="panel">
                    <div className="panel-head justify-between">
                      <span className="text-[12.5px] font-medium text-ink font-mono">{cam}</span>
                      {f && <button type="button" onClick={() => router.push(`/frame/${f.frame_id}`)} className="link text-[12px]">Annotate →</button>}
                    </div>
                    {f ? (
                      <div className="relative">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <button type="button" className="block w-full" aria-label={`open the ${cam} frame in the editor`}
                          onClick={() => router.push(`/frame/${f.frame_id}`)}>
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={`/api/frames/${f.frame_id}/image`} alt={cam} className="w-full aspect-video object-cover bg-bg-2 cursor-pointer" />
                        </button>
                        {/* Where the picked object would land in this view, before anything is written.
                            Drawn as a fraction of the frame so it survives the aspect crop above. */}
                        {(() => {
                          const it = plan?.items.find((x) => x.frame_id === f.frame_id);
                          // Normalised, because the group response carries no frame size and the only
                          // thing a caller can do with a pixel box is draw it.
                          if (!it?.box_norm) return null;
                          const [x1, y1, x2, y2] = it.box_norm;
                          return (
                            <div className="absolute pointer-events-none border-2"
                              title={`${it.action} - ${Math.round(it.visibility * 100)}% visible here`}
                              style={{
                                left: `${100 * x1}%`, top: `${100 * y1}%`,
                                width: `${100 * (x2 - x1)}%`, height: `${100 * (y2 - y1)}%`,
                                borderColor: it.action === "auto_accept" ? "#56D364" : "#E3B341",
                              }} />
                          );
                        })()}
                        {plan?.items.some((x) => x.frame_id === f.frame_id && x.action === "skip") && (
                          <span className="absolute bottom-1 left-1 bg-bg/80 text-ink-2 text-[11px] px-1.5 py-0.5 rounded">
                            Not visible from here
                          </span>
                        )}
                      </div>
                    ) : (
                      <div className="aspect-video bg-bg-2 flex items-center justify-center text-[12px] text-ink-3">No frame from this camera at this instant</div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
