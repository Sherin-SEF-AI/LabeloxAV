"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import type { FrameObject, MulticamGroups } from "@/lib/types";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import Inspector from "@/components/shell/Inspector";

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

  const load = useCallback(async () => {
    setGroups(await api.multicamGroups(sessionId));
  }, [sessionId]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    api.multicamReadiness(sessionId).then(setReady)
      .catch((e) => setMsg(`could not check whether this session is linkable: ${humanizeError(e)}`));
  }, [sessionId]);

  const associate = async () => {
    const r = await api.multicamAssociate(sessionId);
    setMsg(r.reason ? r.reason : `linked ${r.associated} objects into ${r.rig_tracks} rig tracks across ${r.cameras.join(", ")}`);
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
        ? p.reason
        : `${p.counts.auto_accept + p.counts.review} of ${p.counts.targets} view(s) can see it`);
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
        ? `placed in ${r.created} view(s); undo from the agent run ${r.run_id.slice(0, 8)}`
        : "nothing to place: no other view can see this object");
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
        title="MULTICAM"
        subtitle={sessionId.slice(0, 8)}
        meta={
          <>
            {/* A deep link or a fresh tab has no history, and the default fallback is "/" - which
                drops someone who opened a rig view straight to the triage queue. The session this
                view belongs to is the place they meant to be. */}
            <BackButton fallback={`/annotations?session=${sessionId}`} label="session" />
            {groups && <span className="text-ink-3">{cams.length} camera{cams.length !== 1 ? "s" : ""} · {groups.n_groups} sync groups {groups.multicamera ? "" : "(single-camera)"}</span>}
            {msg && <span className="text-warn">{msg}</span>}
          </>
        }
        primaryAction={
          <button onClick={associate} className="border border-accent text-accent px-2 py-1 hover:bg-accent/10">associate across views</button>
        }
      />

      <div className="flex-1 flex min-h-0">
        <Inspector title="sync groups" side="left">
          <div className="p-3 space-y-2">
            <div className="font-mono text-[10px] uppercase text-ink-3">synchronized instants (PPS ts_ns)</div>
            <div className="flex flex-wrap items-center gap-px">
              {groups?.groups.map((g, i) => (
                <button key={i} onClick={() => setSel(i)}
                  title={`ts ${g.ts_ns} · ${Object.keys(g.frames).length} view(s)`}
                  className="w-2 shrink-0 h-8"
                  style={{ background: i === sel ? "#FF7A2F" : Object.keys(g.frames).length > 1 ? "#56D364" : "#3a3f46" }} />
              ))}
            </div>
          </div>
        </Inspector>

        <main className="flex-1 overflow-auto p-4 min-w-0 space-y-3">
          {/* Why this session cannot be linked, in the order the fixes have to happen. Labelling objects
              on a session whose calibration has never been validated produces labels that cannot be
              projected anywhere, so the ordering is part of the answer. */}
          {ready && !ready.ready && (
            <div className="panel p-3 font-mono text-[11px] space-y-2">
              <div className="text-warn uppercase text-[10px]">
                cross-view linking is not available on this session
              </div>
              {ready.blockers.map((b, i) => (
                <div key={b.code} className="flex gap-2">
                  <span className="text-ink-3 shrink-0">{i + 1}.</span>
                  <span className="min-w-0">
                    <span className="block text-ink-2">{b.detail}</span>
                    <span className="block text-ink-3">{b.fix}</span>
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
            <div className="panel p-3 font-mono text-[11px] flex flex-wrap items-center gap-2">
              <span className="text-ink-3">carry an object across from</span>
              {cams.filter((c) => group.frames[c]).map((c) => (
                <button key={c} onClick={() => setSrcCam(c === srcCam ? null : c)}
                  className={`border px-2 py-0.5 rounded ${c === srcCam ? "border-accent text-accent" : "border-line text-ink-2 hover:border-accent"}`}>
                  {c}
                </button>
              ))}
              {srcCam && (
                <span className="text-ink-3">
                  {objects.length ? `${objects.length} object(s) - pick one below` : "no objects on this view"}
                </span>
              )}
              {plan && pick && (
                <button onClick={commitCross} disabled={busy || !plan.counts.targets}
                  className="ml-auto border border-pass/60 text-pass px-2 py-0.5 rounded hover:bg-pass/10 disabled:opacity-40">
                  place in {plan.counts.auto_accept + plan.counts.review} view(s)
                </button>
              )}
            </div>
          )}

          {/* The objects available to carry, from the chosen source view. */}
          {ready?.ready && srcCam && objects.length > 0 && (
            <div className="panel p-2 flex flex-wrap gap-1.5 font-mono text-[10px]">
              {objects.map((o) => (
                <button key={o.object_id} onClick={() => void planCross(o.object_id)} disabled={busy}
                  className={`border px-1.5 py-0.5 rounded ${o.object_id === pick ? "border-accent text-accent" : "border-line text-ink-2 hover:border-accent"} disabled:opacity-40`}>
                  {o.class_name}
                </button>
              ))}
            </div>
          )}

          {/* rig views at the selected instant */}
          {group && (
            <div className={`grid gap-3 ${cams.length > 2 ? "grid-cols-2 lg:grid-cols-3" : cams.length === 2 ? "grid-cols-2" : "grid-cols-1 max-w-2xl"}`}>
              {cams.map((cam) => {
                const f = group.frames[cam];
                return (
                  <div key={cam} className="panel">
                    <div className="font-mono text-[10px] uppercase text-ink-3 px-2 py-1 border-b hairline flex justify-between">
                      <span className="text-ink-2">{cam}</span>
                      {f && <button onClick={() => router.push(`/frame/${f.frame_id}`)} className="text-info hover:text-accent">annotate →</button>}
                    </div>
                    {f ? (
                      <div className="relative">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={`/api/frames/${f.frame_id}/image`} alt={cam} className="w-full aspect-video object-cover bg-bg-2 cursor-pointer" onClick={() => router.push(`/frame/${f.frame_id}`)} />
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
                          <span className="absolute bottom-1 left-1 bg-bg/80 text-ink-3 font-mono text-[9px] px-1">
                            not visible from here
                          </span>
                        )}
                      </div>
                    ) : (
                      <div className="aspect-video bg-bg-2 flex items-center justify-center font-mono text-[10px] text-ink-3">no frame at this instant</div>
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
