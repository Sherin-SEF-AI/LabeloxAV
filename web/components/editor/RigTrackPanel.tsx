"use client";

// M-MC.4 rig track panel: the session's rig tracks (one physical object followed across time and cameras) and,
// on click, a track's timeline of instants showing the cross-camera handoff. A consistency check flags rig
// tracks whose views disagree on the class as cross_cam_inconsistent error candidates (they land in the normal
// review queue); those tracks are marked here so the reviewer can jump straight to them.

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { RigTrackRow, RigTrackTimeline } from "@/lib/types";

export default function RigTrackPanel({ sessionId, onClose, onOpenInstant }: {
  sessionId: string;
  onClose: () => void;
  onOpenInstant?: (objectId: string, cam: string) => void;
}) {
  const [tracks, setTracks] = useState<RigTrackRow[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<RigTrackTimeline | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setBusy(true); setErr(null);
    try {
      await api.multicamRigTracksBuild(sessionId).catch(() => undefined);  // chain identities into tracks first
      setTracks((await api.multicamRigTracks(sessionId)).tracks);
    } catch (e) { setErr(String(e)); }
    finally { setBusy(false); }
  }, [sessionId]);

  useEffect(() => { refresh(); }, [refresh]);

  const openTrack = async (tid: string) => {
    if (open === tid) { setOpen(null); setTimeline(null); return; }
    setOpen(tid);
    try { setTimeline(await api.multicamRigTrackTimeline(sessionId, tid)); }
    catch (e) { setErr("timeline failed: " + String(e)); }
  };
  const consistency = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.multicamConsistencyCheck(sessionId);
      setErr(`consistency: ${r.inconsistent_objects} inconsistent object(s) across ${r.n_tracks} track(s) queued for review`);
      await refresh();
    } catch (e) { setErr("consistency check failed: " + String(e)); }
    finally { setBusy(false); }
  };

  const secs = (a: number | null, b: number | null) => (a != null && b != null ? `${((b - a) / 1e9).toFixed(1)}s` : "");

  return (
    <div className="absolute bottom-3 left-3 z-30 w-[26rem] max-h-[55%] flex flex-col panel bg-bg/95 border border-line rounded shadow-lg font-mono text-[11px]">
      <div className="flex items-center justify-between px-2 py-1.5 border-b hairline">
        <span className="uppercase text-ink-2 tracking-wide">rig tracks {tracks ? `(${tracks.length})` : ""}</span>
        <div className="flex items-center gap-1">
          <button onClick={consistency} disabled={busy} title="flag cross-view class disagreement as review candidates"
            className="border border-warn/50 text-warn px-1.5 py-0.5 rounded hover:bg-warn/10 disabled:opacity-40">consistency check</button>
          <button onClick={onClose} title="close" className="text-ink-3 hover:text-ink px-1">×</button>
        </div>
      </div>
      {err && <div className="px-2 py-1 text-[10px] border-b hairline text-ink-3">{err}</div>}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {tracks?.map((t) => (
          <div key={t.rig_track_id} className={`border rounded ${t.inconsistent ? "border-block/60" : "border-line"}`}>
            <button onClick={() => openTrack(t.rig_track_id)} className="w-full flex items-center justify-between px-1.5 py-1 hover:bg-line/40">
              <span className="text-ink-2 truncate">{t.class_name ?? "?"} <span className="text-ink-3">· {t.instants} inst · {t.cameras.join(",")} · {secs(t.ts_start, t.ts_end)}</span></span>
              {t.inconsistent && <span className="text-block text-[9px] uppercase shrink-0">inconsistent</span>}
            </button>
            {open === t.rig_track_id && timeline && (
              <div className="px-1.5 pb-1.5 flex gap-1 overflow-x-auto">
                {timeline.instants.map((i) => (
                  <div key={i.rig_object_id} className={`shrink-0 border rounded px-1 py-0.5 ${i.conflict ? "border-block/50" : "border-line"}`}>
                    <div className="text-ink-3 text-[9px]">{i.ts_ns != null ? new Date(i.ts_ns / 1e6).toISOString().slice(11, 19) : "?"}</div>
                    {i.members.map((m) => (
                      <button key={m.object_id} onClick={() => m.cam && onOpenInstant?.(m.object_id, m.cam)} title={`${m.cam}: ${m.class_name}`}
                        className="block text-[9px] text-ink-2 hover:text-accent">{m.cam}:{m.class_name}</button>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {tracks && tracks.length === 0 && <div className="text-ink-3 text-[10px]">no rig tracks yet (link objects across views first)</div>}
      </div>
    </div>
  );
}
