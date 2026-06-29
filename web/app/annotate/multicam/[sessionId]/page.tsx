"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { MulticamGroups } from "@/lib/types";
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

  const load = useCallback(async () => {
    setGroups(await api.multicamGroups(sessionId));
  }, [sessionId]);

  useEffect(() => { load(); }, [load]);

  const associate = async () => {
    const r = await api.multicamAssociate(sessionId);
    setMsg(r.reason ? r.reason : `linked ${r.associated} objects into ${r.rig_tracks} rig tracks across ${r.cameras.join(", ")}`);
  };

  const group = groups?.groups[sel];
  const cams = groups?.cameras ?? [];

  return (
    <div className="min-h-screen flex flex-col">
      <PageHeaderBar
        title="MULTICAM"
        subtitle={sessionId.slice(0, 8)}
        meta={
          <>
            <BackButton />
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

        <main className="flex-1 overflow-auto p-4 min-w-0">
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
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img src={`/api/frames/${f.frame_id}/image`} alt={cam} className="w-full aspect-video object-cover bg-bg-2 cursor-pointer" onClick={() => router.push(`/frame/${f.frame_id}`)} />
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
