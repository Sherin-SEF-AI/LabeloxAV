"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { api, apiGet, apiPost, humanizeError } from "@/lib/api";

// Sample drives a new team can load before they have footage of their own (services/samples.py): real
// Indian street video from Wikimedia Commons under CC BY-SA 3.0. Each card carries its credit and links to
// the source, which the licence asks for and which keeps a sample from being mistaken for the team's data.
//
// Loading fetches the file on the server and imports it like an upload: faces and plates blurred, frames
// sampled. The card follows the import job until there is a session to open.

type Sample = {
  id: string; title: string; description: string; size_bytes: number; license: string; license_url: string;
  author: string; source_page: string; credit: string; session_id: string | null;
};
type ImportJob = { status: string; progress: number; error: string | null; session_id: string | null;
                   counts: Record<string, unknown> };

function SampleCard({ s, onLoaded }: { s: Sample; onLoaded: () => void }) {
  const router = useRouter();
  const [job, setJob] = useState<string | null>(null);
  const [state, setState] = useState<ImportJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(s.session_id);

  useEffect(() => {
    if (!job) return;
    let live = true;
    const tick = async () => {
      try {
        const j = await apiGet<ImportJob>(`/api/imports/${job}`);
        if (!live) return;
        setState(j);
        if (j.status === "done") { setSessionId(j.session_id); setJob(null); onLoaded(); return; }
        if (j.status === "error" || j.status === "canceled") { setError(j.error ?? j.status); setJob(null); return; }
        setTimeout(tick, 2000);
      } catch (e) {
        if (live) { setError(humanizeError(e)); setJob(null); }
      }
    };
    void tick();
    return () => { live = false; };
  }, [job, onLoaded]);

  const load = async () => {
    setError(null);
    try {
      const r = await apiPost<{ status: string; job_id?: string; session_id?: string }>(`/api/samples/${s.id}/load`, {});
      if (r.session_id) setSessionId(r.session_id);
      else if (r.job_id) setJob(r.job_id);
    } catch (e) { setError(humanizeError(e)); }
  };
  const open = async () => {
    if (!sessionId) return;
    try { router.push(`/frame/${(await api.firstFrame(sessionId)).frame_id}`); }
    catch (e) { setError(humanizeError(e)); }
  };

  const stage = state ? String(state.counts?.stage ?? "") : "";
  return (
    <div className="rounded-md border hairline p-3 flex flex-col gap-2 min-w-0">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[13px] text-ink">{s.title}</div>
          <div className="text-[12px] text-ink-3">{s.description}</div>
        </div>
        <span className="text-[11.5px] text-ink-3 shrink-0">{Math.round(s.size_bytes / 1e6)} MB</span>
      </div>
      <div className="text-[11.5px] text-ink-3">
        by <a href={s.source_page} target="_blank" rel="noreferrer" className="text-accent hover:text-accent-2">{s.author}</a>,{" "}
        <a href={s.license_url} target="_blank" rel="noreferrer" className="text-accent hover:text-accent-2">{s.license}</a>, via Wikimedia Commons
      </div>
      <div className="flex items-center gap-2">
        {sessionId ? (
          <button className="btn btn-primary" onClick={() => void open()}>Open</button>
        ) : job ? (
          <span className="text-[12px] text-ink-2">
            {stage === "downloading sample" ? "Downloading from Wikimedia Commons..."
              : `Importing: ${Math.round((state?.progress ?? 0) * 100)}%`}
          </span>
        ) : (
          <button className="btn" onClick={() => void load()}>Load sample</button>
        )}
      </div>
      {error && <div className="text-[12px] text-block">{error}</div>}
    </div>
  );
}

export default function SampleDrives({ onLoaded }: { onLoaded?: () => void }) {
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    apiGet<{ samples: Sample[] }>("/api/samples").then((r) => setSamples(r.samples))
      .catch((e) => setFailed(humanizeError(e)));
  }, []);
  const loaded = useCallback(() => onLoaded?.(), [onLoaded]);
  if (failed) return <p className="text-[12px] text-ink-3">Sample drives are unavailable: {failed}</p>;
  if (!samples) return null;
  return (
    <div className="grid sm:grid-cols-2 gap-2">
      {samples.map((s) => <SampleCard key={s.id} s={s} onLoaded={loaded} />)}
    </div>
  );
}
