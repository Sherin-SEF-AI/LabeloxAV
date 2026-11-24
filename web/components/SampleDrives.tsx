"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { api, apiGet, apiPost, humanizeError } from "@/lib/api";
import { T } from "@/lib/i18n";
import { invalidate, useQuery } from "@/lib/useQuery";

// Sample drives a new team can load before they have footage of their own (services/samples.py): real
// Indian street video from Wikimedia Commons under CC BY-SA 3.0. Each card carries its credit and links to
// the source, which the licence asks for and which keeps a sample from being mistaken for the team's data.
//
// Loading fetches the file on the server and imports it like an upload: faces and plates blurred, frames
// sampled. The card follows the import job until there is a session to open.

type Sample = {
  id: string; title: string; description: string; size_bytes: number; license: string; license_url: string;
  author: string; source_page: string; credit: string; session_id: string | null;
  job_id: string | null;   // an import of this sample still running, so a reloaded page picks it back up
};
type ImportJob = { status: string; progress: number; error: string | null; session_id: string | null;
                   counts: Record<string, unknown> };

const ENDED = new Set(["done", "error", "canceled"]);

function SampleCard({ s, onLoaded }: { s: Sample; onLoaded: () => void }) {
  const router = useRouter();
  const [job, setJob] = useState<string | null>(s.session_id ? null : s.job_id);
  const [state, setState] = useState<ImportJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(s.session_id);

  // Follow the import every 2s until it ends one way or the other. A failed request ends it too, with the
  // reason shown, exactly as the hand-written loop did; the hook stops the poll while the tab is hidden.
  const jobQ = useQuery(job ? [`/api/imports/${job}`] : null, () => apiGet<ImportJob>(`/api/imports/${job}`),
    { pollMs: (d, e) => (e || (d && ENDED.has(d.status)) ? 0 : 2000) });
  const onLoadedRef = useRef(onLoaded);
  onLoadedRef.current = onLoaded;
  useEffect(() => {
    const j = jobQ.data;
    if (!j) return;
    setState(j);
    if (j.status === "done") {
      setSessionId(j.session_id); setJob(null);
      void invalidate("/api/samples");
      onLoadedRef.current();
    } else if (j.status === "error" || j.status === "canceled") {
      setError(j.error ?? j.status); setJob(null);
    }
  }, [jobQ.data]);
  useEffect(() => {
    if (jobQ.error) { setError(humanizeError(jobQ.error)); setJob(null); }
  }, [jobQ.error]);

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
          <button className="btn btn-primary" onClick={() => void open()}><T k="common.open">Open</T></button>
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
  const samplesQ = useQuery(["/api/samples"], () => apiGet<{ samples: Sample[] }>("/api/samples"));
  const samples = samplesQ.data?.samples ?? null;
  const failed = samples || !samplesQ.error ? null : humanizeError(samplesQ.error);
  const loaded = useCallback(() => onLoaded?.(), [onLoaded]);
  if (failed) return <p className="text-[12px] text-ink-3">Sample drives are unavailable: {failed}</p>;
  if (!samples) return null;
  return (
    <div className="grid sm:grid-cols-2 gap-2">
      {/* Keyed on what the server says about the sample, not just its id: a card takes its starting state
          from these fields, so a cached list refreshed with a newer answer has to remount it. */}
      {samples.map((s) => <SampleCard key={`${s.id}:${s.session_id ?? ""}:${s.job_id ?? ""}`} s={s} onLoaded={loaded} />)}
    </div>
  );
}
