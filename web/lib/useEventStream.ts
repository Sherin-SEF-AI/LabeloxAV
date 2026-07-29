"use client";

import { useEffect, useState } from "react";

import { getUser } from "./user";

// Subscribe to a server-sent event stream.
//
// This replaces the setInterval polling scattered across the app: nine loops, each re-fetching a full
// snapshot every two or three seconds whether or not anything had changed, none of them pausing when the tab
// was hidden and none backing off on error. The server now pushes only when state actually changes.
//
// EventSource cannot set an Authorization header (the browser API provides no way to), and reads are
// deny-by-default, so the token rides as a query parameter for these endpoints only. That is also why the
// stream carries job progress and nothing sensitive: a URL can reach a proxy log in a way a header does not.
export type StreamState<T> = {
  data: T | null;
  connected: boolean;
  error: string | null;
};

export function useEventStream<T>(path: string | null, eventName: string): StreamState<T> {
  const [data, setData] = useState<T | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!path) return;
    const token = getUser()?.token;
    if (!token) {
      // Without a token the stream 401s immediately and EventSource would retry forever, so say why rather
      // than reconnecting in a loop.
      setError("not signed in");
      return;
    }

    const url = `${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
    const es = new EventSource(url);

    es.addEventListener("open", () => {
      setConnected(true);
      setError(null);
    });
    es.addEventListener(eventName, (ev) => {
      try {
        setData(JSON.parse((ev as MessageEvent).data) as T);
        setError(null);
      } catch {
        // A malformed frame must not tear down a working stream; the next one is usually fine.
      }
    });
    es.addEventListener("error", () => {
      // EventSource reconnects on its own with backoff, so this only reports the gap. Resubscribing here
      // would stack duplicate connections.
      setConnected(false);
    });

    return () => {
      es.close();
      setConnected(false);
    };
  }, [path, eventName]);

  return { data, connected, error };
}

// The shape the /api/events/jobs stream pushes.
export type JobStreamRow = {
  job_id: string;
  status: string;
  stage?: string | null;
  progress?: number | null;
  purpose?: string;
  task_type?: string;
  counts?: Record<string, number>;
  metrics?: Record<string, number> | null;
};

export type IngestProgress = {
  source: string;                 // import_job | batch_log | none
  active: boolean;
  finished: boolean;
  done: number;
  total: number;
  current: string | null;
  frames: number;
  progress: number;
  note?: string;
};

export type JobStream = {
  training: JobStreamRow[];
  import: JobStreamRow[];
  export: JobStreamRow[];
  autolabel: JobStreamRow[];
  // Ingest rides the same stream rather than getting one of its own: one connection serves every page
  // that watches work happen, instead of each opening its own.
  ingest: IngestProgress[];
};

export function useJobStream(): StreamState<JobStream> {
  return useEventStream<JobStream>("/api/events/jobs", "jobs");
}

export function useTrainingStream(jobId: string | null): StreamState<JobStreamRow> {
  return useEventStream<JobStreamRow>(jobId ? `/api/events/training/${jobId}` : null, "training");
}

// Follow one job to completion over the shared stream, for a foreground wait like the import wizard.
//
// Imperative rather than a hook because the caller is inside an async workflow, not a render: the wizard
// awaits this between "upload" and "open the session". The stream carries status and progress; the
// terminal record is fetched once at the end, because the session id the caller needs is not on the wire
// and putting a full job record in every frame would be wasteful for a value read once.
//
// A poll fallback is kept and deliberately slow. If the stream cannot connect (an old proxy that buffers
// event streams, for instance) the wizard must still finish rather than hanging on a progress bar forever.
export function watchImportJob(
  jobId: string,
  onProgress: (job: { status: string; progress?: number | null; counts?: Record<string, number> }) => void,
  opts: { fallbackMs?: number; timeoutMs?: number } = {},
): Promise<string> {
  const fallbackMs = opts.fallbackMs ?? 8000;
  const timeoutMs = opts.timeoutMs ?? 30 * 60 * 1000;

  return new Promise<string>((resolve, reject) => {
    const token = getUser()?.token;
    let done = false;
    let es: EventSource | null = null;
    let poll: ReturnType<typeof setInterval> | null = null;
    let deadline: ReturnType<typeof setTimeout> | null = null;

    const cleanup = () => {
      done = true;
      es?.close();
      if (poll) clearInterval(poll);
      if (deadline) clearTimeout(deadline);
    };

    const settle = async (status: string, error?: string | null) => {
      if (done) return;
      if (status === "error") { cleanup(); reject(new Error(error || "import failed")); return; }
      if (status !== "done") return;
      cleanup();
      try {
        const { apiGet } = await import("./api");
        const job = await apiGet<{ session_id?: string | null }>(`/api/imports/${jobId}`);
        if (!job.session_id) { reject(new Error("import done but no session was created")); return; }
        resolve(job.session_id);
      } catch (e) { reject(e); }
    };

    if (token) {
      es = new EventSource(`/api/events/jobs?token=${encodeURIComponent(token)}`);
      es.addEventListener("jobs", (ev) => {
        try {
          const snap = JSON.parse((ev as MessageEvent).data) as JobStream;
          const row = (snap.import ?? []).find((r) => r.job_id === jobId);
          if (!row) return;
          onProgress(row);
          void settle(row.status, (row as { error?: string }).error);
        } catch { /* a malformed frame must not fail the import the user is watching */ }
      });
    }

    // Slow safety net, not the primary path: it only matters when the stream never arrives.
    poll = setInterval(async () => {
      try {
        const { apiGet } = await import("./api");
        const job = await apiGet<{ status: string; progress?: number; counts?: Record<string, number>;
                                   error?: string | null }>(`/api/imports/${jobId}`);
        onProgress(job);
        void settle(job.status, job.error);
      } catch (e) { cleanup(); reject(e); }
    }, fallbackMs);

    deadline = setTimeout(() => {
      cleanup();
      reject(new Error("the import did not finish in time; check the jobs page for its state"));
    }, timeoutMs);
  });
}
