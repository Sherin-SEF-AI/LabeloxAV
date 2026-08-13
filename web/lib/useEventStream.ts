"use client";

import { useEffect, useState } from "react";

import { getUser } from "./user";
import { getJobs, jobsConnected, subscribeJobs } from "./jobStream";

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
  // Queued totals over every row, not over the tails above. The lists are capped at a recent window, so a
  // client counting them under-reports: 67 autolabel jobs parked for the cloud A100 are all older than the
  // ten most recent, and the top bar read "1 queued" against 68. Optional so an older server still parses.
  waiting?: { training?: number; import?: number; export?: number; autolabel?: number };
};

export function useJobStream(): StreamState<JobStream> {
  // Backed by the shared module-level stream rather than its own EventSource. Six pages call this and the
  // top bar now does too, so a connection per caller meant every one of those pages held two to the same
  // endpoint carrying identical frames. The signature is unchanged, so no caller had to move.
  // Connectivity is held in state alongside the data rather than read at render time, because a stream that
  // drops sends no frame and so would never trigger the render that noticed it had gone.
  const [state, setState] = useState<StreamState<JobStream>>(
    () => ({ data: getJobs(), connected: jobsConnected(), error: null }));
  useEffect(() => subscribeJobs(
    (s) => setState({ data: s, connected: jobsConnected(), error: null })), []);
  return state;
}

export type GpuInfo = {
  index: number; name: string;
  memory_used_mb: number | null; memory_total_mb: number | null; memory_used_frac: number | null;
  utilization_pct: number | null; temperature_c: number | null; power_w: number | null;
  processes: { pid: number | null; name: string; used_mb: number | null }[];
};

export type SystemStream = {
  ts: number;
  gpus: GpuInfo[];
  host: {
    cpu_pct: number; cpu_count: number; load: { "1m": number; "5m": number; "15m": number };
    memory_used_mb: number; memory_total_mb: number; memory_used_frac: number;
    disk: { path: string; used_mb: number; total_mb: number; used_frac: number | null };
  };
  process: { pid: number; rss_mb: number; cpu_pct: number; threads: number; uptime_s: number };
};

/**
 * Machine state for the console.
 *
 * Its own connection rather than a field on the job stream: these are gauges read from the host, the job
 * stream is events read from the database, and folding them together would make every job push carry an
 * nvidia-smi call.
 */
export function useSystemStream(): StreamState<SystemStream> {
  return useEventStream<SystemStream>("/api/events/system", "system");
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
/**
 * Follow one autolabel job to completion, resolving with what it produced.
 *
 * The same shape as watchImportJob and for the same reasons: the shared jobs stream is the fast path and a
 * slow poll is the safety net for when it never connects. It is a separate function rather than a parameter
 * on that one because the two settle differently. An import resolves with the session it created, which
 * needs a second fetch; an autolabel already carries its counts on the status row.
 */
export function watchAutolabelJob(
  jobId: string,
  onProgress: (job: { status: string; progress?: number | null }) => void,
  opts: { fallbackMs?: number; timeoutMs?: number } = {},
): Promise<{ objects?: number }> {
  const fallbackMs = opts.fallbackMs ?? 8000;
  const timeoutMs = opts.timeoutMs ?? 60 * 60 * 1000;

  return new Promise((resolve, reject) => {
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

    const settle = async (status: string, error?: string | null, counts?: Record<string, number> | null) => {
      if (done) return;
      if (status === "error") { cleanup(); reject(new Error(error || "autolabel failed")); return; }
      if (status !== "done" && status !== "complete") return;
      cleanup();
      if (counts) { resolve({ objects: counts.objects ?? counts.written ?? 0 }); return; }
      try {
        const { apiGet } = await import("./api");
        const j = await apiGet<{ counts?: Record<string, number> }>(`/api/autolabel/${jobId}`);
        resolve({ objects: j.counts?.objects ?? 0 });
      } catch (e) { reject(e); }
    };

    if (token) {
      es = new EventSource(`/api/events/jobs?token=${encodeURIComponent(token)}`);
      es.addEventListener("jobs", (ev) => {
        try {
          const snap = JSON.parse((ev as MessageEvent).data) as JobStream;
          const row = (snap.autolabel ?? []).find((r) => r.job_id === jobId);
          if (!row) return;
          onProgress(row);
          void settle(row.status, (row as { error?: string }).error);
        } catch { /* a malformed frame must not fail the run the user is watching */ }
      });
    }

    poll = setInterval(async () => {
      try {
        const { apiGet } = await import("./api");
        const j = await apiGet<{ status: string; progress?: number; error?: string | null;
                                 counts?: Record<string, number> }>(`/api/autolabel/${jobId}`);
        onProgress(j);
        void settle(j.status, j.error, j.counts);
      } catch (e) { cleanup(); reject(e); }
    }, fallbackMs);

    deadline = setTimeout(() => {
      cleanup();
      reject(new Error("the autolabel run did not finish in time; check the jobs page for its state"));
    }, timeoutMs);
  });
}

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
