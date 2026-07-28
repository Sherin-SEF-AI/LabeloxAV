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

export type JobStream = {
  training: JobStreamRow[];
  import: JobStreamRow[];
  export: JobStreamRow[];
  autolabel: JobStreamRow[];
};

export function useJobStream(): StreamState<JobStream> {
  return useEventStream<JobStream>("/api/events/jobs", "jobs");
}

export function useTrainingStream(jobId: string | null): StreamState<JobStreamRow> {
  return useEventStream<JobStreamRow>(jobId ? `/api/events/training/${jobId}` : null, "training");
}
