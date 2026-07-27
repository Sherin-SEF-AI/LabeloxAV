"use client";

import { useEffect } from "react";

// Route-level error boundary. Without one, a single render throw anywhere in a page unmounts the whole tree
// and the user is left staring at a blank white screen with no way back. This keeps the app shell usable and
// offers the two things that actually help: retry the segment, or leave for a page that works.
export default function RouteError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    // Nothing aggregates browser logs yet, so the console is the only place this is visible. Keep the digest:
    // it is the only handle that correlates a user report with a server-side stack.
    console.error("route error", error);
  }, [error]);

  return (
    <div className="p-8 max-w-2xl mx-auto space-y-4" role="alert">
      <h1 className="font-display text-lg text-ink">Something broke on this page</h1>
      <p className="font-mono text-[12px] text-ink-3">
        The rest of the app is fine. Retrying re-renders just this page; if it keeps failing, the message below
        is what to report.
      </p>
      <pre className="panel p-3 font-mono text-[11px] text-warn overflow-x-auto whitespace-pre-wrap">
        {error.message || "unknown error"}
        {error.digest ? `\n\ndigest: ${error.digest}` : ""}
      </pre>
      <div className="flex items-center gap-2">
        <button
          onClick={reset}
          className="font-mono text-xs border border-accent text-accent px-3 py-1 hover:bg-accent/10"
        >
          retry
        </button>
        <a
          href="/"
          className="font-mono text-xs border border-line text-ink-2 px-3 py-1 hover:border-accent"
        >
          back to triage
        </a>
      </div>
    </div>
  );
}
