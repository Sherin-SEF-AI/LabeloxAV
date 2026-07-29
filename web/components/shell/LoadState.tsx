"use client";

import { humanizeError } from "@/lib/api";

// The three things a fetching page owes the person waiting on it: that it is working, that it failed, or
// that it succeeded and there is genuinely nothing here.
//
// A quarter of the pages in this app answered none of them. They rendered an empty container on first paint
// and, if the request failed, kept rendering it forever, which is indistinguishable from a corpus with no
// data in it. The reviewer's reasonable conclusion, that the queue is empty and the shift is over, is the
// worst possible outcome of a dropped request.
//
// Deliberately not a spinner over the whole page: it renders beside the content it describes, so a partial
// load shows what it has rather than hiding everything behind the slowest request.

export default function LoadState({ loading, error, empty, emptyText, onRetry, children }: {
  loading?: boolean;
  error?: unknown;
  empty?: boolean;
  emptyText?: string;
  onRetry?: () => void;
  children?: React.ReactNode;
}) {
  if (error) {
    return (
      <div className="panel px-3 py-2 font-mono text-[11px] flex items-start gap-3">
        <span className="text-block shrink-0">could not load</span>
        {/* The message, not a shrug. humanizeError already turns a status code into a sentence. */}
        <span className="text-ink-2 min-w-0 flex-1">{humanizeError(error)}</span>
        {onRetry && (
          <button onClick={onRetry}
            className="shrink-0 border border-line px-2 py-0.5 text-ink-3 hover:border-accent">
            retry
          </button>
        )}
      </div>
    );
  }
  if (loading) {
    return (
      <div className="panel px-3 py-2 font-mono text-[11px] text-ink-3 flex items-center gap-2">
        <span className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse" />
        loading
      </div>
    );
  }
  if (empty) {
    return (
      <div className="panel px-3 py-6 font-mono text-[11px] text-ink-3 text-center">
        {emptyText ?? "nothing here yet"}
      </div>
    );
  }
  return <>{children}</>;
}
