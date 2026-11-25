"use client";

import { humanizeError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import EmptyState from "@/components/ui/EmptyState";
import Icon from "./Icon";

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
//
// Loading draws placeholder rows shaped like a list rather than a lone "loading" word, so the layout does not
// jump when the data lands; empty uses the shared EmptyState, with an optional title, icon and action.

export default function LoadState({ loading, error, empty, emptyText, emptyTitle, emptyIcon, emptyAction,
  rows = 3, onRetry, children }: {
  loading?: boolean;
  error?: unknown;
  empty?: boolean;
  emptyText?: string;
  emptyTitle?: string;
  emptyIcon?: string;
  emptyAction?: React.ReactNode;
  rows?: number;                 // placeholder rows while loading
  onRetry?: () => void;
  children?: React.ReactNode;
}) {
  const t = useT();
  if (error) {
    return (
      <div role="alert" className="panel border-block/40 bg-block/5 px-3.5 py-3 text-[12.5px] flex items-start gap-3">
        <span className="text-block shrink-0 mt-px"><Icon name="info" size={15} /></span>
        <div className="min-w-0 flex-1">
          <div className="text-ink font-medium">{t("load.failed", "Could not load")}</div>
          {/* The message, not a shrug. humanizeError already turns a status code into a sentence. */}
          <div className="text-ink-2 mt-0.5 break-words">{humanizeError(error)}</div>
        </div>
        {onRetry && (
          <button type="button" onClick={onRetry} className="btn shrink-0">
            {t("action.retry", "Retry")}
          </button>
        )}
      </div>
    );
  }
  if (loading) {
    return (
      <div className="panel px-3.5 py-3 space-y-2.5" role="status" aria-label={t("load.loading", "loading")}>
        {Array.from({ length: Math.max(1, rows) }, (_, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="skeleton h-3" style={{ width: `${[38, 62, 50, 70, 44][i % 5]}%` }} />
            <span className="skeleton h-3 ml-auto w-16" />
          </div>
        ))}
        <span className="sr-only">{t("load.loading", "loading")}</span>
      </div>
    );
  }
  if (empty) {
    return (
      <div className="panel">
        <EmptyState compact icon={emptyIcon ?? "folder"} title={emptyTitle ?? emptyText ?? t("load.empty", "Nothing here yet")}
          action={emptyAction}>
          {emptyTitle ? emptyText : undefined}
        </EmptyState>
      </div>
    );
  }
  return <>{children}</>;
}
