"use client";

import { useEffect } from "react";

import PageShell from "./PageShell";
import { useLabel, useT } from "@/lib/i18n";

// What an error boundary shows, shared by the root one (app/error.tsx) and the per-route ones.
//
// The root boundary is the last line: a throw anywhere unmounts everything under the root layout and
// leaves this in its place, with no sidebar, because the sidebar may be what threw. The routes that load a
// rendering engine get a boundary of their own closer in (app/lidar, app/map, app/inspect, app/training,
// app/agent). three and maplibre fail in ways the rest of the app does not: a lost WebGL context, a GPU
// process that died, a shader the driver rejects. Contained to the route, that costs one page and leaves the
// navigation standing, so the way out is a click instead of a reload.

type BoundaryProps = { error: Error & { digest?: string }; reset: () => void };

export function RouteErrorBody({ error, reset, what, home = "/" }: BoundaryProps & {
  /** The route's name, for a boundary that knows which route it guards. Omitted at the root. */
  what?: string;
  home?: string;
}) {
  const t = useT();
  useEffect(() => {
    // Nothing aggregates browser logs yet, so the console is the only place this is visible. Keep the digest:
    // it is the only handle that correlates a user report with a server-side stack.
    console.error("route error", error);
  }, [error]);

  return (
    <div className="page-wrap-narrow" role="alert">
      <div className="panel p-6 mt-6">
        <div className="flex items-start gap-3">
          <span className="flex items-center justify-center w-9 h-9 rounded-full bg-block/10 text-block shrink-0" aria-hidden>!</span>
          <div className="min-w-0">
            <h1 className="text-[16px] font-semibold text-ink">
              {what ? t("error.segment_title", "{what} stopped working", { what }) : t("error.title", "Something broke on this page")}
            </h1>
            <p className="mt-1 text-[13px] text-ink-3 leading-relaxed">
              {what
                ? t("error.segment_body", "The rest of the app is fine and the sidebar still works. Try again; if it keeps failing, send the details below to whoever runs this installation.")
                : t("error.body", "The rest of the app is fine. Try again; if it keeps failing, send the details below to whoever runs this installation.")}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-5">
          <button type="button" onClick={reset} className="btn btn-primary">{t("action.retry", "Retry")}</button>
          <a href={home} className="btn">{t("error.back_home", "Go to Home")}</a>
        </div>
        {/* The message and digest are for a bug report, not for the person reading the page, so they are
            one click away rather than a block of orange monospace in the middle of it. */}
        <details className="mt-5 text-[12px]">
          <summary className="cursor-pointer text-ink-3 hover:text-ink-2">{t("error.details", "Technical details")}</summary>
          <pre className="mt-2 bg-bg-2 border border-line rounded p-3 text-[12px] text-ink-2 overflow-x-auto whitespace-pre-wrap">
            {error.message || t("error.unknown", "Unknown error")}
            {error.digest ? `\n\ndigest: ${error.digest}` : ""}
          </pre>
        </details>
      </div>
    </div>
  );
}

/**
 * A route's own boundary: the failure inside the app's chrome. `name` is the sidebar's English name for
 * the route ("LiDAR", "Map"); it is translated here like every other navigation label.
 *
 * If the chrome itself is what threw, this throws again while rendering it, and the root boundary takes
 * over. That is the layering working, not a second failure.
 */
export default function SegmentError({ error, reset, name }: BoundaryProps & { name: string }) {
  const label = useLabel();
  return (
    <PageShell active={name}>
      <RouteErrorBody error={error} reset={reset} what={label(name)} />
    </PageShell>
  );
}
