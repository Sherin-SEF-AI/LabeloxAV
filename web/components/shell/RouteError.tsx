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
    <div className="p-8 max-w-2xl mx-auto space-y-4" role="alert">
      <h1 className="font-display text-lg text-ink">
        {what ? t("error.segment_title", "{what} stopped working", { what }) : t("error.title", "Something broke on this page")}
      </h1>
      <p className="text-[12.5px] text-ink-3">
        {what
          ? t("error.segment_body", "The rest of the app is fine and the sidebar still works. Retrying re-renders just this page; if it keeps failing, the message below is what to report.")
          : t("error.body", "The rest of the app is fine. Retrying re-renders just this page; if it keeps failing, the message below is what to report.")}
      </p>
      <pre className="panel p-3 text-[12px] text-warn overflow-x-auto whitespace-pre-wrap">
        {error.message || t("error.unknown", "unknown error")}
        {error.digest ? `\n\ndigest: ${error.digest}` : ""}
      </pre>
      <div className="flex items-center gap-2">
        <button onClick={reset} className="btn btn-on">{t("action.retry", "retry")}</button>
        <a href={home} className="text-[12.5px] border border-line text-ink-2 px-3 py-1 hover:border-accent">
          {what ? t("error.back_home", "back to home") : t("error.back_triage", "back to triage")}
        </a>
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
