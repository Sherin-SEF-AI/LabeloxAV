"use client";

import { useEffect, useState } from "react";
import PulseDot from "@/components/PulseDot";
import { openConsole } from "@/components/console/ConsoleModal";
import { indicatorView } from "@/lib/indicatorState";
import { type JobsSummary, subscribeJobSummary } from "@/lib/jobStream";
import { subscribeUploads, type UploadState } from "@/lib/uploadManager";
import { summarize } from "@/lib/uploadQueue";

// What the system is doing, in the top bar, on every page.
//
// This first shipped watching only `uploadManager`, which is module-scoped browser state. That was the right
// fix for surviving client-side navigation and the wrong scope for a global indicator: a reload or a second
// tab empties it while the server carries on, and it never knew about work it did not start.
//
// So it reads two sources, because they cover different halves of the same batch:
//
//   the jobs stream, for anything the server is running (import, autolabel, training, export). Survives
//   reloads and new tabs, because the truth is on the server rather than in this document.
//
//   the local queue, for the upload phase only. Bytes leaving this browser have no server job until the
//   import starts, so dropping it would leave the first minutes of a 41GB batch invisible.
//
// It renders in every state, including idle. The version that returned null when nothing was running was
// reported missing three times, and each report was right: a control you can only find while the system is
// busy cannot be used to ask whether it is. What the chip says is decided in `lib/indicatorState`, which is
// where the rules are tested; this file is the rendering of that answer and nothing else.

export default function UploadIndicator() {
  const [jobs, setJobs] = useState<JobsSummary>({ running: [], waiting: 0, connected: false });
  const [local, setLocal] = useState<UploadState | null>(null);

  useEffect(() => subscribeJobSummary(setJobs), []);
  useEffect(() => subscribeUploads(setLocal), []);

  const localRunning = !!local?.running;
  const localSummary = local ? summarize(local.items) : null;

  const view = indicatorView({
    running: jobs.running,
    waiting: jobs.waiting,
    // The upload phase only. Once an item reaches `importing` the server has a job for it and that job is on
    // the stream, so counting both would show one clip twice.
    uploading: localRunning
      ? (local?.items ?? []).filter((i) => i.status === "uploading").map((i) => i.progress)
      : [],
    localRunning,
    phase: local?.phase,
    localDone: localSummary ? localSummary.done + localSummary.failed : undefined,
    localTotal: localSummary?.total,
  });

  const working = view.kind === "working";

  return (
    <button
      // Opens the console over the current page rather than navigating to it. The chip is most often
      // clicked from the frame editor, which is the one page where being taken somewhere else costs the
      // reader the thing they were looking at.
      onClick={() => openConsole()}
      // The dot carries the connection state and so must the text: a tooltip that says "idle" beside an
      // amber dot leaves the reader to guess which one is right.
      title={jobs.connected ? view.tip : `${view.tip} (not receiving updates: the live connection dropped)`}
      aria-label={jobs.connected ? view.tip : `${view.tip}. Not receiving updates: the live connection dropped.`}
      data-state={view.kind}
      className={`btn text-[11px] gap-1.5 h-7 items-center ${working ? "" : "opacity-60 hover:opacity-100"}`}>
      {/* Three states, not two.
          Only running work gets the live dot: a pulsing dot over parked jobs is the lie the tooltip exists
          to correct. But an idle chip and a chip whose stream has dropped used to render identically, and
          they mean opposite things - one says the system has nothing to do, the other says this chip has
          stopped being told. lib/jobStream.ts has always known the difference and nothing asked. */}
      {!jobs.connected
        ? <PulseDot tone="warn" halo
            label="not receiving updates: the live connection dropped, so this may be out of date" />
        : working
          ? <PulseDot tone="live" label="work in progress" />
          : <PulseDot tone="idle" label="connected, nothing running" />}
      <span className="hidden sm:inline text-ink-2">{view.verb}</span>
      {view.count !== null && <span className="text-ink-3">{view.count}</span>}
      {view.pct !== null && (
        <span className="hidden md:block w-12 h-1 bg-line rounded overflow-hidden">
          <span className="block h-full bg-accent transition-[width] duration-500 ease-out"
            style={{ width: `${view.pct}%` }} />
        </span>
      )}
    </button>
  );
}
