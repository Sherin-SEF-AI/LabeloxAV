"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
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
  const router = useRouter();
  const [jobs, setJobs] = useState<JobsSummary>({ running: [], waiting: 0 });
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
      onClick={() => router.push(view.href)}
      title={view.tip}
      aria-label={view.tip}
      data-state={view.kind}
      className={`btn text-[11px] gap-1.5 h-7 items-center ${working ? "" : "opacity-60 hover:opacity-100"}`}>
      {/* Only running work gets the live dot. A pulsing dot over parked jobs is the lie the tooltip exists
          to correct, and putting it on the chip itself would make the correction unreadable. */}
      {working
        ? <span className="running-dot" />
        : <span className="w-1.5 h-1.5 rounded-full bg-ink-3 inline-block" />}
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
