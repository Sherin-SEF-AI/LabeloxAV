"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { type JobsSummary, subscribeJobSummary } from "@/lib/jobStream";
import { subscribeUploads, type UploadState } from "@/lib/uploadManager";
import { summarize } from "@/lib/uploadQueue";

// What the system is doing, in the top bar, on every page.
//
// This first shipped watching only `uploadManager`, which is module-scoped browser state. That was the right
// fix for surviving client-side navigation and the wrong scope for a global indicator: a reload or a second
// tab empties it while the server carries on, and it never knew about work it did not start. It could sit
// showing nothing with twelve jobs live on the server, which is what was reported.
//
// So it reads two sources, because they cover different halves of the same batch:
//
//   the jobs stream, for anything the server is running (import, autolabel, training, export). Survives
//   reloads and new tabs, because the truth is on the server rather than in this document. Seeded once from
//   REST, because the stream sleeps its interval before the first frame and half a minute of blank top bar
//   after every page load is the same bug in a different place.
//
//   the local queue, for the upload phase only. Bytes leaving this browser have no server job until the
//   import starts, so dropping it would leave the first minutes of a 41GB batch invisible.
//
// Waiting jobs are counted but never shown as running. This deployment holds 67 autolabel jobs pending since
// late June and 11 parked for a cloud A100; a chip reading "78 running" forever teaches people to ignore it.
// They belong in the tooltip, because "why is nothing happening" is a real question with that as its answer.

export default function UploadIndicator() {
  const router = useRouter();
  const [jobs, setJobs] = useState<JobsSummary>({ running: [], waiting: 0 });
  const [local, setLocal] = useState<UploadState | null>(null);

  useEffect(() => subscribeJobSummary(setJobs), []);
  useEffect(() => subscribeUploads(setLocal), []);

  const localRunning = !!local?.running;
  const localSummary = local ? summarize(local.items) : null;

  // The upload phase only. Once an item reaches `importing` the server has a job for it and that job is on
  // the stream, so counting both would show one clip twice.
  const uploading = localRunning
    ? (local?.items ?? []).filter((i) => i.status === "uploading").length
    : 0;

  const serverRunning = jobs.running.length;
  const total = serverRunning + uploading;

  if (total === 0) return null;

  // Averaged across everything in flight. A single number is the only thing that fits, and it is honest as
  // long as it never claims completion while something is still going.
  const fracs = [
    ...jobs.running.map((r) => r.progress),
    ...(localRunning ? (local?.items ?? []).filter((i) => i.status === "uploading").map((i) => i.progress) : []),
  ];
  const pct = fracs.length
    ? Math.min(99, Math.round((fracs.reduce((a, b) => a + b, 0) / fracs.length) * 100))
    : 0;

  const verb = uploading > 0 && serverRunning === 0 ? "uploading"
    : local?.phase === "autolabeling" ? "labelling"
    : jobs.running.every((r) => r.kind === "autolabel") && serverRunning > 0 ? "labelling"
    : "working";

  const tip = [
    `${total} running`,
    jobs.waiting > 0 ? `${jobs.waiting} queued and not progressing` : null,
    localSummary && localRunning ? `${localSummary.done + localSummary.failed}/${localSummary.total} in this tab` : null,
    "click to watch",
  ].filter(Boolean).join(" · ");

  return (
    <button
      // The local queue has a page of its own; a server job only has the jobs list. Sending somebody to an
      // empty upload page to watch an autolabel run started in another tab would be a dead end.
      onClick={() => router.push(localRunning ? "/annotate/new" : "/jobs")}
      title={tip}
      className="btn text-[11px] gap-1.5 h-7 items-center">
      <span className="running-dot" />
      <span className="hidden sm:inline text-ink-2">{verb}</span>
      <span className="text-ink-3">{total}</span>
      <span className="hidden md:block w-12 h-1 bg-line rounded overflow-hidden">
        <span className="block h-full bg-accent transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%` }} />
      </span>
    </button>
  );
}
