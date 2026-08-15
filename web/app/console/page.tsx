"use client";

import { useRouter } from "next/navigation";
import PageShell from "@/components/shell/PageShell";
import {
  BackgroundPanel, GpuPanel, HostPanel, ProcessPanel, RunningNowPanel, useAgentRuns,
} from "@/components/console/Panels";
import { headline, uptime, workInFlight } from "@/lib/console";
import { useJobStream, useSystemStream } from "@/lib/useEventStream";

// One place that answers "is anything happening".
//
// It was three places before, and two of them were not in the application. Job state was on /jobs, the
// machine was in a terminal, and long-running background work was in a log file on the box. So the answer to
// "why is this slow" required a shell, which the people doing the labelling do not have.
//
// This route is the deep-linkable, in-the-menu form. The same panels also appear in the dialog that opens
// over the current page (components/console/ConsoleModal), which is how the question actually gets asked:
// in the middle of something you cannot afford to navigate away from. Both read the same components, so the
// one nobody happened to be looking at cannot be the one that drifted.

export default function ConsolePage() {
  const router = useRouter();
  const { data: jobs, connected: jobsLive } = useJobStream();
  const { data: sys, connected: sysLive } = useSystemStream();
  const runs = useAgentRuns();

  const { running, waitingTotal } = workInFlight(jobs);
  const live = jobsLive && sysLive;
  const activeRuns = runs.filter((r) => r.status === "running" || r.status === "planned");

  return (
    <PageShell
      active="CONSOLE"
      title="Console"
      subtitle={headline(jobs, sys)}
      right={
        <span className="flex items-center gap-2 font-mono text-[11px] text-ink-3">
          {/* A console whose numbers freeze looks exactly like a quiet machine, so it says which it is. */}
          <span className={`w-1.5 h-1.5 rounded-full ${live ? "bg-pass" : "bg-ink-3"}`} />
          <span>{live ? "live" : "not connected"}</span>
          {sys && <span>· up {uptime(sys.process.uptime_s)}</span>}
        </span>
      }
    >
      <div className="p-4 space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="panel p-3 space-y-3">
            <div className="text-ink font-medium text-sm">GPU</div>
            <GpuPanel sys={sys} />
          </div>
          <div className="panel p-3 space-y-3">
            <div className="text-ink font-medium text-sm">Machine</div>
            <HostPanel sys={sys} />
          </div>
          <div className="panel p-3 space-y-3">
            <div className="text-ink font-medium text-sm">API process</div>
            <ProcessPanel sys={sys} waitingTotal={waitingTotal} />
          </div>
        </div>

        <div className="panel">
          <div className="flex items-center gap-3 px-4 py-2 border-b hairline">
            <span className="text-ink font-medium text-sm">Running now</span>
            <span className="text-ink-3 text-xs">
              {running.length ? `${running.length} job${running.length === 1 ? "" : "s"}` : "nothing"}
            </span>
            <button onClick={() => router.push("/jobs")}
              className="ml-auto font-mono text-[10px] border border-line px-1.5 py-0.5 text-ink-3 hover:border-accent">
              all jobs
            </button>
          </div>
          <div className="px-4">
            <RunningNowPanel jobs={jobs} />
          </div>
        </div>

        {/* Background agent work, which is the part that used to live only in a log file. */}
        <div className="panel">
          <div className="flex items-center gap-3 px-4 py-2 border-b hairline">
            <span className="text-ink font-medium text-sm">Background</span>
            <span className="text-ink-3 text-xs">
              {activeRuns.length ? `${activeRuns.length} active` : "corpus sweeps, relabels, batches"}
            </span>
            <button onClick={() => router.push("/agent")}
              className="ml-auto font-mono text-[10px] border border-line px-1.5 py-0.5 text-ink-3 hover:border-accent">
              agent console
            </button>
          </div>
          <div className="px-4">
            <BackgroundPanel runs={runs} />
          </div>
        </div>
      </div>
    </PageShell>
  );
}
