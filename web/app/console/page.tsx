"use client";

import { useRouter } from "next/navigation";
import PageShell from "@/components/shell/PageShell";
import {
  BackgroundPanel, GpuPanel, HostPanel, ProcessPanel, RunningNowPanel, useAgentRuns,
} from "@/components/console/Panels";
import { headline, uptime, workInFlight } from "@/lib/console";
import { useJobStream, useSystemStream } from "@/lib/useEventStream";
import { num, plural } from "@/lib/format";

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
        <span className="flex items-center gap-2 text-[12px] text-ink-3">
          {/* A console whose numbers freeze looks exactly like a quiet machine, so it says which it is. */}
          <span className={`w-1.5 h-1.5 rounded-full ${live ? "bg-pass" : "bg-ink-3"}`} />
          <span>{live ? "Live" : "Not connected"}</span>
          {sys && <span>· up {uptime(sys.process.uptime_s)}</span>}
        </span>
      }
    >
      <div className="page-wrap space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <section className="panel">
            <div className="panel-head"><h2 className="section-title">GPU</h2></div>
            <div className="p-3.5"><GpuPanel sys={sys} /></div>
          </section>
          <section className="panel">
            <div className="panel-head"><h2 className="section-title">Machine</h2></div>
            <div className="p-3.5"><HostPanel sys={sys} /></div>
          </section>
          <section className="panel">
            <div className="panel-head"><h2 className="section-title">API process</h2></div>
            <div className="p-3.5"><ProcessPanel sys={sys} waitingTotal={waitingTotal} /></div>
          </section>
        </div>

        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Running now</h2>
            <span className="text-ink-3 text-[12px]">
              {running.length ? plural(running.length, "job") : "Nothing running"}
            </span>
            <button type="button" onClick={() => router.push("/jobs")} className="btn ml-auto">
              All jobs
            </button>
          </div>
          <div className="px-4">
            <RunningNowPanel jobs={jobs} />
          </div>
        </section>

        {/* Background agent work, which is the part that used to live only in a log file. */}
        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Background work</h2>
            <span className="text-ink-3 text-[12px]">
              {activeRuns.length ? `${num(activeRuns.length)} active` : "Corpus sweeps, relabels and review batches"}
            </span>
            <button type="button" onClick={() => router.push("/agent")} className="btn ml-auto">
              Agent console
            </button>
          </div>
          <div className="px-4">
            <BackgroundPanel runs={runs} />
          </div>
        </section>
      </div>
    </PageShell>
  );
}
