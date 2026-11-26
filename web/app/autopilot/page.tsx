"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Autopilot: being built (see services/api/routers/tools_agents.py).
export default function Page() {
  return (
    <PageShell active="Autopilot">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="play" title="Autopilot is being built">Overnight labelling that only hands people the items it is unsure of.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
