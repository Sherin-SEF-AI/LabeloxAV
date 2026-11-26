"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Radar and events: being built (see services/api/routers/tools_sensors.py).
export default function Page() {
  return (
    <PageShell active="Radar and events">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="activity" title="Radar and events is being built">Radar returns and event-camera streams, shown over the camera image.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
