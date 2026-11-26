"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Scene reconstruction: being built (see services/api/routers/tools_sensors.py).
export default function Page() {
  return (
    <PageShell active="Scene reconstruction">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="cube" title="Scene reconstruction is being built">Rebuild a drive in 3D, label it there, and project the labels back to every frame.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
