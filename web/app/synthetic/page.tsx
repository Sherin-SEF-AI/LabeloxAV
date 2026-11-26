"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Synthetic variants: being built (see services/api/routers/tools_sensors.py).
export default function Page() {
  return (
    <PageShell active="Synthetic variants">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="wand" title="Synthetic variants is being built">Rain, night and fog versions of hard frames, with their labels carried over.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
