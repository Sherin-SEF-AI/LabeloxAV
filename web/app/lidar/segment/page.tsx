"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Point labels: being built (see services/api/routers/tools_scene3d.py).
export default function Page() {
  return (
    <PageShell active="Point labels">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="cube" title="Point labels is being built">Label every LiDAR point: brush, lasso, ground plane and clusters.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
