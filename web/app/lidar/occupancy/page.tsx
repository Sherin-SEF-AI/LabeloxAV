"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Occupancy: being built (see services/api/routers/tools_scene3d.py).
export default function Page() {
  return (
    <PageShell active="Occupancy">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="grid" title="Occupancy is being built">Which space around the vehicle is filled, in 3D cubes, built from LiDAR or depth and corrected by hand.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
