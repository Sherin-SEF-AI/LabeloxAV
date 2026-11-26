"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Camera-LiDAR alignment: being built (see services/api/routers/tools_driving.py).
export default function Page() {
  return (
    <PageShell active="Camera-LiDAR alignment">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="ruler" title="Camera-LiDAR alignment is being built">Drag the LiDAR points onto the image until they line up, with the error shown live.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
