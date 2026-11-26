"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Rare scenes: being built (see services/api/routers/tools_quality.py).
export default function Page() {
  return (
    <PageShell active="Rare scenes">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="search" title="Rare scenes is being built">The least-covered corners of the data, found by meaning, ready to send for labelling.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
