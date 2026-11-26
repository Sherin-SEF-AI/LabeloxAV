"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Consensus: being built (see services/api/routers/tools_quality.py).
export default function Page() {
  return (
    <PageShell active="Consensus">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="check" title="Consensus is being built">Several people and models label the same frames; where they agree it is accepted, where they differ it comes here.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
