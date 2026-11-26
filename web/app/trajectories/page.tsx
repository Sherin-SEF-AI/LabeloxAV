"use client";

import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";

// Paths and intent: being built (see services/api/routers/tools_driving.py).
export default function Page() {
  return (
    <PageShell active="Paths and intent">
      <div className="page-wrap">
        <div className="panel">
          <EmptyState icon="route" title="Paths and intent is being built">The vehicle&apos;s path, where other road users will go, and what they mean to do.</EmptyState>
        </div>
      </div>
    </PageShell>
  );
}
