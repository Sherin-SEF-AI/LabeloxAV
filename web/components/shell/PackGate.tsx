"use client";

import PageShell from "./PageShell";
import LoadState from "./LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { packEnabled, useFeaturesState } from "@/lib/features";

// A page that belongs to an optional pack renders only where that pack is installed. Opened by URL on an
// install without it, the page would fire requests at routes that do not exist and show a row of failures;
// this says what the page is and how an administrator adds it instead.

export default function PackGate({ pack, title, what, children }: {
  pack: string; title: string; what: string; children: React.ReactNode;
}) {
  const { features, loaded } = useFeaturesState();
  if (loaded && packEnabled(features, pack)) return <>{children}</>;
  return (
    <PageShell active={title} title={title}>
      <div className="page-wrap-narrow">
        {!loaded ? (
          <LoadState loading />
        ) : (
          <section className="panel">
            <EmptyState icon="shield" title="Not installed on this server"
              action={<a href="/" className="btn">Back to home</a>}>
              {title} is part of an optional package: {what}. It is off unless an administrator installs it.
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-ink-2">For administrators</summary>
                <p className="mt-2">Add the package to the server&apos;s .env and restart:</p>
                <code className="mt-1.5 block text-[12.5px] text-ink-2 bg-bg-2 border hairline rounded px-3 py-2 font-mono">
                  LBX_PACKS__OPTIONAL=[&quot;{pack}&quot;]
                </code>
              </details>
            </EmptyState>
          </section>
        )}
      </div>
    </PageShell>
  );
}
