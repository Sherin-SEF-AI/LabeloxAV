"use client";

import PageShell from "./PageShell";
import LoadState from "./LoadState";
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
      <div className="max-w-2xl mx-auto px-4 sm:px-6 py-10">
        {!loaded ? (
          <LoadState loading />
        ) : (
          <section className="panel p-5">
            <h2 className="text-[15px] font-semibold text-ink">Not installed on this server</h2>
            <p className="mt-2 text-[13px] text-ink-2 max-w-[65ch]">
              {title} is part of an optional package: {what}. It is off unless an administrator installs it.
            </p>
            <p className="mt-3 text-[12.5px] text-ink-3">To install it, add the package in the server&apos;s .env and restart:</p>
            <code className="mt-1.5 block text-[12.5px] text-ink-2 bg-bg-2 border hairline rounded px-3 py-2">
              LBX_PACKS__OPTIONAL=[&quot;{pack}&quot;]
            </code>
          </section>
        )}
      </div>
    </PageShell>
  );
}
