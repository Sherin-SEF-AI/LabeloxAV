"use client";

import TopNav from "@/components/TopNav";
import PageHeaderBar from "./PageHeaderBar";

// The single dashboard chrome wrapper. Renders TopNav (its {active, right} contract preserved verbatim, so
// the grouped AppSwitcher + Cmd+K palette keep working), then an optional title/primary-action bar, then an
// optional filters band, then the content. Pure chrome and opt-in: a page adopts it by replacing its outer
// min-h-screen flex + TopNav + ad-hoc header with one PageShell, moving no data logic. The title bar is
// omitted entirely when there is nothing to show in it, so a page that only wants the wrapped TopNav still
// gets consistent structure without an empty band. Content scrolls under fixed chrome.

export default function PageShell({ active, title, subtitle, right, primaryAction, meta, filters, children }: {
  active: string;
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  primaryAction?: React.ReactNode;
  meta?: React.ReactNode;
  filters?: React.ReactNode;
  children: React.ReactNode;
}) {
  const showBar = Boolean(title || primaryAction || meta || subtitle);
  return (
    <div className="h-screen flex flex-col">
      <TopNav active={active} right={right} />
      {showBar && <PageHeaderBar title={title ?? active} subtitle={subtitle} meta={meta} primaryAction={primaryAction} />}
      {filters && (
        <div className="flex items-center gap-2 px-4 py-2 border-b hairline shrink-0 font-mono text-[11px] overflow-x-auto no-scrollbar">
          {filters}
        </div>
      )}
      <main className="flex-1 min-h-0 overflow-auto">{children}</main>
    </div>
  );
}
