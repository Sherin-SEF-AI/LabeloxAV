"use client";

import TopNav from "@/components/TopNav";
import PageHeaderBar from "./PageHeaderBar";
import Sidebar from "./Sidebar";
import { usePathname } from "next/navigation";
import { titleFor } from "@/lib/sidebar";
import Breadcrumbs, { type Crumb } from "./Breadcrumbs";

// The single dashboard chrome wrapper. Renders TopNav (its {active, right} contract preserved verbatim, so
// the menu bar + Cmd+K palette keep working), then an optional title/primary-action bar, then an
// optional filters band, then the content. Pure chrome and opt-in: a page adopts it by replacing its outer
// min-h-screen flex + TopNav + ad-hoc header with one PageShell, moving no data logic. The title bar is
// omitted entirely when there is nothing to show in it, so a page that only wants the wrapped TopNav still
// gets consistent structure without an empty band. Content scrolls under fixed chrome.

export default function PageShell({ active, title, subtitle, right, primaryAction, meta, filters, crumbs, children }: {
  active: string;
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  primaryAction?: React.ReactNode;
  meta?: React.ReactNode;
  filters?: React.ReactNode;
  crumbs?: Crumb[];   // explicit trail for pages with real context (a frame that knows its session); else auto
  children: React.ReactNode;
}) {
  // A page in the sidebar is titled with the sidebar's name for it, so the header always matches what was
  // clicked. Pages set their own titles over the years, several still carrying the retired platform code
  // names; a page outside the sidebar (a single object, a track) keeps the title it passes.
  const nav = titleFor(usePathname() ?? "");
  const shownTitle = nav?.title ?? title;
  const showBar = Boolean(shownTitle || primaryAction || meta || subtitle);
  return (
    // min-h-0 on the column and overflow on the bands: without it a narrow viewport clips the filter row
    // and the content beneath it rather than letting either scroll.
    // The sidebar beside a column holding the top bar, the page's title and filter bands, and the content.
    <div className="h-[100dvh] flex min-h-0">
      {/* First in the tab order, before the sidebar, hidden until focused: a keyboard user should not have
          to walk the whole navigation to reach what they came for. */}
      <a href="#content" className="skip-link">skip to content</a>
      <Sidebar />
    <div className="flex-1 min-w-0 flex flex-col min-h-0">
      {/* The header crumb only appears when the page has no title bar of its own, so the location is never
          spelled out twice in two different styles directly above itself. */}
            {/* First in the tab order, hidden until focused. Every page here puts a menu bar, a
          header and often a filter band before the content, and a keyboard user should not have
          to walk all of it to reach what they came for. */}
      <TopNav active={active} right={right} showCrumb={!showBar} />
      {/* The breadcrumb trail: history-aware back + the path to here, on every wrapped page. */}
      {crumbs && crumbs.length > 0 && (
        <div className="flex items-center px-4 py-1.5 border-b hairline shrink-0 overflow-x-auto no-scrollbar">
          <Breadcrumbs items={crumbs} />
        </div>
      )}
      {showBar && <PageHeaderBar title={shownTitle ?? active} parent={nav?.parent} subtitle={subtitle} meta={meta}
        primaryAction={primaryAction} />}
      {filters && (
        <div className="flex items-center gap-2 px-4 py-2 border-b hairline shrink-0 font-mono text-[11px] overflow-x-auto no-scrollbar">
          {filters}
        </div>
      )}
      <main id="content" className="flex-1 min-h-0 overflow-auto">{children}</main>
    </div>
    </div>
  );
}
