"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

// The location trail, so you can see where you are in the platform -> tool -> session -> frame -> object
// hierarchy and click up a level instead of guessing. Computed from the pathname by default; a page with
// real context (a frame that knows its session) passes explicit `items` to name the middle of the trail.
//
// The leading control is a history-aware back: it returns to wherever you actually came from, falling back to
// the parent crumb when there is no history (a deep link or a fresh tab). This is the one place "back" lives,
// so it behaves the same everywhere instead of each page hard-coding a destination.

export type Crumb = { label: string; href?: string };

// Friendly names for path segments; anything not listed is title-cased. Dynamic id segments are shortened.
const LABELS: Record<string, string> = {
  "": "Home", explore: "Explore", search: "Search", analytics: "Analytics", govern: "Governance",
  review: "Review", queue: "Queue", datasets: "Datasets", jobs: "Jobs", projects: "Projects",
  quality: "Quality", curation: "Curation", discovery: "Discovery", scenarios: "Scenarios",
  training: "Training", calibration: "Calibration", inspect: "Inspector", map: "Map", lidar: "LiDAR",
  annotate: "Annotate", multicam: "Multi-camera", frame: "Frame", object: "Object", annotations: "Sessions",
  platforms: "Platforms", integrations: "Integrations", agent: "Agent", import: "Import", inertial: "Inertial",
  collaborate: "Collaborate", linked: "Linked", "new": "New",
};

function humanSeg(seg: string): string {
  if (LABELS[seg]) return LABELS[seg];
  // a uuid or long id -> short form
  if (/^[0-9a-f-]{16,}$/i.test(seg)) return seg.slice(0, 8);
  return seg.charAt(0).toUpperCase() + seg.slice(1).replace(/[-_]/g, " ");
}

function fromPath(pathname: string): Crumb[] {
  const segs = pathname.split("/").filter(Boolean);
  const out: Crumb[] = [{ label: "Home", href: "/" }];
  let acc = "";
  segs.forEach((s, i) => {
    acc += "/" + s;
    out.push({ label: humanSeg(s), href: i < segs.length - 1 ? acc : undefined });
  });
  return out;
}

export default function Breadcrumbs({ items, fallback = "/" }: { items?: Crumb[]; fallback?: string }) {
  const pathname = usePathname();
  const router = useRouter();
  const crumbs = items && items.length ? items : fromPath(pathname || "/");

  const goBack = () => {
    if (typeof window !== "undefined" && window.history.length > 1) router.back();
    else router.push(crumbs.length > 1 ? crumbs[crumbs.length - 2].href ?? fallback : fallback);
  };

  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 min-w-0 font-mono text-[11px]">
      <button onClick={goBack} title="back (Alt+Left)" aria-label="Go back"
        className="text-ink-3 hover:text-accent border border-line hover:border-accent px-1.5 py-0.5 shrink-0">
        &larr;
      </button>
      <ol className="flex items-center gap-1.5 min-w-0 overflow-hidden">
        {crumbs.map((c, i) => (
          <li key={i} className="flex items-center gap-1.5 min-w-0">
            {i > 0 && <span className="text-ink-4" aria-hidden>/</span>}
            {c.href ? (
              <Link href={c.href} className="text-ink-3 hover:text-accent truncate">{c.label}</Link>
            ) : (
              <span className="text-ink-2 truncate" aria-current="page">{c.label}</span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
