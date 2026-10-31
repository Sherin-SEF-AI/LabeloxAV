// The application's navigation: one definition the sidebar renders and the tests check.
//
// It replaces two systems that had grown side by side: a Blender-style menu bar of ten menus and seventy
// entries, and a switcher between seven "platforms" with invented names (SANYX, CALYX, SIEVYX...). The same
// page often sat in a different group in each, and a new person on a team had no way to tell which pages
// were the daily workflow and which were research instruments.
//
// So the sidebar is organised by what a small labelling team does, in plain words, and everything that is
// not part of that daily loop lives under Labs: still one click away for whoever wants it, hidden by default
// so it does not crowd out the work. The command palette (Cmd K) still reaches every page, Labs included.

export type NavItem = {
  key: string;
  label: string;
  href: string;
  icon: string;
  hint: string;
  // Other path prefixes that belong to this entry, so it stays highlighted on its sub-pages
  // (/review/rapid is still "Review").
  match?: string[];
  // Sub-pages, listed under the entry while you are inside it. Several of these were reachable only from the
  // platform switcher this sidebar replaced; none of their parent pages links to them.
  children?: { label: string; href: string }[];
};

export type NavSection = {
  key: string;
  label: string;
  items: NavItem[];
  // Labs sections are hidden until someone switches Labs on (Settings > Preferences), or opens one of their
  // pages directly, in which case the section shows so the page is never orphaned from the navigation.
  labs?: boolean;
  // Belongs to an optional pack (settings.packs.optional on the server); shown only when that pack is
  // installed, because its routes do not exist otherwise.
  pack?: string;
};

export const NAV: NavSection[] = [
  {
    key: "work",
    label: "Work",
    items: [
      { key: "home", label: "Home", href: "/", icon: "home",
        hint: "what is waiting, what is running, and where to start" },
      { key: "queue", label: "My queue", href: "/queue", icon: "target",
        hint: "objects waiting for a decision, most valuable first" },
      { key: "review", label: "Review", href: "/review/queue", icon: "check", match: ["/review"],
        hint: "auto-labels and likely mistakes waiting for a person",
        children: [{ label: "Queue", href: "/review/queue" }, { label: "Rapid review", href: "/review/rapid" },
                   { label: "Crop grid", href: "/review/grid" }, { label: "Attribute sweep", href: "/annotate/attrsweep" }] },
      { key: "projects", label: "Projects", href: "/projects", icon: "clipboard",
        hint: "labelling projects, assignments and stages" },
      { key: "errors", label: "Label errors", href: "/quality/errors", icon: "flag",
        hint: "labels the models think are wrong, to confirm or dismiss" },
    ],
  },
  {
    key: "data",
    label: "Data",
    items: [
      { key: "sessions", label: "Sessions", href: "/annotations", icon: "folder", match: ["/annotate"],
        hint: "every recorded drive and upload; open one to label it" },
      { key: "upload", label: "Upload", href: "/annotate/new", icon: "upload",
        hint: "add video or images and start labelling" },
      { key: "import", label: "Import", href: "/import", icon: "database",
        hint: "bring in a labelled dataset (COCO, YOLO, KITTI, nuScenes, CVAT...)",
        children: [{ label: "Import a dataset", href: "/import" }, { label: "Migrate from another tool", href: "/import/migrate" }] },
      { key: "explore", label: "Explore", href: "/explore", icon: "scan",
        hint: "filter, tag and map the corpus" },
      { key: "search", label: "Search", href: "/search", icon: "search",
        hint: "find frames by example or by description" },
      { key: "datasets", label: "Datasets", href: "/datasets", icon: "layers",
        hint: "versioned exports in standard formats",
        children: [{ label: "Datasets", href: "/datasets" }, { label: "Dataset query", href: "/datasets/query" }] },
    ],
  },
  {
    key: "scenes",
    label: "3D and scenes",
    items: [
      { key: "lidar", label: "LiDAR", href: "/lidar", icon: "cube",
        hint: "point clouds and 3D cuboids",
        children: [{ label: "Point clouds", href: "/lidar" }, { label: "3D cuboids", href: "/lidar/annotate" },
                   { label: "3D to camera", href: "/lidar/linked" }] },
      { key: "events", label: "Driving events", href: "/events", icon: "activity",
        hint: "cut-ins, hard brakes and near-misses found in the drives",
        children: [{ label: "Events", href: "/events" }, { label: "Event search", href: "/events/search" }] },
      { key: "scenarios", label: "Scenarios", href: "/scenarios", icon: "route",
        hint: "behaviour patterns mined across sessions" },
      { key: "map", label: "Map", href: "/map", icon: "target",
        hint: "where the data was recorded, and the fused road map" },
      { key: "inspect", label: "Session inspector", href: "/inspect", icon: "grid",
        hint: "sensor timeline and health of one recording" },
    ],
  },
  {
    key: "models",
    label: "Models",
    items: [
      { key: "training", label: "Training", href: "/training", icon: "chip",
        hint: "train and compare detectors on your labels" },
      { key: "quality", label: "Evaluation", href: "/quality", icon: "chart", match: ["/labelox/quality"],
        hint: "accuracy on the gold set, per class",
        children: [{ label: "Gold set", href: "/quality" }, { label: "Label audit", href: "/labelox/quality" }] },
      { key: "govern", label: "Model registry", href: "/govern", icon: "shield",
        match: ["/govern"], hint: "which model labels new data, and the promotion gate",
        children: [{ label: "Registry", href: "/govern" }, { label: "Compliance", href: "/govern/compliance" },
                   { label: "Personal data access", href: "/govern/pii-access" }] },
      { key: "analytics", label: "Analytics", href: "/analytics", icon: "activity",
        hint: "corpus coverage, label throughput and where accuracy is lacking" },
      { key: "jobs", label: "Jobs", href: "/jobs", icon: "server",
        hint: "auto-label, import, export and training runs" },
    ],
  },
  {
    key: "labs",
    label: "Labs",
    labs: true,
    items: [
      { key: "agent", label: "Agent console", href: "/agent", icon: "robot",
        hint: "automated QA and fix proposals" },
      { key: "autonomy", label: "Autonomy", href: "/autonomy", icon: "target",
        hint: "what the system may accept on its own, per class" },
      { key: "campaigns", label: "Campaigns", href: "/campaigns", icon: "flag",
        hint: "improvement loops the system runs" },
      { key: "curation", label: "Curation", href: "/curation", icon: "layers",
        match: ["/sievyx", "/discovery"], hint: "what to label next, rare scenes and long tail",
        children: [{ label: "Frame selection", href: "/curation" }, { label: "Label budget", href: "/sievyx" },
                   { label: "Long tail", href: "/sievyx/longtail" }, { label: "Discovery", href: "/discovery" }] },
      { key: "ingest-health", label: "Ingest health", href: "/sanyx", icon: "eye",
        hint: "recording quality scores and quarantine",
        children: [{ label: "Health board", href: "/sanyx" }, { label: "Predictive", href: "/sanyx/predictive" }] },
      { key: "calibration", label: "Calibration", href: "/calyx", icon: "ruler",
        match: ["/calibration"], hint: "camera calibration drift per rig",
        children: [{ label: "Rig timeline", href: "/calyx" }, { label: "Recovery", href: "/calyx/recovery" },
                   { label: "Camera validation", href: "/calibration" }] },
      { key: "pseudo-gt", label: "Pseudo ground truth", href: "/oraclyx", icon: "wand",
        hint: "offline fusion that auto-truths the confident majority",
        children: [{ label: "Consensus board", href: "/oraclyx" }, { label: "4D pseudo ground truth", href: "/oraclyx/pseudogt" }] },
      { key: "slice-eval", label: "Slice evaluation", href: "/verdyx", icon: "chart",
        hint: "champion against challenger, per slice",
        children: [{ label: "Slice matrix", href: "/verdyx" }, { label: "Safety evaluation", href: "/verdyx/safety" }] },
      { key: "edge", label: "Edge deployment", href: "/forgyx", icon: "terminal",
        hint: "quantise, benchmark and package models for vehicles",
        children: [{ label: "Latency and accuracy", href: "/forgyx" }, { label: "Deploy", href: "/forgyx/deploy" },
                   { label: "Field performance", href: "/forgyx/field" }] },
      { key: "scorecards", label: "Labeller scorecards", href: "/scorecards", icon: "person",
        hint: "accuracy per person and vendor, per class" },
      { key: "reasoner", label: "Reasoning trace", href: "/reasoner", icon: "list",
        hint: "what ran before each auto-label, and whether it was right" },
      { key: "lineage", label: "Lineage", href: "/lineage", icon: "route",
        hint: "what a model is made of, and where a session ended up" },
      { key: "collaborate", label: "Branches", href: "/collaborate", icon: "link",
        hint: "dataset branches and merge requests (needs the lakefs profile)" },
      { key: "inertial", label: "Ego motion", href: "/inertial", icon: "activity",
        hint: "vehicle speed, turns and manoeuvres" },
      { key: "ops", label: "Operations", href: "/ops", icon: "server", match: ["/flywheel"],
        hint: "service levels, reproducibility, label efficiency",
        children: [{ label: "Operations", href: "/ops" }, { label: "Adaptive flywheel", href: "/flywheel/adaptive" }] },
    ],
  },
];

// Added to NAV only when the pack is installed (see sectionsFor). LabeloxSec is a separate optional package for
// authorised security deployments; an AV research install never shows it.
export const PACK_SECTIONS: NavSection[] = [
  {
    key: "security",
    label: "Security",
    pack: "sec",
    items: [
      { key: "sec-console", label: "Security console", href: "/labeloxsec", icon: "shield",
        hint: "plate reads, watchlist hits and static-camera sessions" },
      { key: "sec-incidents", label: "Incidents", href: "/labeloxsec/incidents", icon: "flag",
        hint: "zones, line crossings and cross-camera sightings" },
    ],
  },
];

/** The sections to show for an installation: the product, plus each installed optional pack's section. */
export function sectionsFor(enabledPacks: (pack: string) => boolean): NavSection[] {
  return [...NAV, ...PACK_SECTIONS.filter((s) => s.pack && enabledPacks(s.pack))];
}

// Pinned to the bottom of the sidebar rather than in a section: they are about the installation and the
// person, not about the data.
export const NAV_FOOTER: NavItem[] = [
  { key: "settings", label: "Settings", href: "/settings", icon: "gear",
    // Its own pages are linked from the settings page rather than listed here: a pinned footer that grew five
    // rows whenever you opened it would push the sections above out of view on a laptop screen.
    match: ["/integrations", "/profile", "/activity", "/console"],
    hint: "system status, integrations, preferences" },
];

/** Whether `pathname` is at or below `prefix`. "/" matches only itself, or it would match every page. */
function under(pathname: string, prefix: string): boolean {
  if (prefix === "/") return pathname === "/";
  const base = prefix.replace(/\/$/, "");
  return pathname === base || pathname.startsWith(`${base}/`);
}

/** The entry a path belongs to: the one whose href or match prefix is the longest match. */
export function activeKey(pathname: string, sections: NavSection[] = [...NAV, ...PACK_SECTIONS],
                          footer: NavItem[] = NAV_FOOTER): string | null {
  let bestKey: string | null = null;
  let bestLen = -1;
  for (const item of [...sections.flatMap((s) => s.items), ...footer]) {
    for (const prefix of [item.href.split("?")[0], ...(item.match ?? []), ...(item.children ?? []).map((c) => c.href)]) {
      if (under(pathname, prefix) && prefix.length > bestLen) {
        bestKey = item.key;
        bestLen = prefix.length;
      }
    }
  }
  return bestKey;
}

/** Whether a path belongs to a Labs page, so its section can be shown even with Labs switched off. */
export function isLabsPath(pathname: string): boolean {
  const key = activeKey(pathname);
  return NAV.some((s) => s.labs && s.items.some((i) => i.key === key));
}

/** Every sidebar entry, flat, for the command palette and for tests. Optional packs' entries included. */
export const NAV_DESTINATIONS: NavItem[] = [...NAV.flatMap((s) => s.items), ...PACK_SECTIONS.flatMap((s) => s.items),
  ...NAV_FOOTER];

/** The optional pack a path belongs to, or null for a product page. */
export function packOf(href: string): string | null {
  const path = href.split("?")[0];
  for (const s of PACK_SECTIONS) {
    if (s.items.some((i) => path === i.href || path.startsWith(`${i.href}/`))) return s.pack ?? null;
  }
  return null;
}

/** The sidebar's name for a path, as the page header shows it: the entry, and the sub-page when there is one. */
export function titleFor(pathname: string): { title: string; parent?: string } | null {
  for (const d of NAV_DESTINATIONS) {
    const child = (d.children ?? []).find((c) => c.href === pathname);
    if (child) return child.label === d.label ? { title: d.label } : { title: child.label, parent: d.label };
    if (d.href.split("?")[0] === pathname) return { title: d.label };
  }
  return null;
}

/** Every path the sidebar can take you to, sub-pages included. */
export const NAV_HREFS: string[] = NAV_DESTINATIONS.flatMap((d) => [d.href, ...(d.children ?? []).map((c) => c.href)]);
