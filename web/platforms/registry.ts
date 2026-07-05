// The platform registry (frontend mirror of platforms/registry.py). LabeloxAV is the host; the annotation
// core and the six folded subsystems are Platforms: distinct navigable UIs behind a launcher, over one shared
// backend spine. This drives the launcher, the platform switcher, and the cross-platform session view.
//
// Operational Materialism: a platform tile is neutral graphite. Color is earned only by live state (a SANYX
// quarantine count, a CALYX block, a VERDYX pending verdict), never by decoration. Keep this file in sync with
// platforms/registry.py; GET /api/platforms exposes the backend copy so the two can be reconciled.

export type PlatformId = "labelox" | "sanyx" | "calyx" | "sievyx" | "oraclyx" | "verdyx" | "forgyx";
export type PlatformGate = "health" | "calibration" | "eval" | "benchmark" | null;
export type PlatformNavItem = { href: string; label: string; hint?: string };

export type Platform = {
  id: PlatformId;
  label: string;
  role: string;
  glyph: string;            // short mono rail glyph
  order: number;
  gate: PlatformGate;       // this platform can block progression / promotion
  flywheelStage: number | null;
  home: string;             // route the launcher tile opens
  nav: PlatformNavItem[];   // destinations within the platform (existing routes today)
};

export const PLATFORMS: Platform[] = [
  {
    id: "labelox", label: "Labelox", role: "annotation core: three-path auto-label + human workspace",
    glyph: "LBX", order: 0, gate: null, flywheelStage: 4, home: "/",
    nav: [
      { href: "/", label: "Triage", hint: "object queue ranked by value" },
      { href: "/agent", label: "Agent Console", hint: "autonomous QA and fix queue" },
      { href: "/review/queue", label: "Review queue", hint: "active learning + error candidates" },
      { href: "/annotations", label: "Annotations", hint: "browse and resume sessions" },
      { href: "/jobs", label: "Jobs", hint: "import, training, autolabel runs" },
    ],
  },
  {
    id: "sanyx", label: "SANYX", role: "ingest QA: health score, quarantine bad sessions",
    glyph: "SNX", order: 1, gate: "health", flywheelStage: 1, home: "/inspect",
    nav: [
      { href: "/inspect", label: "Inspector", hint: "MCAP session inspection" },
      { href: "/sanyx", label: "Ingest board", hint: "health scores and quarantine (M1)" },
    ],
  },
  {
    id: "calyx", label: "CALYX", role: "calibration monitor: extrinsic / intrinsic drift",
    glyph: "CLX", order: 2, gate: "calibration", flywheelStage: 2, home: "/calibration",
    nav: [{ href: "/calibration", label: "Calibration", hint: "camera validation and drift" }],
  },
  {
    id: "sievyx", label: "SIEVYX", role: "curation and mining: embed, rank, dedup, decide what to label",
    glyph: "SVX", order: 3, gate: null, flywheelStage: 3, home: "/curation",
    nav: [
      { href: "/curation", label: "Curation", hint: "frame-level active learning" },
      { href: "/search", label: "Search", hint: "visual and semantic similarity" },
      { href: "/discovery", label: "Discovery", hint: "rare-scenario novelty queue" },
      { href: "/scenarios", label: "Scenarios", hint: "behavioral scenario mining" },
    ],
  },
  {
    id: "oraclyx", label: "ORACLYX", role: "offline-fusion pseudo-GT: auto-truth the majority, route disagreements",
    glyph: "ORX", order: 4, gate: null, flywheelStage: 5, home: "/lidar",
    nav: [
      { href: "/lidar", label: "LiDAR", hint: "point cloud explorer" },
      { href: "/map", label: "HD map", hint: "fused map and provenance" },
      { href: "/inertial", label: "Inertial", hint: "ego-state timeline and events" },
    ],
  },
  {
    id: "verdyx", label: "VERDYX", role: "slice evaluation: per-slice regression + champion / challenger verdict",
    glyph: "VDX", order: 5, gate: "eval", flywheelStage: 7, home: "/training",
    nav: [
      { href: "/training", label: "Training", hint: "training jobs and model runs" },
      { href: "/govern", label: "Govern", hint: "loop control and championship" },
      { href: "/quality", label: "Quality", hint: "gold sets and slice metrics" },
      { href: "/analytics", label: "Analytics", hint: "corpus health and loop signal" },
    ],
  },
  {
    id: "forgyx", label: "FORGYX", role: "edge optimization: quantize, compile, benchmark, gate on latency and accuracy",
    glyph: "FGX", order: 6, gate: "benchmark", flywheelStage: 8, home: "/datasets",
    nav: [
      { href: "/datasets", label: "Datasets", hint: "sealed dataset delivery" },
      { href: "/forgyx", label: "Deployments", hint: "benchmark matrix and artifacts (M8)" },
    ],
  },
];

export const PLATFORMS_ORDERED = [...PLATFORMS].sort((a, b) => a.order - b.order);

export function platformById(id: string): Platform | undefined {
  return PLATFORMS.find((p) => p.id === id);
}

export function platformForPath(path: string): Platform | undefined {
  // Longest home/nav prefix wins, so /training resolves to VERDYX rather than the Labelox root.
  let best: Platform | undefined;
  let bestLen = -1;
  for (const p of PLATFORMS) {
    for (const href of [p.home, ...p.nav.map((n) => n.href)]) {
      if (href !== "/" && path.startsWith(href) && href.length > bestLen) {
        best = p;
        bestLen = href.length;
      }
    }
  }
  return best ?? platformById("labelox");
}
