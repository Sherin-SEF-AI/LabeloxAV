// The application menu bar, defined as data.
//
// Blender's model: a small number of top-level menus, each a short list broken into labelled sections by
// separators, with the icon on the left and the shortcut right-aligned. Everything the app can do is reachable
// from here, so a new destination is one entry in this file rather than another button competing for header
// space. The header stays the same size whether the app has 40 destinations or 400.
//
// An item is one of:
//   href      navigate somewhere
//   event     dispatch a window event (the palette, the shortcut sheet, existing in-page actions)
//   items     a submenu
//   disabled  shown but greyed, the way Blender greys Revert when there is nothing to revert

export type MenuItem = {
  key: string;
  label: string;
  icon?: string;
  shortcut?: string;
  href?: string;
  event?: string;
  items?: MenuItem[];
  disabled?: boolean;
  hint?: string;
  separatorBefore?: boolean;
};

export type Menu = { key: string; label: string; items: MenuItem[] };

// Import formats the backend actually registers (services/imports/run.py ADAPTERS + RAW_FORMATS).
const IMPORT_FORMATS: [string, string][] = [
  ["coco", "COCO"], ["yolo", "YOLO"], ["pascalvoc", "Pascal VOC"],
  ["cvat", "CVAT XML"], ["labelstudio", "Label Studio JSON"],
  ["openlabel", "OpenLABEL"], ["nuscenes", "nuScenes"], ["kitti", "KITTI"],
  ["bdd", "BDD100K"], ["mapillary", "Mapillary"], ["parquet", "Parquet (lossless)"],
  ["images", "Images folder"], ["video", "Video"], ["mcap", "MCAP"],
];

// Export targets wired in services/export/dataset.py.
const EXPORT_FORMATS: [string, string][] = [
  ["coco", "COCO"], ["yolo", "YOLO"], ["cvat", "CVAT XML"],
  ["labelstudio", "Label Studio JSON"], ["openlabel", "OpenLABEL"],
  ["nuscenes", "nuScenes (3D)"], ["kitti", "KITTI"], ["bdd", "BDD100K"],
  ["parquet", "Parquet (lossless)"],
];

export const MENUS: Menu[] = [
  {
    key: "file",
    label: "File",
    items: [
      { key: "new", label: "New annotation", icon: "plus", shortcut: "Ctrl N", href: "/annotate/new",
        hint: "start from an upload" },
      { key: "open", label: "Open session", icon: "list", shortcut: "Ctrl O", href: "/annotations",
        hint: "browse and resume sessions" },
      { key: "import", label: "Import", icon: "clipboard", separatorBefore: true,
        items: IMPORT_FORMATS.map(([f, label]) => ({
          key: `import-${f}`, label, href: `/import?format=${f}`,
        })) },
      { key: "export", label: "Export", icon: "save",
        items: EXPORT_FORMATS.map(([f, label]) => ({
          key: `export-${f}`, label, href: `/datasets?format=${f}`,
        })) },
      { key: "datasets", label: "Datasets", icon: "layers", href: "/datasets",
        hint: "sealed dataset delivery", separatorBefore: true },
      { key: "sources", label: "Storage sources", icon: "link", href: "/integrations",
        hint: "registered S3, GCS and Azure buckets" },
      { key: "webhooks", label: "Webhooks", icon: "activity", href: "/integrations?tab=webhooks",
        hint: "outbound event subscriptions" },
    ],
  },
  {
    key: "edit",
    label: "Edit",
    items: [
      { key: "tags", label: "Tags", icon: "flag", href: "/explore",
        hint: "bulk tag from the explorer" },
      { key: "views", label: "Saved views", icon: "eye", href: "/explore?panel=views",
        hint: "reusable cohort filters" },
      { key: "palette", label: "Command palette", icon: "search", shortcut: "Cmd K",
        event: "lbx:palette", separatorBefore: true },
    ],
  },
  {
    key: "view",
    label: "View",
    items: [
      { key: "platforms", label: "Platform launcher", icon: "grid", href: "/platforms",
        hint: "the seven planes of the data engine" },
      { key: "triage", label: "Triage queue", icon: "target", href: "/",
        hint: "objects ranked by value" },
      { key: "explore", label: "Explore", icon: "scan", href: "/explore", separatorBefore: true,
        hint: "embeddings map, facets, tags" },
      { key: "search", label: "Search", icon: "search", href: "/search",
        hint: "visual and semantic similarity" },
      { key: "scenarios", label: "Scenarios", icon: "route", href: "/scenarios",
        hint: "behavioral scenario mining" },
      { key: "discovery", label: "Discovery", icon: "wand", href: "/discovery",
        hint: "rare-scenario novelty queue" },
      { key: "curation", label: "Curation", icon: "layers", href: "/curation",
        hint: "frame-level active learning" },
      { key: "analytics", label: "Analytics", icon: "activity", href: "/analytics",
        separatorBefore: true, hint: "corpus health and loop signal" },
      { key: "inspector", label: "Session inspector", icon: "grid", href: "/inspect",
        hint: "MCAP timeline and panels" },
    ],
  },
  {
    key: "label",
    label: "Label",
    items: [
      { key: "projects", label: "Projects", icon: "clipboard", href: "/projects",
        hint: "assign jobs, stages, scorecards" },
      { key: "myjobs", label: "My jobs", icon: "check", href: "/projects?mine=1",
        hint: "what is assigned to you" },
      { key: "review", label: "Review queue", icon: "review", href: "/review/queue",
        separatorBefore: true, hint: "active learning and error candidates" },
      { key: "rapid", label: "Rapid review", icon: "check", href: "/review/rapid",
        hint: "one crop, one keystroke" },
      { key: "grid", label: "Crop grid", icon: "layers", href: "/review/grid",
        hint: "many crops, one keystroke each" },
      { key: "annotations", label: "Annotations", icon: "list", href: "/annotations",
        hint: "browse and resume" },
      { key: "agent", label: "Agent console", icon: "activity", href: "/agent",
        hint: "autonomous QA and fix queue" },
      { key: "multicam", label: "Multi-camera", icon: "layers", href: "/annotations?rig=1",
        separatorBefore: true, hint: "synchronized rig annotation" },
      { key: "jobsrun", label: "Runs", icon: "activity", href: "/jobs",
        hint: "import, training and autolabel runs" },
    ],
  },
  {
    key: "quality",
    label: "Quality",
    items: [
      { key: "quality", label: "Quality sheet", icon: "check", href: "/quality",
        hint: "gold set and gate metrics" },
      { key: "reasoner", label: "Reasoning layer", icon: "target", href: "/reasoner",
        hint: "what ran before each label, and whether it was right" },
      { key: "calibration", label: "Calibration", icon: "ruler", href: "/calibration",
        hint: "camera validation" },
      { key: "training", label: "Training", icon: "activity", href: "/training",
        separatorBefore: true, hint: "training jobs and model registry" },
      { key: "govern", label: "Govern", icon: "flag", href: "/govern",
        hint: "loop control, champion gate, kill switch" },
      { key: "campaigns", label: "Campaigns", icon: "target", href: "/campaigns",
        hint: "the improvement loop, run by the system" },
      { key: "lineage", label: "Lineage", icon: "route", href: "/lineage",
        hint: "what a model is made of, and what a session ended up in" },
      { key: "field", label: "Field performance", icon: "activity", href: "/forgyx/field",
        hint: "what deployed devices report against the bench" },
      { key: "collaborate", label: "Collaborate", icon: "link", href: "/collaborate",
        hint: "branches and merge requests" },
    ],
  },
  {
    key: "spatial",
    label: "Spatial",
    items: [
      { key: "lidar", label: "LiDAR", icon: "lidar3d", href: "/lidar", hint: "point cloud explorer" },
      // These two workspaces existed but had no inbound link from anywhere in the app, so the 3D cuboid
      // editor and the 3D-to-2D linked view were reachable only by typing a URL.
      { key: "lidar-annotate", label: "3D cuboids", icon: "lidar3d", href: "/lidar/annotate",
        hint: "annotate cuboids on a point cloud" },
      { key: "lidar-linked", label: "3D to camera", icon: "lidar3d", href: "/lidar/linked",
        hint: "cuboid projected onto its camera view" },
      { key: "map", label: "HD map", icon: "route", href: "/map", hint: "fused map and provenance" },
      { key: "inertial", label: "Inertial", icon: "activity", href: "/inertial",
        hint: "ego state, events, maneuvers" },
    ],
  },
  {
    // The second domain. LabeloxSec is a pack, not a platform: it changes what the same spine is authorised to
    // do rather than adding another stage to the loop, which is why it lives in its own menu instead of the
    // platform switcher.
    key: "security",
    label: "Security",
    items: [
      { key: "labeloxsec", label: "LabeloxSec console", icon: "flag", href: "/labeloxsec",
        hint: "plate reads, watchlist, static-camera sessions" },
      { key: "sec-hits", label: "Watchlist hits", icon: "activity", href: "/labeloxsec?hits=1",
        hint: "reads that matched a watched mark" },
      { key: "sec-incidents", label: "Incidents", icon: "flag", href: "/labeloxsec/incidents",
        hint: "zones, crossings, and cross-camera sightings" },
    ],
  },
  {
    // The account and its trail. These were unreachable because they did not exist: the only credential was
    // an admin-minted token, so there was nothing to manage and nowhere to see what you had done.
    key: "account",
    label: "Account",
    items: [
      { key: "profile", label: "Your account", icon: "user", href: "/profile",
        hint: "password, two-factor, sessions" },
      { key: "activity", label: "Activity", icon: "activity", href: "/activity",
        hint: "what you and the team have done" },
      { key: "pii-access", label: "PII access log", icon: "shield", href: "/govern/pii-access",
        hint: "who viewed personal data (admin)", separatorBefore: true },
      { key: "language", label: "Language", icon: "info", event: "lbx:language",
        hint: "English, हिन्दी, ಕನ್ನಡ, தமிழ்", separatorBefore: true },
      { key: "tour", label: "Show the tour again", icon: "info", event: "lbx:tour",
        hint: "the four things worth knowing on day one" },
    ],
  },
  {
    key: "window",
    label: "Window",
    items: [
      { key: "palette2", label: "Command palette", icon: "search", shortcut: "Cmd K", event: "lbx:palette" },
      { key: "shortcuts", label: "Keyboard shortcuts", icon: "keyboard", shortcut: "?",
        event: "lbx:shortcuts" },
      { key: "platformsw", label: "Platforms", icon: "grid", href: "/platforms", separatorBefore: true },
    ],
  },
  {
    key: "help",
    label: "Help",
    items: [
      { key: "help-shortcuts", label: "Keyboard shortcuts", icon: "keyboard", shortcut: "?",
        event: "lbx:shortcuts" },
      { key: "help-platforms", label: "The seven platforms", icon: "grid", href: "/platforms",
        separatorBefore: true },
    ],
  },
];

/** A menu entry that definitely navigates: `href` is required, so consumers need no narrowing. */
export type Destination = MenuItem & { href: string };

/** Flat list of every navigable destination, for the command palette and for tests. */
export const MENU_DESTINATIONS: Destination[] = MENUS.flatMap((m) =>
  m.items
    .flatMap((i) => (i.items ? i.items : [i]))
    .filter((i): i is Destination => typeof i.href === "string"),
);
