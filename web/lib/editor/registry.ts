// The single source of truth for the redesigned interface. The UI grows by organization, not addition:
// a new destination is one entry in APP_GROUPS, a new editor tool is one entry in a group's tools, and a
// new mode is one entry in MODES. Nothing here widens a toolbar or a nav row; the chrome reads this data
// and renders groups (not flat lists), so the layout absorbs new features with zero structural change.

export type NavItem = { href: string; label: string; hint?: string };
export type NavGroup = { key: string; label: string; items: NavItem[] };

// Grouped application navigation. Replaces the flat TopNav scroll; drives the AppSwitcher and the command
// palette. Add a destination by adding one item to a group (or a new group), never a peer in a flat row.
export const APP_GROUPS: NavGroup[] = [
  {
    key: "work",
    label: "Work",
    items: [
      { href: "/", label: "Triage", hint: "object queue ranked by value" },
      { href: "/review/queue", label: "Review queue", hint: "active learning + error candidates" },
      { href: "/annotations", label: "Annotations", hint: "browse and resume sessions" },
      { href: "/jobs", label: "Jobs", hint: "import, training, autolabel runs" },
    ],
  },
  {
    key: "discover",
    label: "Discover",
    items: [
      { href: "/search", label: "Search", hint: "visual and semantic similarity" },
      { href: "/scenarios", label: "Scenarios", hint: "behavioral scenario mining" },
      { href: "/discovery", label: "Discovery", hint: "rare-scenario novelty queue" },
      { href: "/curation", label: "Curation", hint: "frame-level active learning" },
    ],
  },
  {
    key: "data",
    label: "Data",
    items: [
      { href: "/annotate/new", label: "New", hint: "start an annotation from upload" },
      { href: "/import", label: "Import", hint: "ingest an external dataset" },
      { href: "/datasets", label: "Datasets", hint: "sealed dataset delivery" },
    ],
  },
  {
    key: "quality",
    label: "Quality and models",
    items: [
      { href: "/quality", label: "Quality", hint: "gold set, gate B metrics" },
      { href: "/analytics", label: "Analytics", hint: "corpus health and loop signal" },
      { href: "/calibration", label: "Calibration", hint: "camera validation" },
      { href: "/training", label: "Training", hint: "training jobs and registry" },
      { href: "/govern", label: "Govern", hint: "loop control and championship" },
      { href: "/collaborate", label: "Collaborate", hint: "branches and merge requests" },
    ],
  },
  {
    key: "spatial",
    label: "Spatial",
    items: [
      { href: "/lidar", label: "LiDAR", hint: "point cloud explorer" },
      { href: "/map", label: "HD map", hint: "fused map and provenance" },
    ],
  },
];

export const ALL_DESTINATIONS: NavItem[] = APP_GROUPS.flatMap((g) => g.items);

// ---- Editor modes (used by the moded EditorShell from Phase 2 on) -------------------------------------
// A mode is a mutually exclusive toolset on the canvas. Its groups collapse to one tool-strip button each
// (active tool shown, variants in a flyout, cycled by repeated hotkey), so the strip is one row forever.

export type ToolDef = { key: string; label: string; hotkey: string; cursor?: string };
export type ToolGroup = { key: string; label: string; tools: ToolDef[] };
export type CanvasKind = "konva" | "three" | "table";
export type EditorMode = {
  key: string;
  label: string;
  rail: string;       // short mono glyph/label for the fixed-width left rail
  hotkey: string;     // mode switch key
  canvas: CanvasKind;
  groups: ToolGroup[];
};

export const MODES: EditorMode[] = [
  {
    key: "objects", label: "Objects", rail: "OBJ", hotkey: "1", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "draw", label: "Draw", tools: [
        { key: "box", label: "box", hotkey: "B" },
        { key: "polygon", label: "polygon", hotkey: "G" },
        { key: "polyline", label: "polyline", hotkey: "L" },
      ] },
      { key: "ai", label: "AI assist", tools: [
        { key: "sam-point", label: "sam point", hotkey: "S" },
        { key: "sam-box", label: "sam box", hotkey: "M" },
        { key: "magic-wand", label: "wand", hotkey: "W" },
      ] },
      { key: "mask", label: "Mask edit", tools: [
        { key: "brush", label: "brush", hotkey: "P" },
        { key: "eraser", label: "eraser", hotkey: "E" },
        { key: "superpixel", label: "cells", hotkey: "U" },
      ] },
      { key: "region", label: "Region", tools: [{ key: "adverse", label: "adverse", hotkey: "D" }] },
      { key: "measure", label: "Measure", tools: [{ key: "measure", label: "measure", hotkey: "R" }] },
    ],
  },
  {
    key: "lanes", label: "Lanes and drivable", rail: "LANE", hotkey: "2", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "lane", label: "Lane", tools: [{ key: "lane-add", label: "add lane", hotkey: "B" }] },
    ],
  },
  {
    key: "pose", label: "Pose and behavior", rail: "POSE", hotkey: "3", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "pose", label: "Pose", tools: [{ key: "keypoint", label: "keypoint", hotkey: "K" }] },
    ],
  },
  {
    key: "lidar3d", label: "3D and LiDAR", rail: "3D", hotkey: "4", canvas: "three",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "cuboid", label: "Cuboid", tools: [
        { key: "cuboid-add", label: "add box", hotkey: "B" },
        { key: "cuboid-lift", label: "lift 2D to 3D", hotkey: "C" },
      ] },
      { key: "measure", label: "Measure", tools: [{ key: "measure", label: "measure", hotkey: "R" }] },
    ],
  },
  {
    key: "review", label: "Review", rail: "QA", hotkey: "5", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
    ],
  },
];

export function modeByKey(key: string): EditorMode | undefined {
  return MODES.find((m) => m.key === key);
}
