// The editor's mode and tool registry.
//
// This file also held APP_GROUPS/ALL_DESTINATIONS: a complete second navigation registry, 33 destinations,
// with no consumers anywhere. It had drifted from lib/menus.ts - the one the menu bar and the command
// palette both read - and the drift had a cost: /events and /events/search were listed only here, so they
// were unreachable from the actual navigation. They are in menus.ts now, and the dead half is gone rather
// than left as a second source of truth for anyone to update by mistake.

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
      { key: "select", label: "Select", tools: [{ key: "select", label: "Select", hotkey: "V" }] },
      { key: "draw", label: "Draw", tools: [
        { key: "box", label: "Box", hotkey: "B" },
        { key: "polygon", label: "Polygon", hotkey: "G" },
        { key: "polyline", label: "Polyline", hotkey: "L" },
        // The whole extent of a partly hidden object. Draws into bbox_amodal, never into bbox: the
        // visible box is what every existing consumer means by "the box".
        { key: "amodal", label: "Whole extent", hotkey: "K" },
      ] },
      { key: "ai", label: "AI assist", tools: [
        { key: "sam-point", label: "SAM point", hotkey: "S" },
        { key: "sam-box", label: "SAM box", hotkey: "M" },
        { key: "magic-wand", label: "Wand", hotkey: "W" },
      ] },
      { key: "mask", label: "Mask edit", tools: [
        { key: "brush", label: "Brush", hotkey: "P" },
        { key: "eraser", label: "Eraser", hotkey: "E" },
        { key: "superpixel", label: "Cells", hotkey: "U" },
      ] },
      { key: "region", label: "Region", tools: [{ key: "adverse", label: "Region", hotkey: "D" }] },
      { key: "cuboid", label: "3D box", tools: [{ key: "cuboid", label: "Cuboid", hotkey: "C" }] },
      { key: "measure", label: "Measure", tools: [{ key: "measure", label: "Measure", hotkey: "R" }] },
    ],
  },
  {
    // Lane drawing is driven by the lane panel's own controls rather than by the canvas tool, so the
    // strip carries Select alone. It used to list five lane types and three surfaces here; none of them
    // reached the canvas, so every one of those buttons was inert and every one of those hotkeys did
    // nothing. A tool belongs in this registry when the canvas dispatches it and not before.
    key: "lanes", label: "Lanes and drivable", rail: "LANE", hotkey: "2", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "Select", hotkey: "V" }] },
    ],
  },
  {
    key: "semantic", label: "Semantic", rail: "SEM", hotkey: "3", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "Select", hotkey: "V" }] },
      // A region drawn for a class and a region drawn for an instance are the same gesture, so semantic
      // reuses the object canvas's polygon and eraser rather than teaching a second pair.
      { key: "semantic", label: "Semantic", tools: [
        { key: "sem-region", label: "Region", hotkey: "G" },
        { key: "sem-erase", label: "Erase", hotkey: "E" },
      ] },
    ],
  },
  {
    key: "events", label: "Events", rail: "EVT", hotkey: "4", canvas: "table",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "Select", hotkey: "V" }] },
    ],
  },
  {
    key: "pose", label: "Pose and behavior", rail: "POSE", hotkey: "5", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "Select", hotkey: "V" }] },
      { key: "pose", label: "Pose", tools: [{ key: "keypoint", label: "Keypoint", hotkey: "K" }] },
      { key: "measure", label: "Measure", tools: [{ key: "measure", label: "Measure", hotkey: "R" }] },
    ],
  },
  {
    // The point-cloud viewer is driven by its own panel controls, not by the canvas tool, so the same
    // rule applies here as to lanes: Select alone until a tool actually reaches the canvas.
    key: "lidar3d", label: "3D and LiDAR", rail: "3D", hotkey: "6", canvas: "three",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "Select", hotkey: "V" }] },
    ],
  },
  {
    key: "review", label: "Review", rail: "QA", hotkey: "7", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "Select", hotkey: "V" }] },
    ],
  },
];

// Every lookup below takes the mode list as a last argument and defaults it to MODES. The default is the
// registry as shipped; the argument is the same registry with an annotator's rebound keys applied
// (lib/editor/keymap.ts). One set of functions answering for both is what keeps "what does this key do"
// and "do two things want this key" from being computed two different ways.

export function modeByKey(key: string, modes: EditorMode[] = MODES): EditorMode | undefined {
  return modes.find((m) => m.key === key);
}

/** Every tool key a mode's strip offers, in strip order. */
export function toolsForMode(modeKey: string, modes: EditorMode[] = MODES): string[] {
  return (modeByKey(modeKey, modes)?.groups ?? []).flatMap((g) => g.tools.map((t) => t.key));
}

/** The strip's groups for a mode, falling back to Objects so an unknown mode still renders something. */
export function groupsForMode(modeKey: string, modes: EditorMode[] = MODES): ToolGroup[] {
  return modeByKey(modeKey, modes)?.groups ?? modeByKey("objects", modes)?.groups ?? [];
}

/**
 * The tool a single keystroke selects in a mode, or null when that key means nothing here.
 *
 * Resolved per mode, which is what makes the same letter able to mean two things without either one
 * being unreachable. The editor previously dispatched tools from a hardcoded if/else chain that knew
 * nothing about modes, so `k` was written twice in it: the first branch won for every mode and the
 * second, which selected the whole-extent tool, could never run. That tool had a button and a documented
 * shortcut and no way to reach it.
 */
export function toolForHotkey(modeKey: string, key: string, modes: EditorMode[] = MODES): string | null {
  const want = key.toLowerCase();
  for (const g of groupsForMode(modeKey, modes)) {
    for (const t of g.tools) {
      if (t.hotkey.toLowerCase() === want) return t.key;
    }
  }
  return null;
}

/** Modes whose hotkeys collide inside one mode. Empty is the invariant; the test asserts it. */
export function hotkeyCollisions(modes: EditorMode[] = MODES): { mode: string; hotkey: string; tools: string[] }[] {
  const out: { mode: string; hotkey: string; tools: string[] }[] = [];
  for (const m of modes) {
    const byKey = new Map<string, string[]>();
    for (const g of m.groups) {
      for (const t of g.tools) {
        const k = t.hotkey.toLowerCase();
        byKey.set(k, [...(byKey.get(k) ?? []), t.key]);
      }
    }
    for (const [hotkey, tools] of byKey) {
      if (tools.length > 1) out.push({ mode: m.key, hotkey, tools });
    }
  }
  return out;
}
