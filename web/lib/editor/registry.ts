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
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "draw", label: "Draw", tools: [
        { key: "box", label: "box", hotkey: "B" },
        { key: "polygon", label: "polygon", hotkey: "G" },
        { key: "polyline", label: "polyline", hotkey: "L" },
        // The whole extent of a partly hidden object. Draws into bbox_amodal, never into bbox: the
        // visible box is what every existing consumer means by "the box".
        { key: "amodal", label: "whole extent", hotkey: "K" },
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
      // Lane type is a tool rather than a property set after the fact, because it decides whether a crossing
      // of this boundary is a manoeuvre or an offence, and picking it while drawing is when the annotator
      // is actually looking at the line.
      { key: "lane", label: "Lane", tools: [
        { key: "lane-solid", label: "solid", hotkey: "B" },
        { key: "lane-dashed", label: "dashed", hotkey: "N" },
        { key: "lane-double", label: "double", hotkey: "J" },
        { key: "lane-edge", label: "road edge", hotkey: "H" },
        { key: "lane-implicit", label: "implicit", hotkey: "Y" },
      ] },
      { key: "freespace", label: "Free space", tools: [
        { key: "drivable", label: "drivable", hotkey: "F" },
        { key: "non-drivable", label: "non drivable", hotkey: "X" },
        { key: "fallback", label: "fallback", hotkey: "Z" },
      ] },
      { key: "laneops", label: "Lane ops", tools: [
        { key: "lane-propose", label: "propose", hotkey: "O" },
        { key: "lane-propagate", label: "propagate", hotkey: "P" },
      ] },
    ],
  },
  {
    key: "semantic", label: "Semantic", rail: "SEM", hotkey: "3", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "paint", label: "Paint", tools: [
        { key: "sem-polygon", label: "region", hotkey: "G" },
        { key: "sem-brush", label: "brush", hotkey: "P" },
        { key: "sem-eraser", label: "eraser", hotkey: "E" },
        { key: "sem-fill", label: "fill", hotkey: "F" },
      ] },
      { key: "assist", label: "Assist", tools: [
        { key: "sem-superpixel", label: "cells", hotkey: "U" },
        { key: "sem-auto", label: "auto segment", hotkey: "A" },
      ] },
    ],
  },
  {
    key: "events", label: "Events", rail: "EVT", hotkey: "4", canvas: "table",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "mark", label: "Mark", tools: [
        { key: "event-point", label: "instant", hotkey: "I" },
        { key: "event-interval", label: "interval", hotkey: "T" },
      ] },
      { key: "derive", label: "Derive", tools: [
        { key: "event-derive", label: "derive", hotkey: "D" },
        { key: "event-link-lanes", label: "link lanes", hotkey: "K" },
      ] },
    ],
  },
  {
    key: "pose", label: "Pose and behavior", rail: "POSE", hotkey: "5", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
      { key: "pose", label: "Pose", tools: [{ key: "keypoint", label: "keypoint", hotkey: "K" }] },
    ],
  },
  {
    key: "lidar3d", label: "3D and LiDAR", rail: "3D", hotkey: "6", canvas: "three",
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
    key: "review", label: "Review", rail: "QA", hotkey: "7", canvas: "konva",
    groups: [
      { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
    ],
  },
];

export function modeByKey(key: string): EditorMode | undefined {
  return MODES.find((m) => m.key === key);
}
