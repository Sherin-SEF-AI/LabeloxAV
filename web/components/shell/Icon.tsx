import { createElement } from "react";

// The interface icon set, transcribed from the approved Annotation Interface design (lucide-style, 24x24,
// 1.7 stroke, currentColor). One small component so the mode rail, tool strip, layers, top bar, and bottom
// bar all read from the same glyphs. color comes from the surrounding text color, size is in px.

type Prim = [string, Record<string, string | number>];

const ICONS: Record<string, Prim[]> = {
  cursor: [["path", { d: "M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z" }]],
  box: [["rect", { x: 3, y: 3, width: 18, height: 18, rx: 2 }]],
  polygon: [["path", { d: "M12 2l9 6-3.4 11H6.4L3 8z" }]],
  polyline: [["path", { d: "M4 19 9 9l4 4 7-9" }], ["circle", { cx: 4, cy: 19, r: 1.4 }], ["circle", { cx: 20, cy: 4, r: 1.4 }]],
  wand: [["path", { d: "m3 21 11-11" }], ["path", { d: "M15 3.5l1 2.2 2.2 1-2.2 1-1 2.2-1-2.2-2.2-1 2.2-1z" }]],
  target: [["circle", { cx: 12, cy: 12, r: 7.5 }], ["circle", { cx: 12, cy: 12, r: 2.6 }], ["path", { d: "M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3" }]],
  scan: [["path", { d: "M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2" }], ["rect", { x: 8, y: 8, width: 8, height: 8, rx: 1 }]],
  brush: [["path", { d: "M9.06 11.9 3 18v3h3l6.06-6.06" }], ["path", { d: "m14 7 3 3" }], ["path", { d: "M18.5 3.5a2.12 2.12 0 0 1 3 3L14.5 13.5l-4-4z" }]],
  eraser: [["path", { d: "m7 21 13-13-5-5L2 16l5 5z" }], ["path", { d: "M22 21H7" }], ["path", { d: "m5 11 5 5" }]],
  grid: [["rect", { x: 3, y: 3, width: 18, height: 18, rx: 1 }], ["path", { d: "M9 3v18M15 3v18M3 9h18M3 15h18" }]],
  ruler: [["path", { d: "M3 8 8 3l13 13-5 5z" }], ["path", { d: "m8 8 2 2M11 5l2 2M14 8l2 2M5 11l2 2" }]],
  route: [["circle", { cx: 6, cy: 19, r: 2 }], ["circle", { cx: 18, cy: 5, r: 2 }], ["path", { d: "M6 17V9a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4" }]],
  layers: [["path", { d: "m12 2 9 5-9 5-9-5 9-5z" }], ["path", { d: "m3 12 9 5 9-5M3 17l9 5 9-5" }]],
  cuboid: [["path", { d: "M12 2 3 7v10l9 5 9-5V7z" }], ["path", { d: "M3 7l9 5 9-5M12 12v10" }]],
  person: [["circle", { cx: 12, cy: 4.5, r: 2 }], ["path", { d: "M12 6.5v7M8 10.5h8M9 21l3-7 3 7" }]],
  activity: [["path", { d: "M22 12h-4l-3 9L9 3l-3 9H2" }]],
  keypoint: [["circle", { cx: 12, cy: 12, r: 2.6 }], ["path", { d: "M12 2v4M12 18v4M2 12h4M18 12h4" }]],
  link: [["path", { d: "M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1" }], ["path", { d: "M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 0 0 7 7l1-1" }]],
  list: [["path", { d: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" }]],
  check: [["path", { d: "M20 6 9 17l-5-5" }]],
  x: [["path", { d: "M18 6 6 18M6 6l12 12" }]],
  close: [["path", { d: "M18 6 6 18M6 6l12 12" }]],
  flag: [["path", { d: "M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" }], ["path", { d: "M4 22V4" }]],
  comment: [["path", { d: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" }]],
  clipboard: [["rect", { x: 8, y: 2, width: 8, height: 4, rx: 1 }], ["path", { d: "M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" }], ["path", { d: "m9 14 2 2 4-4" }]],
  undo: [["path", { d: "M9 14 4 9l5-5" }], ["path", { d: "M4 9h11a5 5 0 0 1 0 10h-4" }]],
  redo: [["path", { d: "m15 14 5-5-5-5" }], ["path", { d: "M20 9H9a5 5 0 0 0 0 10h4" }]],
  save: [["path", { d: "M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" }], ["path", { d: "M17 21v-8H7v8M7 3v5h8" }]],
  confirm: [["path", { d: "M18 6 7 17l-5-5" }], ["path", { d: "m22 10-7.5 7.5L13 16" }]],
  chevL: [["path", { d: "m15 18-6-6 6-6" }]],
  chevR: [["path", { d: "m9 18 6-6-6-6" }]],
  chevD: [["path", { d: "m6 9 6 6 6-6" }]],
  prev: [["path", { d: "m15 18-6-6 6-6" }]],
  next: [["path", { d: "m9 18 6-6-6-6" }]],
  search: [["circle", { cx: 11, cy: 11, r: 7.5 }], ["path", { d: "m21 21-4.3-4.3" }]],
  eye: [["path", { d: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" }], ["circle", { cx: 12, cy: 12, r: 3 }]],
  eyeOff: [["path", { d: "M9.9 4.2A9 9 0 0 1 12 4c6.5 0 10 8 10 8a13 13 0 0 1-2.2 2.9" }], ["path", { d: "M6.6 6.6A12.8 12.8 0 0 0 2 12s3.5 7 10 7a8.9 8.9 0 0 0 4-1" }], ["path", { d: "m2 2 20 20" }]],
  plus: [["circle", { cx: 12, cy: 12, r: 9 }], ["path", { d: "M12 8v8M8 12h8" }]],
  trash: [["path", { d: "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" }]],
  info: [["circle", { cx: 12, cy: 12, r: 9 }], ["path", { d: "M12 16v-4M12 8h.01" }]],
  keyboard: [["rect", { x: 2, y: 6, width: 20, height: 12, rx: 2 }], ["path", { d: "M6 10h.01M10 10h.01M14 10h.01M18 10h.01M6 14h12" }]],
  fit: [["path", { d: "M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3" }]],
  zoomIn: [["circle", { cx: 11, cy: 11, r: 7.5 }], ["path", { d: "m21 21-4.3-4.3M11 8v6M8 11h6" }]],
  zoomOut: [["circle", { cx: 11, cy: 11, r: 7.5 }], ["path", { d: "m21 21-4.3-4.3M8 11h6" }]],
  dot: [["circle", { cx: 12, cy: 12, r: 3 }]],
};

export type IconName = keyof typeof ICONS;

export default function Icon({ name, size = 16 }: { name: string; size?: number }) {
  const prims = ICONS[name] ?? ICONS.dot;
  return createElement(
    "svg",
    { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round" },
    prims.map((p, i) => createElement(p[0], { key: i, ...p[1] })),
  );
}

// Tool key (the editor's dispatch keys) to icon name, and mode key to icon name. Centralized so a new tool
// or mode maps to a glyph in one place, with no change to the chrome components.
export const TOOL_ICON: Record<string, string> = {
  select: "cursor",
  box: "box", polygon: "polygon", polyline: "polyline",
  "sam-point": "target", "sam-box": "scan", "magic-wand": "wand",
  brush: "brush", eraser: "eraser", superpixel: "grid",
  keypoint: "keypoint", adverse: "flag", cuboid: "cuboid",
  "cuboid-add": "cuboid", "cuboid-lift": "wand", "lane-add": "route",
  measure: "ruler",
};

export const MODE_ICON: Record<string, string> = {
  objects: "box",
  lanes: "route",
  pose: "person",
  lidar3d: "cuboid",
  review: "clipboard",
};
