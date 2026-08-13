// The console's sections, and finding one by typing.
//
// As a page, the console was somewhere you navigated to, which means leaving whatever you were doing. The
// question it answers ("is anything happening, and is it my machine or my job") is one you ask in the middle
// of something else, so the answer has to arrive over the work rather than instead of it.
//
// A list of sections plus a search box is the shape that survives growth: every panel that would otherwise
// compete for room on one page becomes a row here, and the search stays the way in once there are more rows
// than fit the eye.

export type ConsoleSectionId =
  | "overview" | "jobs" | "background" | "canvas" | "gpu" | "machine" | "process";

export type ConsoleSection = {
  id: ConsoleSectionId;
  label: string;
  /** The heading the sidebar groups under, as in the settings dialog this follows. */
  group: string;
  icon: string;
  /** Words somebody would plausibly type looking for this, beyond its own label. */
  keywords: string[];
};

export const CONSOLE_SECTIONS: ConsoleSection[] = [
  { id: "overview", label: "Overview", group: "Activity", icon: "activity",
    keywords: ["status", "summary", "everything", "what is happening", "live"] },
  { id: "jobs", label: "Running now", group: "Activity", icon: "play",
    keywords: ["jobs", "queue", "queued", "progress", "autolabel", "training", "export", "import"] },
  { id: "background", label: "Background", group: "Activity", icon: "robot",
    keywords: ["agent", "runs", "sweep", "relabel", "batch", "corpus"] },
  { id: "canvas", label: "Canvas", group: "Activity", icon: "shapes",
    keywords: ["editor", "sam", "segment", "propagate", "mask", "save", "frame"] },
  { id: "gpu", label: "GPU", group: "Machine", icon: "chip",
    keywords: ["cuda", "vram", "memory", "utilisation", "utilization", "temperature", "nvidia", "busy"] },
  { id: "machine", label: "Host", group: "Machine", icon: "server",
    keywords: ["cpu", "ram", "memory", "disk", "storage", "load", "space", "full"] },
  { id: "process", label: "API process", group: "Machine", icon: "terminal",
    keywords: ["uptime", "pid", "threads", "resident", "rss", "backend", "server"] },
];

/**
 * Sections matching a typed query, in their declared order.
 *
 * Matches the label, the group and the keywords, because the words people reach for are rarely the words on
 * the row: "disk full" and "vram" are how the two questions this console exists for are actually phrased,
 * and neither appears in a section's name.
 */
export function filterSections(query: string,
                               sections: readonly ConsoleSection[] = CONSOLE_SECTIONS): ConsoleSection[] {
  const q = query.trim().toLowerCase();
  if (!q) return [...sections];
  const terms = q.split(/\s+/);
  return sections.filter((s) => {
    const hay = `${s.label} ${s.group} ${s.keywords.join(" ")}`.toLowerCase();
    // Every term has to land somewhere, so a second word narrows rather than widens.
    return terms.every((t) => hay.includes(t));
  });
}

/**
 * The section to show for a query, given what is currently selected.
 *
 * Keeping the selection when it still matches is what makes typing feel like filtering rather than
 * navigating: a search that jumped to the first result would move the reader off the panel they are reading
 * as soon as they typed a letter.
 */
export function resolveSelection(query: string, current: ConsoleSectionId,
                                 sections: readonly ConsoleSection[] = CONSOLE_SECTIONS,
                                 ): ConsoleSectionId | null {
  const matches = filterSections(query, sections);
  if (!matches.length) return null;
  return matches.some((s) => s.id === current) ? current : matches[0].id;
}

/** Sections in sidebar order, grouped, so the sidebar does not re-derive the grouping itself. */
export function groupSections(sections: readonly ConsoleSection[]): { group: string; items: ConsoleSection[] }[] {
  const out: { group: string; items: ConsoleSection[] }[] = [];
  for (const s of sections) {
    const last = out[out.length - 1];
    if (last && last.group === s.group) last.items.push(s);
    else out.push({ group: s.group, items: [s] });
  }
  return out;
}
