// What the top-bar activity chip should say, as a value rather than as JSX.
//
// The chip used to render nothing when nothing was running. That reads as correct and is not: it was asked
// for three times by somebody who could not find it, because a control that only exists while the system
// happens to be busy cannot be looked at to answer "is anything happening". The answer "no" is an answer,
// and it needs somewhere to live.
//
// So there is always a chip, in one of three states, and the difference between them is the whole point:
//
//   working   something is actually progressing. Dot pulses, count and bar shown.
//   queued    the system is holding work it is not doing. This deployment carries 67 autolabel jobs parked
//             for a cloud A100 since late June and one pending training job. That is the answer to "why is
//             nothing happening", and it was previously buried in a tooltip nobody had a reason to hover.
//   idle      nothing running, nothing queued.
//
// Only `working` uses the live dot. A chip reading "68 running" forever teaches people to ignore the chip,
// which costs more than never having built it.

export type IndicatorKind = "working" | "queued" | "idle";

export type IndicatorView = {
  kind: IndicatorKind;
  /** The word shown next to the dot. */
  verb: string;
  /** The number shown after the verb, or null when a count would say nothing. */
  count: number | null;
  /** Bar fill 0..99, or null when there is no work to fill it with. */
  pct: number | null;
  /** Hover text: the full story, including the part the chip has no room for. */
  tip: string;
  /** Where a click goes. */
  href: string;
};

export type IndicatorInput = {
  /** Server jobs actually progressing, with their kind and 0..1 progress. */
  running: { kind: string; progress: number }[];
  /** Server jobs held but not progressing. */
  waiting: number;
  /** Local upload progress, one entry per file still uploading from this browser. */
  uploading: number[];
  /** Whether this tab's own queue is running, which is what makes the upload page the right destination. */
  localRunning: boolean;
  /** This tab's queue phase, when it has one. */
  phase?: string | null;
  /** Finished and total for this tab's queue, for the tooltip. */
  localDone?: number;
  localTotal?: number;
};

export function indicatorView(x: IndicatorInput): IndicatorView {
  const serverRunning = x.running.length;
  const total = serverRunning + x.uploading.length;

  // The local queue has a page of its own; a server job only has the jobs list. Sending somebody to an empty
  // upload page to watch an autolabel run started in another tab would be a dead end.
  const href = x.localRunning && x.uploading.length > 0 ? "/annotate/new" : "/jobs";

  if (total === 0) {
    if (x.waiting > 0) {
      return {
        kind: "queued", verb: "queued", count: x.waiting, pct: null,
        tip: `${x.waiting} job${x.waiting === 1 ? "" : "s"} queued and not progressing · click to see why`,
        href: "/jobs",
      };
    }
    return { kind: "idle", verb: "idle", count: null, pct: null,
             tip: "nothing running · click to open jobs", href: "/jobs" };
  }

  // Averaged across everything in flight. One number is all that fits, and it is honest as long as it never
  // claims completion while something is still going, hence the cap at 99.
  const fracs = [...x.running.map((r) => r.progress), ...x.uploading];
  const pct = Math.min(99, Math.round((fracs.reduce((a, b) => a + b, 0) / fracs.length) * 100));

  const verb = x.uploading.length > 0 && serverRunning === 0 ? "uploading"
    : x.phase === "autolabeling" ? "labelling"
    : serverRunning > 0 && x.running.every((r) => r.kind === "autolabel") ? "labelling"
    : "working";

  const tip = [
    `${total} running`,
    x.waiting > 0 ? `${x.waiting} queued and not progressing` : null,
    x.localRunning && x.localTotal ? `${x.localDone ?? 0}/${x.localTotal} in this tab` : null,
    "click to watch",
  ].filter(Boolean).join(" · ");

  return { kind: "working", verb, count: total, pct, tip, href };
}
