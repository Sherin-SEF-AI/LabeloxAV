// Color is earned: each state maps to exactly one signal color. Beyond the object/annotation states, the
// cross-cutting domain statuses used across the dashboards (jobs, training, calibration, quality,
// governance, collaboration) map onto the same four signals, so a status chip reads identically anywhere.
const STATE_COLOR: Record<string, string> = {
  // object / annotation states
  auto_accept: "text-pass border-pass/40 bg-pass/10",
  accepted: "text-pass border-pass/40 bg-pass/10",
  review: "text-warn border-warn/40 bg-warn/10",
  annotate: "text-accent border-accent/40 bg-accent/10",
  rejected: "text-block border-block/40 bg-block/10",
  // terminal-good
  done: "text-pass border-pass/40 bg-pass/10",
  succeeded: "text-pass border-pass/40 bg-pass/10",
  success: "text-pass border-pass/40 bg-pass/10",
  complete: "text-pass border-pass/40 bg-pass/10",
  pass: "text-pass border-pass/40 bg-pass/10",
  measured: "text-pass border-pass/40 bg-pass/10",
  promote: "text-pass border-pass/40 bg-pass/10",
  promoted: "text-pass border-pass/40 bg-pass/10",
  approved: "text-pass border-pass/40 bg-pass/10",
  merged: "text-pass border-pass/40 bg-pass/10",
  // in-progress / informational
  running: "text-info border-info/40 bg-info/10",
  active: "text-info border-info/40 bg-info/10",
  open: "text-info border-info/40 bg-info/10",
  // waiting / caution
  queued: "text-warn border-warn/40 bg-warn/10",
  pending: "text-warn border-warn/40 bg-warn/10",
  warn: "text-warn border-warn/40 bg-warn/10",
  paused: "text-warn border-warn/40 bg-warn/10",
  pause: "text-warn border-warn/40 bg-warn/10",
  unmeasured: "text-ink-3 border-line bg-line/30",
  // terminal-bad
  failed: "text-block border-block/40 bg-block/10",
  fail: "text-block border-block/40 bg-block/10",
  error: "text-block border-block/40 bg-block/10",
  reverted: "text-block border-block/40 bg-block/10",
  reject: "text-block border-block/40 bg-block/10",
  // job kinds (the unified jobs stream colors its kind column with this chip)
  import: "text-info border-info/40 bg-info/10",
  training: "text-accent border-accent/40 bg-accent/10",
  autolabel: "text-pass border-pass/40 bg-pass/10",
  // model registry roles
  champion: "text-pass border-pass/40 bg-pass/10",
  challenger: "text-info border-info/40 bg-info/10",
};

// Which sources mean "a machine put this here". `source` collapses to "human" the moment a person touches
// the row, so anything not in this set was last written by a person.
const MACHINE_SOURCES = new Set([
  "fused", "auto_accept", "imported", "recall", "interpolated", "interp", "propagated", "relabel",
]);

export function StateBadge({ state, source, conf }: { state: string; source?: string; conf?: number }) {
  const cls = STATE_COLOR[state] ?? "text-ink-2 border-line bg-line/30";
  // Read as words: "auto_accept" is a database value, "Auto accept" is what a person reads.
  const label = state ? state.charAt(0).toUpperCase() + state.slice(1).replace(/[_-]/g, " ") : state;

  // Who accepted this, drawn rather than described.
  //
  // `accepted` and `auto_accept` both rendered solid green with no actor, so an object sitting at 0.34
  // confidence looked exactly like one a reviewer had ruled on. That badge is what an annotator reads to
  // decide whether to trust a label, and it was answering a different question than the one being asked.
  //
  // Dashed is machine, solid is human, matching the stroke grammar used on canvas. A machine acceptance is
  // still green, because the gate did accept it; it just no longer claims a person did.
  const machine = source != null && MACHINE_SOURCES.has(source);
  const border = machine ? "border-dashed" : "border-solid";

  // The pathology worth naming: accepted, but on a score the gate would not auto-accept today. Silent
  // before, and the exact case where "accepted" reads as more settled than it is.
  const weak = conf != null && conf < 0.6 && (state === "accepted" || state === "auto_accept");

  const title = machine
    ? `accepted by ${source}, no human has ruled on it`
    : source
      ? `last written by ${source}`
      : undefined;

  return (
    // Never wraps. "queued-cloud" broke onto two lines in the jobs and training tables and pushed those
    // rows taller than their neighbours, which reads as a layout fault rather than a long word.
    <span title={title}
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded text-[11px] font-medium leading-4 border ${border} px-1.5 py-px ${cls}`}>
      {label}
      {machine && <span className="opacity-70">auto</span>}
      {weak && <span className="text-warn" title="below the auto-accept floor">low</span>}
    </span>
  );
}

export function ConfBar({ conf }: { conf: number }) {
  const color = conf >= 0.95 ? "bg-pass" : conf >= 0.6 ? "bg-warn" : "bg-accent";
  return (
    <span className="inline-flex items-center gap-2 text-[12px] tabular-nums">
      <span className="w-14 h-1.5 rounded-full overflow-hidden bg-line relative">
        <span className={`absolute left-0 top-0 h-full ${color}`} style={{ width: `${conf * 100}%` }} />
      </span>
      {conf.toFixed(2)}
    </span>
  );
}
