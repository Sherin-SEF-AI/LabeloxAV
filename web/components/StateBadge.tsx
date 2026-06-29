// Color is earned: each state maps to exactly one signal color. Beyond the object/annotation states, the
// cross-cutting domain statuses used across the dashboards (jobs, training, calibration, quality,
// governance, collaboration) map onto the same four signals, so a status chip reads identically anywhere.
const STATE_COLOR: Record<string, string> = {
  // object / annotation states
  auto_accept: "text-pass border-pass",
  accepted: "text-pass border-pass",
  review: "text-warn border-warn",
  annotate: "text-accent border-accent",
  rejected: "text-block border-block",
  // terminal-good
  done: "text-pass border-pass",
  succeeded: "text-pass border-pass",
  success: "text-pass border-pass",
  complete: "text-pass border-pass",
  pass: "text-pass border-pass",
  measured: "text-pass border-pass",
  promote: "text-pass border-pass",
  promoted: "text-pass border-pass",
  approved: "text-pass border-pass",
  merged: "text-pass border-pass",
  // in-progress / informational
  running: "text-info border-info",
  active: "text-info border-info",
  open: "text-info border-info",
  // waiting / caution
  queued: "text-warn border-warn",
  pending: "text-warn border-warn",
  warn: "text-warn border-warn",
  paused: "text-warn border-warn",
  pause: "text-warn border-warn",
  unmeasured: "text-ink-3 border-line",
  // terminal-bad
  failed: "text-block border-block",
  fail: "text-block border-block",
  error: "text-block border-block",
  reverted: "text-block border-block",
  reject: "text-block border-block",
};

export function StateBadge({ state }: { state: string }) {
  const cls = STATE_COLOR[state] ?? "text-ink-2 border-line";
  return (
    <span className={`font-mono text-[11px] uppercase tracking-wide border px-1.5 py-0.5 ${cls}`}>
      {state}
    </span>
  );
}

export function ConfBar({ conf }: { conf: number }) {
  const color = conf >= 0.95 ? "bg-pass" : conf >= 0.6 ? "bg-warn" : "bg-accent";
  return (
    <span className="inline-flex items-center gap-2 font-mono text-xs">
      <span className="w-14 h-1.5 bg-line relative">
        <span className={`absolute left-0 top-0 h-full ${color}`} style={{ width: `${conf * 100}%` }} />
      </span>
      {conf.toFixed(2)}
    </span>
  );
}
