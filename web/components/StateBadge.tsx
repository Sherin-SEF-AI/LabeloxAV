// Color is earned: each state maps to exactly one signal color.
const STATE_COLOR: Record<string, string> = {
  auto_accept: "text-pass border-pass",
  accepted: "text-pass border-pass",
  review: "text-warn border-warn",
  annotate: "text-accent border-accent",
  rejected: "text-block border-block",
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
