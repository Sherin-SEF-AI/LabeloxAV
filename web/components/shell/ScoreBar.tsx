// A labeled 0..1 metric bar for signals that are NOT model confidence (value-queue uncertainty/diversity/
// rarity/error, criticality, agreement, low-conf fraction). It reuses ConfBar's exact track so the bar
// aesthetic is identical, but takes an explicit label and a tone so the amber accent stays reserved for
// true confidence (ConfBar). Color is earned: neutral by default, never amber unless asked.

const TONE: Record<string, string> = {
  accent: "bg-accent",
  info: "bg-info",
  warn: "bg-warn",
  neutral: "bg-ink-3",
};

export default function ScoreBar({ label, value, tone = "neutral", showValue = true }: {
  label?: string;
  value: number;
  tone?: "accent" | "info" | "warn" | "neutral";
  showValue?: boolean;
}) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className="inline-flex items-center gap-2 font-mono text-[11px]">
      {label && <span className="text-ink-3 shrink-0">{label}</span>}
      <span className="w-14 h-1.5 bg-line relative inline-block">
        <span className={`absolute left-0 top-0 h-full ${TONE[tone]}`} style={{ width: `${w}%` }} />
      </span>
      {showValue && <span className="text-ink-3">{value.toFixed(2)}</span>}
    </span>
  );
}
