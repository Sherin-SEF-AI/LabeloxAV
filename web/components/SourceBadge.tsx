// A small pill that shows where an object or session came from (imported dataset vs your own work).
import { objectSource, sessionOrigin } from "@/lib/source";

export function Badge({ label, cls }: { label: string; cls: string }) {
  return (
    <span className={`inline-block px-1.5 py-[1px] border rounded-sm text-[9px] font-mono uppercase tracking-wide leading-tight ${cls}`}>
      {label}
    </span>
  );
}

export function ObjectSourceBadge({ source, importFormat }: { source?: string; importFormat?: string | null }) {
  const s = objectSource(source, importFormat);
  return <Badge label={s.label} cls={s.cls} />;
}

export function SessionOriginBadge({ origin }: { origin?: string }) {
  const o = sessionOrigin(origin);
  return <Badge label={o.label} cls={o.cls} />;
}
