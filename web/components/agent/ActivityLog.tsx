"use client";

import { type ActivityEntry, type ActivityLog as Log, runningCount } from "@/lib/activityLog";

// The console's transcript. It replaces a single shared message line that 64 call sites wrote to, where
// starting a second operation erased the first one's result and a background job landing minutes later
// erased whatever was on screen. These results are the output of the work, not an acknowledgement of it, so
// they persist until cleared rather than auto-dismissing the way a toast does.

const DOT: Record<ActivityEntry["status"], string> = {
  running: "text-warn",
  ok: "text-pass",
  failed: "text-block",
};

const MARK: Record<ActivityEntry["status"], string> = { running: "*", ok: "+", failed: "!" };

function elapsed(e: ActivityEntry): string | null {
  if (e.settledAt === undefined) return null;
  const ms = e.settledAt - e.startedAt;
  // Sub-second work is the common case and printing "0.4s" beside every row is noise; only durations worth
  // noticing get shown, because the point of the number here is spotting the slow ones.
  if (ms < 1000) return null;
  return ms < 60_000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms / 60_000)}m`;
}

function clockOf(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export default function ActivityLog({ log, onClear }: { log: Log; onClear: () => void }) {
  if (log.entries.length === 0) return null;
  const running = runningCount(log);

  return (
    <div className="panel">
      <div className="flex items-center gap-3 px-4 py-2 border-b hairline">
        <div className="text-ink font-medium text-sm">Activity</div>
        <div className="text-ink-3 text-xs">
          {running > 0 ? `${running} running` : `${log.entries.length} recent`}
        </div>
        <button onClick={onClear}
          className="ml-auto font-mono text-[10px] border border-line text-ink-3 px-2 py-0.5 rounded hover:text-ink">
          clear
        </button>
      </div>
      {/* aria-live so a background result is announced when it lands, since it may arrive long after the
          click that started it and nothing else on the page will move. */}
      <div aria-live="polite" className="max-h-64 overflow-y-auto divide-y hairline">
        {log.entries.map((e) => {
          const took = elapsed(e);
          return (
            <div key={e.id} className="px-4 py-1.5 flex items-start gap-2 font-mono text-[11px]">
              <span className={`${DOT[e.status]} shrink-0`} aria-hidden>{MARK[e.status]}</span>
              <span className="text-ink-3 shrink-0 tabular-nums">{clockOf(e.startedAt)}</span>
              <span className="text-ink-2 shrink-0">{e.action}</span>
              <span className="flex-1 break-words text-ink">
                {e.message}
                {e.hint ? <span className="text-warn"> ({e.hint})</span> : null}
              </span>
              {took ? <span className="text-ink-3 shrink-0 tabular-nums">{took}</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
