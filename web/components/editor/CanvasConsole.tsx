"use client";

import { useEffect, useState } from "react";
import { type CanvasOp, subscribeOps, summarizeOps } from "@/lib/canvasOps";
import { level, mb } from "@/lib/console";
import { useSystemStream } from "@/lib/useEventStream";

// The console, inside the canvas.
//
// The editor runs real work on every interaction and looked idle while it did: SAM segmentation, mask
// composition, propagation across frames, drivable inference, an autosave behind all of it. Each was
// fire-and-forget with a one-line flash on the way out, so a slow call and a call that never went look the
// same from the canvas, and the answer to "why is this taking so long" lived on a page you would have to
// leave the frame to reach.
//
// It carries the machine as well as the operations, because the two questions are one question here. "SAM is
// slow" and "a training job holds the GPU" are the same event seen from two ends, and the second is the one
// that tells you whether to wait.
//
// Collapsed by default and remembered. An editor is a place people spend hours, and a panel that reopens on
// every navigation is a panel they close on every navigation.

const KEY = "lbx.canvasConsole.open";

const DOT: Record<CanvasOp["status"], string> = {
  running: "bg-accent", ok: "bg-pass", failed: "bg-block",
};

function elapsed(op: CanvasOp): string {
  const ms = (op.endedAt ?? Date.now()) - op.startedAt;
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export default function CanvasConsole() {
  const [open, setOpen] = useState(false);
  const [ops, setOps] = useState<CanvasOp[]>([]);
  // Only while open. A frame editor holding a server-sent-events connection for a panel nobody has opened is
  // one connection per tab, ticking every three seconds, for nothing.
  const { data: sys } = useSystemStream(open);

  useEffect(() => subscribeOps(setOps), []);

  useEffect(() => {
    try { setOpen(window.localStorage.getItem(KEY) === "1"); } catch { /* private mode */ }
  }, []);

  // The editor owns the keymap, so it raises an event rather than this component adding a second global
  // key listener that would fire inside text inputs the editor already knows to ignore.
  useEffect(() => {
    const onToggle = () => setOpen((o) => {
      try { window.localStorage.setItem(KEY, o ? "0" : "1"); } catch { /* private mode */ }
      return !o;
    });
    window.addEventListener("lbx:canvas-console", onToggle);
    return () => window.removeEventListener("lbx:canvas-console", onToggle);
  }, []);

  const toggle = () => {
    setOpen((o) => {
      try { window.localStorage.setItem(KEY, o ? "0" : "1"); } catch { /* private mode */ }
      return !o;
    });
  };

  // Ticks the elapsed clock on running operations. Without it a slow call shows a frozen duration, which is
  // the thing this panel exists to disprove.
  const [, force] = useState(0);
  useEffect(() => {
    if (!open || !ops.some((o) => o.status === "running")) return;
    const t = setInterval(() => force((n) => n + 1), 500);
    return () => clearInterval(t);
  }, [open, ops]);

  const summary = summarizeOps(ops);
  const gpu = sys?.gpus?.[0];
  const recent = ops.slice().reverse().slice(0, 8);

  return (
    // Above the filmstrip, which is absolutely positioned in this same container at bottom-0. Sitting at
    // bottom-3 put the toggle on top of a thumbnail, which is a control covering a control.
    <div className="absolute bottom-24 right-3 z-30 flex flex-col items-end gap-1.5 pointer-events-none">
      {open && (
        // `reveal` is the app's existing motion primitive: 150ms, and already disabled under
        // prefers-reduced-motion in globals.css, so this respects that setting without repeating it.
        <div className="reveal pointer-events-auto w-[320px] bg-bg-2/95 border border-line backdrop-blur-sm
                        shadow-lg font-mono text-[11px]">
          <div className="flex items-center gap-2 px-2.5 py-1.5 border-b hairline">
            <span className="text-ink-2">canvas</span>
            <span className="text-ink-3">{summary.label}</span>
            <button onClick={toggle} aria-label="close console"
              className="ml-auto text-ink-3 hover:text-ink">esc</button>
          </div>

          <div className="max-h-52 overflow-auto">
            {recent.length ? recent.map((op) => (
              <div key={op.id} className="flex items-center gap-2 px-2.5 py-1 border-b hairline last:border-0">
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT[op.status]}
                                  ${op.status === "running" ? "running-dot" : ""}`} />
                <span className="text-ink-2 truncate">{op.label}</span>
                <span className="ml-auto shrink-0 text-ink-3">{elapsed(op)}</span>
                {op.detail && (
                  <span className={`shrink-0 max-w-[110px] truncate
                                    ${op.status === "failed" ? "text-block" : "text-ink-3"}`}
                    title={op.detail}>{op.detail}</span>
                )}
              </div>
            )) : (
              <div className="px-2.5 py-3 text-center text-ink-3">
                nothing running. Segment, propagate or save and it appears here.
              </div>
            )}
          </div>

          {/* The machine, because "SAM is slow" and "something else holds the GPU" are the same event. */}
          {gpu && (
            <div className="px-2.5 py-1.5 border-t hairline space-y-1">
              <div className="flex justify-between">
                <span className="text-ink-3">gpu</span>
                <span className={level(gpu.memory_used_frac) === "ok" ? "text-ink-2" : "text-warn"}>
                  {mb(gpu.memory_used_mb)} / {mb(gpu.memory_total_mb)} · {Math.round(gpu.utilization_pct ?? 0)}%
                </span>
              </div>
              {gpu.processes.length > 1 && (
                <div className="text-ink-3 truncate" title={gpu.processes.map((p) => p.name).join(", ")}>
                  shared with {gpu.processes.length - 1} other process
                  {gpu.processes.length - 1 === 1 ? "" : "es"}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* The toggle says enough to make opening it a decision rather than a habit. */}
      <button onClick={toggle}
        title="what the canvas is doing (\\)"
        className={`pointer-events-auto flex items-center gap-1.5 px-2 h-6 border font-mono text-[10px]
                    bg-bg-2/90 backdrop-blur-sm transition-colors
                    ${summary.failed ? "border-block text-block"
                      : summary.running ? "border-accent text-accent" : "border-line text-ink-3"}`}>
        {summary.running > 0 && <span className="running-dot" />}
        <span>{summary.label}</span>
      </button>
    </div>
  );
}
