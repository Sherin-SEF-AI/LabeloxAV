"use client";

// M-F.3 natural-language bulk editing: type an operation over the frame or session ("select every parked
// autorickshaw", "reclassify the fallback objects to push_cart", "flag every pedestrian with no visible mask").
// The command is PREVIEWED first (what it would touch, read-only) and only applied on explicit confirmation,
// landing on a reviewable, reversible agent run. Language proposes; the gate disposes.

import { useState } from "react";
import { api } from "@/lib/api";

type Preview = Awaited<ReturnType<typeof api.nlEditPreview>>;

export default function BulkEditBar({ frameId, sessionId, onApplied }: {
  frameId: string;
  sessionId?: string | null;
  onApplied?: () => void;
}) {
  const [cmd, setCmd] = useState("");
  const [scope, setScope] = useState<"frame" | "session">("frame");
  const [useVlm, setUseVlm] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const doPreview = async () => {
    if (!cmd.trim()) return;
    setBusy(true); setMsg(null); setPreview(null);
    try {
      const body = scope === "session" && sessionId ? { command: cmd, session_id: sessionId, use_vlm: useVlm }
        : { command: cmd, frame_id: frameId, use_vlm: useVlm };
      setPreview(await api.nlEditPreview(body));
    } catch (e) { setMsg("preview failed: " + String(e)); }
    finally { setBusy(false); }
  };

  const doApply = async () => {
    if (!preview || preview.count === 0) return;
    setBusy(true); setMsg(null);
    try {
      const r = await api.nlEditApply({ command: cmd, object_ids: preview.objects.map((o) => o.object_id) });
      setMsg(`${r.operation}: ${r.edited} object(s) applied, routed to ${r.routed_to} (reversible run ${r.run_id.slice(0, 8)})`);
      setPreview(null); setCmd("");
      onApplied?.();
    } catch (e) { setMsg("apply failed: " + String(e)); }
    finally { setBusy(false); }
  };

  const op = preview?.plan?.operation;
  const canApply = preview && preview.count > 0 && op !== "select" && !(op === "reclassify" && preview.plan.to_class_id == null);

  return (
    <div className="border-t hairline p-2 space-y-1.5 font-mono text-[10px]">
      <div className="uppercase text-ink-3">natural-language bulk edit</div>
      <div className="flex gap-1">
        <input value={cmd} onChange={(e) => setCmd(e.target.value)} onKeyDown={(e) => e.key === "Enter" && doPreview()}
          placeholder="e.g. reclassify the fallback objects to push_cart"
          className="flex-1 min-w-0 bg-bg-2 border border-line rounded px-1.5 py-1 text-ink-2 focus:border-accent outline-none" />
        <button onClick={doPreview} disabled={busy || !cmd.trim()} className="border border-line px-1.5 rounded text-ink-3 hover:border-accent disabled:opacity-40">preview</button>
      </div>
      <div className="flex items-center gap-2 text-ink-3">
        <button onClick={() => setScope("frame")} className={scope === "frame" ? "text-accent" : "hover:text-ink"}>this frame</button>
        {sessionId && <button onClick={() => setScope("session")} className={scope === "session" ? "text-accent" : "hover:text-ink"}>this session</button>}
        <label className="ml-auto flex items-center gap-1 cursor-pointer" title="use the VLM to confirm referential phrases (near/behind/...)">
          <input type="checkbox" checked={useVlm} onChange={(e) => setUseVlm(e.target.checked)} className="accent-accent" /> VLM refine
        </label>
      </div>

      {preview && (
        <div className="border border-line rounded p-1.5 space-y-1 bg-bg-2">
          <div className="flex items-center justify-between">
            <span className="text-ink-2">{op} would touch <span className="text-accent">{preview.count}</span> object(s){preview.vlm_refined ? " (VLM-refined)" : ""}</span>
            {canApply && <button onClick={doApply} disabled={busy} className="border border-accent/60 text-accent bg-accent/10 px-1.5 py-0.5 rounded hover:bg-accent/20">confirm and apply</button>}
          </div>
          {preview.warnings.map((w, i) => <div key={i} className="text-warn">{w}</div>)}
          <div className="flex flex-wrap gap-1 max-h-16 overflow-y-auto">
            {preview.objects.slice(0, 30).map((o) => (
              <span key={o.object_id} className="border border-line rounded px-1 text-ink-3">{o.class_name}</span>
            ))}
            {preview.count > 30 && <span className="text-ink-3">+{preview.count - 30} more</span>}
          </div>
          {op === "select" && <div className="text-ink-3">select is a preview only; use reclassify or flag to change data</div>}
        </div>
      )}
      {msg && <div className="text-ink-2">{msg}</div>}
    </div>
  );
}
