"use client";

// M-F.5 scene graph + VLM dataset generation: propose typed scene-graph relations between the frame's objects
// (geometry-proposed, human-confirmed), then generate a GROUNDED multimodal training target (scene description,
// hazards, per-agent intent, ego-action justification) from the confirmed labels, review it, and it becomes part
// of the exportable VLM dataset. Nothing enters the dataset without human approval.

import { useCallback, useEffect, useState } from "react";
import { api , humanizeError } from "@/lib/api";
import { Busy } from "@/components/Spinner";

type Rel = Awaited<ReturnType<typeof api.relationsList>>[number];
type Target = Awaited<ReturnType<typeof api.vlmTargetsList>>[number];

export default function SceneGraphPanel({ frameId, embedded = false }: { frameId: string; embedded?: boolean }) {
  const [rels, setRels] = useState<Rel[]>([]);
  const [targets, setTargets] = useState<Target[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    setRels(await api.relationsList(frameId).catch(() => []));
    setTargets(await api.vlmTargetsList(frameId).catch(() => []));
  }, [frameId]);
  useEffect(() => { load(); }, [load]);

  const propose = async () => {
    setBusy(true); setMsg(null);
    try { const r = await api.relationsPropose(frameId); setMsg(`proposed ${r.proposed} relation(s)`); await load(); }
    catch (e) { setMsg("propose failed: " + humanizeError(e)); } finally { setBusy(false); }
  };
  const relStatus = async (id: string, status: string) => {
    setBusy(true);
    try { await api.relationStatus(id, status); await load(); } catch (e) { setMsg(humanizeError(e)); } finally { setBusy(false); }
  };
  const generate = async () => {
    setBusy(true); setMsg("generating grounded target (VLM)...");
    try { const r = await api.vlmTargetGenerate(frameId); setMsg(`generated a target from ${r.grounding.objects} objects, ${r.grounding.relations} relations`); await load(); }
    catch (e) { setMsg("generate failed: " + humanizeError(e)); } finally { setBusy(false); }
  };
  const tgtStatus = async (id: string, status: string) => {
    setBusy(true);
    try { await api.vlmTargetStatus(id, status); await load(); } catch (e) { setMsg(humanizeError(e)); } finally { setBusy(false); }
  };

  return (
    <div className={`space-y-2 font-mono text-[10px] ${embedded ? "" : "border-t hairline p-2"}`}>
      <div className="flex items-center justify-between">
        {!embedded && <span className="uppercase text-ink-3">scene graph + vlm dataset</span>}
        <div className={`flex gap-1 ${embedded ? "ml-auto" : ""}`}>
          <button onClick={propose} disabled={busy} className="flex items-center gap-1 border border-line px-1.5 rounded text-ink-3 hover:border-accent disabled:opacity-40">{busy && <Busy />}propose relations</button>
          <button onClick={generate} disabled={busy} className={`flex items-center gap-1 border border-info/50 text-info px-1.5 rounded hover:bg-info/10 disabled:opacity-40 ${busy ? "running" : ""}`}>{busy && <Busy className="border-t-info" />}generate target</button>
        </div>
      </div>

      {/* scene-graph relations */}
      {rels.length > 0 && (
        <div className="space-y-1">
          <div className="text-ink-3 uppercase text-[9px]">relations ({rels.length})</div>
          {rels.slice(0, 20).map((r) => (
            <div key={r.relationship_id} className="flex items-center gap-1">
              <span className="flex-1 truncate text-ink-2">{r.from_name} <span className="text-accent">{r.kind.replace(/_/g, " ")}</span> {r.to_name}</span>
              <span className={`text-[9px] uppercase ${r.status === "confirmed" ? "text-pass" : "text-warn"}`}>{r.status}</span>
              {r.status !== "confirmed" && (
                <>
                  <button onClick={() => relStatus(r.relationship_id, "confirmed")} disabled={busy} title="confirm" className="text-pass hover:text-accent">✓</button>
                  <button onClick={() => relStatus(r.relationship_id, "rejected")} disabled={busy} title="reject" className="text-block hover:text-accent">×</button>
                </>
              )}
            </div>
          ))}
        </div>
      )}

      {/* generated VLM targets (grounded, review-gated) */}
      {targets.map((t) => {
        const c = t.content as { scene_description?: string; hazards?: string[]; ego_action?: { action?: string; justification?: string } };
        return (
          <div key={t.target_id} className="border border-line rounded p-1.5 bg-bg-2 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-ink-3 uppercase text-[9px]">vlm target · {(t.grounding.object_ids || []).length} objs · {(t.grounding.relation_ids || []).length} rels</span>
              <span className="flex items-center gap-1">
                <span className={`text-[9px] uppercase ${t.status === "approved" ? "text-pass" : t.status === "rejected" ? "text-block" : "text-warn"}`}>{t.status}</span>
                {t.status === "generated" && (
                  <>
                    <button onClick={() => tgtStatus(t.target_id, "approved")} disabled={busy} title="approve for export" className="text-pass hover:text-accent">approve</button>
                    <button onClick={() => tgtStatus(t.target_id, "rejected")} disabled={busy} title="reject" className="text-block hover:text-accent">reject</button>
                  </>
                )}
              </span>
            </div>
            {c.scene_description && <div className="text-ink-2 leading-snug">{c.scene_description}</div>}
            {c.hazards && c.hazards.length > 0 && <div className="text-warn">hazards: {c.hazards.join("; ")}</div>}
            {c.ego_action?.action && <div className="text-ink-3">ego: <span className="text-accent">{c.ego_action.action}</span> - {c.ego_action.justification}</div>}
          </div>
        );
      })}
      {msg && <div className="text-ink-2">{msg}</div>}
    </div>
  );
}
