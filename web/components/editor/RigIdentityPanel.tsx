"use client";

// M-MC.2 rig identity panel: the rig-first object list for the focused group. It shows one entry per physical
// object (linked identities grouped across cameras, conflicts first), then the still-unlinked singletons that
// the reviewer can select and link into one identity. A DINOv3 appearance assist proposes cross-camera pairs;
// it never links on its own (agents propose, gates dispose) - the reviewer accepts a suggestion with one click.
// This tier needs no calibration: linking is identity only, never a geometric projection.

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { LinkSuggestion, RigObjectsResponse } from "@/lib/types";
import { classColor } from "@/lib/colors";

export default function RigIdentityPanel({ sessionId, groupId, onClose, onSelectObject }: {
  sessionId: string;
  groupId: string;
  onClose: () => void;
  onSelectObject?: (objectId: string, cam: string) => void;
}) {
  const [data, setData] = useState<RigObjectsResponse | null>(null);
  const [suggestions, setSuggestions] = useState<LinkSuggestion[] | null>(null);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try { setData(await api.multicamRigObjects(sessionId, groupId)); }
    catch (e) { setErr(String(e)); }
  }, [sessionId, groupId]);

  useEffect(() => { refresh(); }, [refresh]);

  const toggle = (oid: string) => setSel((s) => { const n = new Set(s); n.has(oid) ? n.delete(oid) : n.add(oid); return n; });

  const linkSelected = async () => {
    if (sel.size < 2) return;
    setBusy(true); setErr(null);
    try { await api.multicamLink({ session_id: sessionId, group_id: groupId, object_ids: [...sel], source: "manual" }); setSel(new Set()); await refresh(); }
    catch (e) { setErr("link failed: " + String(e)); }
    finally { setBusy(false); }
  };
  const acceptSuggestion = async (s: LinkSuggestion) => {
    setBusy(true); setErr(null);
    try { await api.multicamLink({ session_id: sessionId, group_id: groupId, object_ids: [s.a, s.b], source: "appearance" }); setSuggestions((x) => (x ? x.filter((y) => y !== s) : x)); await refresh(); }
    catch (e) { setErr("accept failed: " + String(e)); }
    finally { setBusy(false); }
  };
  const unlink = async (oid: string) => {
    setBusy(true); setErr(null);
    try { await api.multicamUnlink(oid); await refresh(); }
    catch (e) { setErr("unlink failed: " + String(e)); }
    finally { setBusy(false); }
  };
  const suggest = async () => {
    setBusy(true); setErr(null);
    try { const r = await api.multicamSuggestLinks(sessionId, groupId); setSuggestions(r.suggestions); }
    catch (e) { setErr("suggest failed: " + String(e)); }
    finally { setBusy(false); }
  };

  const dot = (cid: number | null) => <span className="inline-block w-2 h-2 rounded-full mr-1 align-middle" style={{ background: cid != null ? classColor(cid) : "#6C727A" }} />;

  return (
    <div className="absolute top-3 left-3 z-30 w-64 max-h-[80%] flex flex-col panel bg-bg/95 border border-line rounded shadow-lg font-mono text-[11px]">
      <div className="flex items-center justify-between px-2 py-1.5 border-b hairline">
        <span className="uppercase text-ink-2 tracking-wide">rig identities</span>
        <button onClick={onClose} title="close" className="text-ink-3 hover:text-ink px-1">×</button>
      </div>
      <div className="flex items-center gap-1 px-2 py-1.5 border-b hairline">
        <button onClick={suggest} disabled={busy} className="border border-line px-1.5 py-0.5 rounded text-ink-3 hover:border-accent disabled:opacity-40">suggest (appearance)</button>
        <button onClick={linkSelected} disabled={busy || sel.size < 2} title="link the selected objects into one identity"
          className="border border-accent/60 text-accent bg-accent/10 px-1.5 py-0.5 rounded hover:bg-accent/20 disabled:opacity-40">link {sel.size || ""}</button>
      </div>
      {err && <div className="px-2 py-1 text-block text-[10px] border-b hairline">{err}</div>}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {suggestions && suggestions.length > 0 && (
          <div className="space-y-1">
            <div className="text-ink-3 uppercase text-[9px]">appearance suggestions</div>
            {suggestions.map((s, i) => (
              <div key={i} className="flex items-center justify-between gap-1 border border-info/30 rounded px-1.5 py-1">
                <span className="text-ink-2 truncate">{s.cam_a} ↔ {s.cam_b} <span className="text-ink-3">cos {s.cos}</span></span>
                <button onClick={() => acceptSuggestion(s)} disabled={busy} className="border border-info/50 text-info px-1 rounded hover:bg-info/10 shrink-0">link</button>
              </div>
            ))}
          </div>
        )}
        {suggestions && suggestions.length === 0 && <div className="text-ink-3 text-[10px]">no appearance suggestions above threshold</div>}

        <div className="space-y-1">
          <div className="text-ink-3 uppercase text-[9px]">identities ({data?.rig_objects.length ?? 0})</div>
          {data?.rig_objects.map((r) => (
            <div key={r.rig_object_id} className={`border rounded px-1.5 py-1 ${r.conflict ? "border-block/60" : "border-line"}`}>
              <div className="flex items-center justify-between">
                <span className="text-ink-2 truncate">{dot(r.class_id)}{r.class_name ?? "?"}</span>
                {r.conflict && <span className="text-block text-[9px] uppercase">conflict</span>}
              </div>
              <div className="mt-0.5 flex flex-wrap gap-1">
                {r.members.map((m) => (
                  <button key={m.object_id} onClick={() => onSelectObject?.(m.object_id, m.cam)} title={`${m.cam}: ${m.class_name}`}
                    className="border border-line rounded px-1 text-[9px] text-ink-3 hover:border-accent">{m.cam}</button>
                ))}
                <button onClick={() => unlink(r.members[0].object_id)} disabled={busy} title="dissolve / remove first member"
                  className="text-ink-3 hover:text-block text-[9px] ml-auto">unlink</button>
              </div>
            </div>
          ))}
          {data && data.rig_objects.length === 0 && <div className="text-ink-3 text-[10px]">no linked identities yet</div>}
        </div>

        <div className="space-y-1">
          <div className="text-ink-3 uppercase text-[9px]">unlinked ({data?.singletons.length ?? 0})</div>
          {data?.singletons.map((s) => (
            <label key={s.object_id} className="flex items-center gap-1.5 cursor-pointer hover:text-ink">
              <input type="checkbox" checked={sel.has(s.object_id)} onChange={() => toggle(s.object_id)} className="accent-accent" />
              <span className="text-ink-3 text-[9px] uppercase w-10 shrink-0">{s.cam}</span>
              <span className="text-ink-2 truncate">{dot(s.class_id)}{s.class_name}</span>
            </label>
          ))}
          {data && data.singletons.length === 0 && <div className="text-ink-3 text-[10px]">all objects linked</div>}
        </div>
      </div>
    </div>
  );
}
