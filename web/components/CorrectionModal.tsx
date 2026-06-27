"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { CorrectionCandidate, CorrectionCoverage, CorrectionSuggestion } from "@/lib/types";

// Interactive AI correction: after one fix (Truck->Bus, or an attribute), find visually-similar objects
// that still carry the old value, preview them, deselect wrong matches, and bulk-apply. Shared by the
// frame editor and triage. Operational Materialism tokens; no motion.

export type CorrectionChange = { old: string | number | boolean | null; new: string | number | boolean | null; attrKey?: string };

export default function CorrectionModal({
  objectId,
  kind,
  change,
  onClose,
  onApplied,
}: {
  objectId: string;
  kind: "class" | "attr";
  change: CorrectionChange;
  onClose: () => void;
  onApplied?: (n: number) => void;
}) {
  const [sug, setSug] = useState<CorrectionSuggestion | null>(null);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [threshold, setThreshold] = useState(0.82);
  const [sameCam, setSameCam] = useState(false);
  const [camId, setCamId] = useState<string | null>(null);
  const [cov, setCov] = useState<CorrectionCoverage | null>(null);
  const [loading, setLoading] = useState(true);
  const [applying, setApplying] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // source camera (for the "same camera" filter) + coverage
  useEffect(() => {
    api.object(objectId).then((o) => setCamId(o.cam_id)).catch(() => {});
    api.correctionCoverage().then(setCov).catch(() => {});
  }, [objectId]);

  const query = useCallback(async () => {
    setLoading(true);
    const filters: Record<string, unknown> = {};
    if (sameCam && camId) filters.cam_id = camId;
    try {
      const body =
        kind === "class"
          ? { object_id: objectId, kind, old_class_name: String(change.old), new_class_name: String(change.new), filters, threshold }
          : { object_id: objectId, kind, attr_key: change.attrKey, old_value: change.old, new_value: change.new, filters, threshold };
      const r = await api.correctionSuggest(body as Parameters<typeof api.correctionSuggest>[0]);
      setSug(r);
      setSel(new Set(r.candidates.filter((c) => !c.already).map((c) => c.object_id))); // default-select non-already
    } catch (e) {
      setMsg(String(e));
    } finally {
      setLoading(false);
    }
  }, [objectId, kind, change, threshold, sameCam, camId]);

  // debounce re-query on threshold / filter change
  useEffect(() => {
    const t = setTimeout(query, 250);
    return () => clearTimeout(t);
  }, [query]);

  const toggle = (id: string) =>
    setSel((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  const apply = async () => {
    if (!sel.size) return;
    setApplying(true);
    try {
      const ids = [...sel];
      if (kind === "class") await api.bulkReview(ids, "reclassify", String(change.new));
      else await api.bulkReview(ids, "set_attrs", undefined, undefined, { [change.attrKey as string]: change.new });
      onApplied?.(ids.length);
      onClose();
    } catch (e) {
      setMsg(String(e));
      setApplying(false);
    }
  };

  const title = useMemo(
    () =>
      kind === "class"
        ? `${change.old} → ${change.new}`
        : `${change.attrKey}: ${String(change.old)} → ${String(change.new)}`,
    [kind, change],
  );

  const lowCoverage = cov && cov.pct < 50;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <div className="panel w-full max-w-3xl max-h-[85vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        {/* header */}
        <div className="flex items-center justify-between px-4 py-2 border-b hairline">
          <div className="font-mono text-xs">
            <span className="text-ink-3">you corrected</span>{" "}
            <span className="text-accent">{title}</span>
            {sug && (
              <span className="text-ink-3">
                {" "}— {sug.count} similar {kind === "class" ? `${change.old}` : ""} found
              </span>
            )}
          </div>
          <button onClick={onClose} className="font-mono text-xs text-ink-3 hover:text-ink border border-line px-2 py-0.5">esc</button>
        </div>

        {/* controls */}
        <div className="flex items-center gap-4 px-4 py-2 border-b hairline font-mono text-[11px] text-ink-3">
          <label className="flex items-center gap-2">
            similarity ≥ <span className="text-ink-2 w-8">{threshold.toFixed(2)}</span>
            <input type="range" min={0.7} max={0.97} step={0.01} value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))} className="accent-accent" />
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" checked={sameCam} onChange={(e) => setSameCam(e.target.checked)} disabled={!camId} className="accent-accent" />
            same camera{camId ? ` (${camId})` : ""}
          </label>
          <span className="ml-auto text-ink-2">{sel.size} selected</span>
        </div>

        {lowCoverage && (
          <div className="px-4 py-1.5 border-b hairline font-mono text-[11px] text-warn flex items-center gap-2">
            embeddings cover only {cov?.pct}% of objects — similar-search may miss matches.
            <button onClick={() => { api.computeObjectEmbeddings().then(() => setMsg("embedding corpus in background…")); }}
              className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent">compute embeddings</button>
          </div>
        )}

        {/* grid */}
        <div className="flex-1 overflow-auto p-3">
          {loading ? (
            <div className="text-center font-mono text-xs text-ink-3 py-10">searching…</div>
          ) : !sug || !sug.candidates.length ? (
            <div className="text-center font-mono text-xs text-ink-3 py-10">
              no similar objects above the threshold{msg ? ` — ${msg}` : ""}.
            </div>
          ) : (
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {sug.candidates.map((c: CorrectionCandidate) => {
                const on = sel.has(c.object_id);
                return (
                  <button key={c.object_id} onClick={() => toggle(c.object_id)}
                    className={`relative border ${on ? "border-accent" : "border-line opacity-60"} hover:opacity-100`}
                    title={`${c.class_name} · sim ${c.score.toFixed(2)} · conf ${c.conf.toFixed(2)}`}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={c.crop_url} alt="" className="w-full h-20 object-cover bg-bg-2" />
                    <span className={`absolute top-0 left-0 px-1 font-mono text-[10px] ${on ? "bg-accent text-bg" : "bg-bg/80 text-ink-3"}`}>
                      {on ? "✓" : "○"}
                    </span>
                    <span className="absolute bottom-0 right-0 bg-bg/80 font-mono text-[9px] px-0.5 text-ink-2">{c.score.toFixed(2)}</span>
                    {c.already && <span className="absolute bottom-0 left-0 bg-bg/80 font-mono text-[9px] px-0.5 text-pass">already</span>}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-2 border-t hairline font-mono text-[11px]">
          <button onClick={onClose} className="border border-line px-3 py-1 text-ink-3 hover:text-ink">cancel</button>
          <button onClick={apply} disabled={!sel.size || applying}
            className="border border-pass text-pass px-3 py-1 hover:bg-pass/10 disabled:opacity-40">
            {applying ? "applying…" : `apply ${kind === "class" ? change.new : change.new} to ${sel.size}`}
          </button>
        </div>
      </div>
    </div>
  );
}
