"use client";

import { useCallback, useEffect, useState } from "react";

import { api, humanizeError } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast, toastError, toastSuccess } from "@/lib/toast";
import LoadState from "@/components/shell/LoadState";

// Repairing the vocabulary, not only growing it.
//
// merge, rename and retire have existed server-side since the ontology-repair work and were reachable from
// nothing: the only ontology write the interface offered was minting a class. So a mistake in the
// vocabulary - a duplicate, a typo, a class that should never have been minted - could be added to and
// never fixed, which is the exact asymmetry that work set out to close.
//
// Admin-gated on the server. The three actions differ in how far they reach and the UI says so rather than
// presenting them as equivalent buttons:
//
//   merge   rewrites the class of every object carrying it, and is reversible as one run
//   rename  keeps the id, so every object follows the rename; sidecar classes only
//   retire  stops it being offered, and deletes nothing - immutable prediction history still points at it

export default function OntologyRepair() {
  const confirm = useConfirm();
  const [classes, setClasses] = useState<OntologyClass[]>([]);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [fromId, setFromId] = useState<string>("");
  const [toId, setToId] = useState<string>("");
  const [renameId, setRenameId] = useState<string>("");
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [lastMerge, setLastMerge] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      setClasses((await api.ontology()).classes);
    } catch (e) {
      setErr(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const nameOf = (id: string) => classes.find((c) => String(c.id) === id)?.name ?? id;

  const doMerge = async () => {
    if (!fromId || !toId || fromId === toId) return;
    const ok = await confirm({
      title: "Merge classes",
      // Named, and with the reach stated: this is not a vocabulary edit, it rewrites labels.
      body: `Every object and track labelled ${nameOf(fromId)} becomes ${nameOf(toId)}. `
        + "This rewrites the corpus, and it is reversible as one run.",
      confirmLabel: "merge",
    });
    if (!ok) return;
    setBusy(true);
    try {
      const r = await api.ontologyMerge(Number(fromId), Number(toId));
      setLastMerge(r.run_id);
      toast(`merged ${r.moved} object${r.moved === 1 ? "" : "s"}: ${r.from} → ${r.to}`, "success", 5000, {
        label: "undo",
        run: async () => {
          try {
            const back = await api.ontologyRevertMerge(r.run_id);
            toastSuccess(`reverted ${back.reverted}`);
            setLastMerge(null);
          } catch (e) {
            toastError(`undo failed: ${humanizeError(e)}`);
          }
          void load();
        },
      });
      await load();
    } catch (e) {
      // The server refuses a merge that crosses an l0 boundary - object into infra, say - because that is
      // a different kind of thing rather than a finer name for the same one. Surfacing the reason matters:
      // it is the guard doing its job, not a failure.
      toastError(humanizeError(e));
    } finally {
      setBusy(false);
    }
  };

  const doRename = async () => {
    if (!renameId || !newName.trim()) return;
    setBusy(true);
    try {
      const r = await api.ontologyRename(Number(renameId), newName.trim());
      toastSuccess(`renamed to ${r.name}; every object carrying it follows`);
      setNewName("");
      await load();
    } catch (e) {
      toastError(humanizeError(e));
    } finally {
      setBusy(false);
    }
  };

  const doRetire = async (id: number, name: string) => {
    const ok = await confirm({
      title: `Retire ${name}`,
      body: "It stops being offered for new labels. Nothing is deleted: objects and immutable prediction "
        + "history that already point at it keep it.",
      confirmLabel: "retire",
    });
    if (!ok) return;
    setBusy(true);
    try {
      const r = await api.ontologyRetire([id]);
      if (r.retired.length) toastSuccess(`retired ${name}`);
      else toast(`${name} was not retired (it is not a sidecar class)`, "warn");
      await load();
    } catch (e) {
      toastError(humanizeError(e));
    } finally {
      setBusy(false);
    }
  };

  const custom = classes.filter((c) => c.id >= 200);

  return (
    <section className="panel">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        <span>repair the vocabulary</span>
        <span className="ml-auto normal-case text-ink-4">admin only</span>
      </div>
      <div className="p-3 space-y-4">
        {err != null ? (
          <LoadState error={err} onRetry={() => void load()} />
        ) : loading ? (
          <div className="font-mono text-xs text-ink-3/50 animate-pulse py-3 text-center">loading...</div>
        ) : (
          <>
            <div className="space-y-1">
              <div className="font-mono text-[10px] uppercase text-ink-3">merge</div>
              <div className="flex items-center gap-2 flex-wrap font-mono text-[12px]">
                <select value={fromId} onChange={(e) => setFromId(e.target.value)}
                  aria-label="class to merge from"
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink min-w-[160px]">
                  <option value="">from…</option>
                  {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
                <span className="text-ink-3">→</span>
                <select value={toId} onChange={(e) => setToId(e.target.value)}
                  aria-label="class to merge into"
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink min-w-[160px]">
                  <option value="">into…</option>
                  {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
                <button onClick={doMerge} disabled={busy || !fromId || !toId || fromId === toId}
                  className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-40">
                  merge
                </button>
                {lastMerge && <span className="text-ink-4">last run {lastMerge.slice(0, 8)}</span>}
              </div>
              <div className="font-mono text-[10px] text-ink-4">
                rewrites every object and track carrying the class; refused across an l0 boundary
              </div>
            </div>

            <div className="space-y-1">
              <div className="font-mono text-[10px] uppercase text-ink-3">rename</div>
              <div className="flex items-center gap-2 flex-wrap font-mono text-[12px]">
                <select value={renameId} onChange={(e) => setRenameId(e.target.value)}
                  aria-label="class to rename"
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink min-w-[160px]">
                  <option value="">class…</option>
                  {custom.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
                <input value={newName} onChange={(e) => setNewName(e.target.value)}
                  placeholder="new name" aria-label="new class name"
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink" />
                <button onClick={doRename} disabled={busy || !renameId || !newName.trim()}
                  className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-40">
                  rename
                </button>
              </div>
              <div className="font-mono text-[10px] text-ink-4">
                custom classes only; the governed vocabulary is versioned in its file, not edited here
              </div>
            </div>

            <div className="space-y-1">
              <div className="font-mono text-[10px] uppercase text-ink-3">retire</div>
              {custom.length === 0 ? (
                <div className="font-mono text-[11px] text-ink-3">no custom classes to retire</div>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {custom.map((c) => (
                    <button key={c.id} onClick={() => doRetire(c.id, c.name)} disabled={busy}
                      className="font-mono text-[11px] border border-line px-2 py-0.5 hover:border-accent disabled:opacity-40">
                      {c.name} ✕
                    </button>
                  ))}
                </div>
              )}
              <div className="font-mono text-[10px] text-ink-4">
                stops it being offered; deletes nothing, because prediction history still points at it
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
