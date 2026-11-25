"use client";

import { useCallback, useEffect, useState } from "react";

import { api, humanizeError } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast, toastError, toastSuccess } from "@/lib/toast";
import LoadState from "@/components/shell/LoadState";
import { Badge } from "@/components/ui/kit";
import { humanize, shortId } from "@/lib/format";

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
      confirmLabel: "Merge",
    });
    if (!ok) return;
    setBusy(true);
    try {
      const r = await api.ontologyMerge(Number(fromId), Number(toId));
      setLastMerge(r.run_id);
      toast(`Merged ${r.moved} object${r.moved === 1 ? "" : "s"}: ${r.from} → ${r.to}`, "success", 5000, {
        label: "Undo",
        run: async () => {
          try {
            const back = await api.ontologyRevertMerge(r.run_id);
            toastSuccess(`Reverted ${back.reverted}`);
            setLastMerge(null);
          } catch (e) {
            toastError(`Undo failed: ${humanizeError(e)}`);
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
      toastSuccess(`Renamed to ${r.name}; every object carrying it follows`);
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
      confirmLabel: "Retire",
    });
    if (!ok) return;
    setBusy(true);
    try {
      const r = await api.ontologyRetire([id]);
      if (r.retired.length) toastSuccess(`Retired ${name}`);
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
      <div className="panel-head">
        <h2 className="section-title">Repair the class vocabulary</h2>
        <span className="ml-auto"><Badge>Admins only</Badge></span>
      </div>
      <div className="p-4 space-y-5">
        {err != null ? (
          <LoadState error={err} onRetry={() => void load()} />
        ) : loading ? (
          <LoadState loading rows={3} />
        ) : (
          <>
            <div className="grid gap-5 lg:grid-cols-2">
            <div className="space-y-1.5">
              <div className="label-caps">Merge two classes</div>
              <div className="flex items-center gap-2 flex-wrap text-[12.5px]">
                <select value={fromId} onChange={(e) => setFromId(e.target.value)}
                  aria-label="class to merge from"
                  className="field text-[12.5px] min-w-[170px]">
                  <option value="">Merge this class...</option>
                  {classes.map((c) => <option key={c.id} value={c.id}>{humanize(c.name)}</option>)}
                </select>
                <span className="text-ink-3" aria-hidden>→</span>
                <select value={toId} onChange={(e) => setToId(e.target.value)}
                  aria-label="class to merge into"
                  className="field text-[12.5px] min-w-[170px]">
                  <option value="">...into this one</option>
                  {classes.map((c) => <option key={c.id} value={c.id}>{humanize(c.name)}</option>)}
                </select>
                <button onClick={doMerge} disabled={busy || !fromId || !toId || fromId === toId}
                  className="btn">
                  Merge
                </button>
                {lastMerge && <span className="text-ink-3 text-[12px]">Last run <span className="font-mono">{shortId(lastMerge)}</span></span>}
              </div>
              <div className="section-desc">
                Moves every object and track to the second class. Refused when the two are different kinds of
                thing (an object into road infrastructure, say).
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="label-caps">Rename a custom class</div>
              <div className="flex items-center gap-2 flex-wrap text-[12.5px]">
                <select value={renameId} onChange={(e) => setRenameId(e.target.value)}
                  aria-label="class to rename"
                  className="field text-[12.5px] min-w-[170px]">
                  <option value="">Choose a class...</option>
                  {custom.map((c) => <option key={c.id} value={c.id}>{humanize(c.name)}</option>)}
                </select>
                <input value={newName} onChange={(e) => setNewName(e.target.value)}
                  placeholder="New name" aria-label="new class name"
                  className="field text-[12.5px]" />
                <button onClick={doRename} disabled={busy || !renameId || !newName.trim()}
                  className="btn">
                  Rename
                </button>
              </div>
              <div className="section-desc">
                Custom classes only. The governed vocabulary is versioned in its own file and is not edited here.
              </div>
            </div>
            </div>

            <div className="space-y-1.5">
              <div className="label-caps">Retire a custom class</div>
              {custom.length === 0 ? (
                <div className="text-[12.5px] text-ink-3">There are no custom classes to retire.</div>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {custom.map((c) => (
                    <button key={c.id} onClick={() => doRetire(c.id, c.name)} disabled={busy}
                      aria-label={`Retire ${c.name}`} data-tip={`Retire ${c.name}`}
                      className="chip hover:border-block/60 hover:text-block disabled:opacity-40">
                      {humanize(c.name)} <span aria-hidden className="text-ink-3">✕</span>
                    </button>
                  ))}
                </div>
              )}
              <div className="section-desc">
                A retired class is no longer offered for new labels. Nothing is deleted: existing labels and
                prediction history keep it.
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
