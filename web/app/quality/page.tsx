"use client";

import { useEffect, useState } from "react";
import { toast } from "@/lib/toast";
import { useRouter } from "next/navigation";
import { api, type GoldSetRow, type QualitySheet , humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Stat } from "@/components/ui/kit";
import { ago, humanize, num, plural } from "@/lib/format";
import { useQuery } from "@/lib/useQuery";

// Gate B (M9): the quality sheet. Per-class precision/recall against a sealed gold set, Safe-mIoU,
// and calibration ECE. Seal a gold set + fit isotonic calibration from here. Measurement (GPU) runs
// out of band via `make m9`; this page reads the cached numbers.

function Section({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">{title}</h2>
        {right && <div className="ml-auto flex items-center gap-2 font-normal">{right}</div>}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

const f3 = (v: number | null | undefined) => (v != null ? v.toFixed(3) : "-");

export default function QualityPage() {
  const router = useRouter();
  const goldQ = useQuery(["goldSets"], () => api.goldSets());
  const goldSets: GoldSetRow[] = goldQ.data ?? [];
  const [selected, setSelected] = useState<string | null>(null);
  // Keyed by the gold set, so flipping between two sets redraws from the cache. The previous sheet
  // stays up while the next loads, and a set with no sheet (or a failed read) shows none, as before.
  const sheetQ = useQuery(selected ? ["qualitySheet", selected] : null, () => api.qualitySheet(selected as string),
    { keepPrevious: true });
  const sheet: QualitySheet | null = sheetQ.error ? null : sheetQ.data ?? null;
  const [busy, setBusy] = useState<string | null>(null);

  const refreshGoldSets = goldQ.reload;

  // Open on the first gold set once the list arrives, unless one is already chosen.
  const firstGoldId = goldSets[0]?.gold_id;
  useEffect(() => { if (firstGoldId) setSelected((cur) => cur ?? firstGoldId); }, [firstGoldId]);

  async function onSeal() {
    setBusy("seal");
    try {
      const r = await api.sealGold({ name: "fleet-v1" });
      await refreshGoldSets();
      setSelected(r.gold_id);
    } catch (e) {
      toast(humanizeError(e));
    } finally {
      setBusy(null);
    }
  }

  const selectedUsable = goldSets.find((g) => g.gold_id === selected)?.usable !== false;

  async function measure() {
    if (!selected) return;
    setBusy("measure");
    try {
      const r = await api.measureGold(selected);
      toast(`Measuring against ${r.n_alive} objects; the results appear here when it finishes`);
    } catch (e) {
      toast(humanizeError(e));
    } finally {
      setBusy(null);
    }
  }

  async function onFit() {
    setBusy("fit");
    try {
      const r = await api.fitCalibration({ gold_id: selected ?? undefined });
      toast(r.trustworthy
        ? `Fitted isotonic: ${r.n_train} pairs (${r.negatives} rejected), in-sample ECE ${r.report.ece ?? "n/a"}\n${r.uri}`
        : `Fitted, but not usable: ${r.warning}`, r.trustworthy ? "success" : "error");
    } catch (e) {
      toast(humanizeError(e));
    } finally {
      setBusy(null);
    }
  }

  const m = sheet?.metrics ?? {};
  const pr = m.per_class_pr ?? {};
  const prRows = Object.entries(pr).sort((a, b) => a[1].ap50 - b[1].ap50);

  const sel = goldSets.find((g) => g.gold_id === selected) ?? null;

  return (
    <PageShell
      active="QUALITY"
      subtitle="Score the champion model against a sealed set of human-verified labels."
      primaryAction={
        <>
          <button onClick={onFit} disabled={busy !== null} className="btn"
            data-tip="Fit a confidence calibration on the selected gold set, so a score of 0.8 means right 80% of the time">
            {busy === "fit" ? "Fitting..." : "Fit calibration"}
          </button>
          <button onClick={onSeal} disabled={busy !== null} className="btn btn-primary"
            data-tip="Freeze the current human-verified labels into a new gold set">
            {busy === "seal" ? "Sealing..." : "Seal new gold set"}
          </button>
        </>
      }
    >
      <div className="page-wrap space-y-4">
        <Section title="Gold sets" right={<span className="text-[12px] text-ink-3">{plural(goldSets.length, "set")}</span>}>
          {goldQ.error && !goldQ.data ? (
            <LoadState error={goldQ.error} onRetry={() => void refreshGoldSets()} />
          ) : goldQ.loading && !goldQ.data ? (
            <LoadState loading rows={3} />
          ) : goldSets.length ? (
            <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" role="listbox" aria-label="Gold sets">
              {goldSets.map((g) => {
                const on = selected === g.gold_id;
                return (
                  <button
                    key={g.gold_id}
                    role="option"
                    aria-selected={on}
                    onClick={() => setSelected(g.gold_id)}
                    className={`text-left rounded-md border px-3 py-2.5 transition-colors ${
                      on ? "border-accent bg-accent/10" : "border-line hover:border-ink-3 bg-head/30"
                    }`}
                    title={g.gold_id}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-[13px] font-medium text-ink truncate flex-1">{g.name}</span>
                      <Badge state={g.measured ? "pass" : "muted"}>{g.measured ? "Measured" : "Not measured"}</Badge>
                    </div>
                    <div className="text-[12px] text-ink-3 mt-1">
                      {plural(g.n_objects, "object")} · {plural(g.n_frames, "frame")}
                      {g.created_at ? ` · ${ago(g.created_at)}` : ""}
                    </div>
                    {/* A gold set is a list of object ids, so it rots when those objects are deleted and goes
                        on claiming its sealed size. A rotted set seeds no honeypots and measures nothing, and
                        it looked identical to a healthy one here. */}
                    {g.usable === false && (
                      <div className="text-[12px] text-block mt-1"
                        title={`${g.n_missing} of ${g.n_objects} objects no longer exist; this set grades nothing`}>
                        Unusable: its objects were deleted
                      </div>
                    )}
                    {g.usable && g.n_missing > 0 && (
                      <div className="text-[12px] text-warn mt-1"
                        title={`${g.n_missing} of ${g.n_objects} objects no longer exist`}>
                        {num(g.n_alive)} of {num(g.n_objects)} objects still exist
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          ) : (
            <EmptyState compact icon="chart" title="No gold sets yet"
              action={<button onClick={onSeal} disabled={busy !== null} className="btn btn-primary">Seal new gold set</button>}>
              A gold set freezes your human-verified labels so every model can be scored against the same answers.
            </EmptyState>
          )}
        </Section>

        {sheet?.found && sheet.measured ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <Stat label="mAP@50" value={f3(m.map50)} />
              <Stat label="mAP@50-95" value={f3(m.map)} />
              <Stat label="Precision" value={f3(m.precision)} />
              <Stat label="Recall" value={f3(m.recall)} />
              <Stat
                label="Safe-mIoU"
                value={f3(m.safe_miou)}
                sub={m.safety_weight ? `safety weight ${m.safety_weight}` : undefined}
              />
              <Stat
                label="Calibration error"
                value={m.calibration?.ece != null ? m.calibration.ece.toFixed(3) : "n/a"}
                sub={m.calibration?.n_train ? `ECE over ${num(m.calibration.n_train)} pairs` : "not fitted yet"}
              />
            </div>

            <Section title="Precision and recall by class"
              right={<span className="text-[12px] text-ink-3">{plural(prRows.length, "class", "classes")}, weakest first</span>}>
              {prRows.length ? (
                <div className="overflow-x-auto">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th className="w-[34%]">
                          <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-info inline-block" />Precision</span>
                        </th>
                        <th className="w-[34%]">
                          <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-accent inline-block" />Recall</span>
                        </th>
                        <th className="num">AP50</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prRows.map(([name, v]) => (
                        <tr key={name}>
                          <td className="text-ink" title={name}>{humanize(name)}</td>
                          <td>
                            <div className="flex items-center gap-2">
                              <span className="flex-1 h-2 rounded-sm bg-line relative overflow-hidden">
                                <span className="absolute left-0 top-0 h-full bg-info" style={{ width: `${v.precision * 100}%` }} />
                              </span>
                              <span className="w-10 text-right tabular-nums text-ink-3">{v.precision.toFixed(2)}</span>
                            </div>
                          </td>
                          <td>
                            <div className="flex items-center gap-2">
                              <span className="flex-1 h-2 rounded-sm bg-line relative overflow-hidden">
                                <span className="absolute left-0 top-0 h-full bg-accent" style={{ width: `${v.recall * 100}%` }} />
                              </span>
                              <span className="w-10 text-right tabular-nums text-ink-3">{v.recall.toFixed(2)}</span>
                            </div>
                          </td>
                          <td className="num">{v.ap50.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-[12.5px] text-ink-3 py-4 text-center">No per-class metrics were recorded.</div>
              )}
              <div className="mt-3 text-[12px] text-ink-3">Model scored: <span className="font-mono text-ink-2">{m.weights ?? "-"}</span></div>
            </Section>
          </>
        ) : sheetQ.loading && selected ? (
          <LoadState loading rows={4} />
        ) : (
          <div className="panel">
            {sheet?.found ? (
              <EmptyState icon="chart" title={`${sel?.name ?? "This gold set"} has not been scored yet`}
                action={
                  // It used to print `make m9 ARGS="--gold <id>"` here, at a reviewer who has no shell. A
                  // rotted set is refused by the server rather than spending GPU measuring nothing.
                  <button onClick={measure} disabled={busy === "measure" || !selectedUsable}
                    data-tip={selectedUsable
                      ? "Score it against the current champion (runs on the GPU in the background)"
                      : "This gold set's objects no longer exist, so there is nothing to measure against"}
                    className="btn btn-primary">
                    {busy === "measure" ? "Measuring..." : "Measure now"}
                  </button>
                }>
                {selectedUsable
                  ? "Measuring scores the current champion model against it. The results appear here when the run finishes."
                  : "Its objects have been deleted, so it cannot be measured. Seal a new gold set instead."}
              </EmptyState>
            ) : (
              <EmptyState icon="chart" title="Choose a gold set">
                Pick a gold set above to see how the champion model scores against it.
              </EmptyState>
            )}
          </div>
        )}
      </div>
    </PageShell>
  );
}
