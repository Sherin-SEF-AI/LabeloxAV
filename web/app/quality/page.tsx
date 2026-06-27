"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type GoldSetRow, type QualitySheet } from "@/lib/api";
import TopNav from "@/components/TopNav";

// Gate B (M9): the quality sheet. Per-class precision/recall against a sealed gold set, Safe-mIoU,
// and calibration ECE. Seal a gold set + fit isotonic calibration from here. Measurement (GPU) runs
// out of band via `make m9`; this page reads the cached numbers.

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="panel px-3 py-3">
      <div className="font-mono text-[11px] uppercase text-ink-3">{label}</div>
      <div className="font-mono text-2xl text-ink mt-1">{value}</div>
      {sub && <div className="font-mono text-[11px] text-ink-3 mt-0.5">{sub}</div>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        {title}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

export default function QualityPage() {
  const router = useRouter();
  const [goldSets, setGoldSets] = useState<GoldSetRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [sheet, setSheet] = useState<QualitySheet | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function refreshGoldSets() {
    const gs = await api.goldSets();
    setGoldSets(gs);
    if (!selected && gs.length) setSelected(gs[0].gold_id);
  }

  useEffect(() => {
    refreshGoldSets();
  }, []);

  useEffect(() => {
    if (selected) api.qualitySheet(selected).then(setSheet).catch(() => setSheet(null));
  }, [selected]);

  async function onSeal() {
    setBusy("seal");
    try {
      const r = await api.sealGold({ name: "fleet-v1" });
      await refreshGoldSets();
      setSelected(r.gold_id);
    } catch (e) {
      alert(String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onFit() {
    setBusy("fit");
    try {
      const r = await api.fitCalibration({ gold_id: selected ?? undefined });
      alert(`Fitted isotonic: ${r.n_train} pairs, ECE ${r.report.ece ?? "n/a"}\n${r.uri}`);
    } catch (e) {
      alert(String(e));
    } finally {
      setBusy(null);
    }
  }

  const m = sheet?.metrics ?? {};
  const pr = m.per_class_pr ?? {};
  const prRows = Object.entries(pr).sort((a, b) => a[1].ap50 - b[1].ap50);

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="QUALITY" right={
        <>
          <button onClick={onSeal} disabled={busy !== null}
            className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-50">
            {busy === "seal" ? "sealing..." : "seal gold"}
          </button>
          <button onClick={onFit} disabled={busy !== null}
            className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-50">
            {busy === "fit" ? "fitting..." : "fit calibration"}
          </button>
        </>
      } />

      <main className="flex-1 overflow-auto p-4 space-y-4">
        <Section title={`gold sets (${goldSets.length})`}>
          {goldSets.length ? (
            <div className="flex flex-wrap gap-2">
              {goldSets.map((g) => (
                <button
                  key={g.gold_id}
                  onClick={() => setSelected(g.gold_id)}
                  className={`font-mono text-[11px] border px-2 py-1 ${
                    selected === g.gold_id ? "border-accent text-ink" : "border-line text-ink-3"
                  }`}
                  title={g.gold_id}
                >
                  {g.name} · {g.n_objects} obj · {g.n_frames} fr{" "}
                  <span className={g.measured ? "text-pass" : "text-warn"}>
                    {g.measured ? "measured" : "unmeasured"}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">
              no gold sets sealed yet — click &quot;seal gold&quot;
            </div>
          )}
        </Section>

        {sheet?.found && sheet.measured ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <Stat label="mAP@50" value={m.map50 != null ? m.map50.toFixed(3) : "-"} />
              <Stat label="mAP@50-95" value={m.map != null ? m.map.toFixed(3) : "-"} />
              <Stat label="precision" value={m.precision != null ? m.precision.toFixed(3) : "-"} />
              <Stat label="recall" value={m.recall != null ? m.recall.toFixed(3) : "-"} />
              <Stat
                label="safe-mIoU"
                value={m.safe_miou != null ? m.safe_miou.toFixed(3) : "-"}
                sub={m.safety_weight ? `weight ${m.safety_weight}` : undefined}
              />
              <Stat
                label="calib ECE"
                value={m.calibration?.ece != null ? m.calibration.ece.toFixed(3) : "n/a"}
                sub={m.calibration?.n_train ? `${m.calibration.n_train} pairs` : "not fitted"}
              />
            </div>

            <Section title={`per-class precision / recall (${prRows.length} classes)`}>
              <div className="space-y-1">
                {prRows.map(([name, v]) => (
                  <div key={name} className="flex items-center gap-2">
                    <div className="w-32 shrink-0 font-mono text-[11px] text-ink-2 truncate" title={name}>
                      {name}
                    </div>
                    <div className="flex-1 h-3 bg-line relative" title={`P ${v.precision}`}>
                      <div className="absolute left-0 top-0 h-full bg-info" style={{ width: `${v.precision * 100}%` }} />
                    </div>
                    <div className="w-10 text-right font-mono text-[11px] text-ink-3">{v.precision.toFixed(2)}</div>
                    <div className="flex-1 h-3 bg-line relative" title={`R ${v.recall}`}>
                      <div className="absolute left-0 top-0 h-full bg-accent" style={{ width: `${v.recall * 100}%` }} />
                    </div>
                    <div className="w-10 text-right font-mono text-[11px] text-ink-3">{v.recall.toFixed(2)}</div>
                  </div>
                ))}
                {!prRows.length && (
                  <div className="font-mono text-xs text-ink-3 py-4 text-center">no per-class metrics</div>
                )}
              </div>
              <div className="flex gap-4 mt-2 font-mono text-[11px] text-ink-3">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 bg-info inline-block" /> precision
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 bg-accent inline-block" /> recall
                </span>
                <span className="ml-auto">model: {m.weights ?? "-"}</span>
              </div>
            </Section>
          </>
        ) : (
          <div className="font-mono text-xs text-ink-3 py-8 text-center panel">
            {sheet?.found
              ? "gold set sealed but not measured. Run: make m9 ARGS=\"--gold " + (selected ?? "<id>") + "\""
              : "select or seal a gold set"}
          </div>
        )}
      </main>
    </div>
  );
}
