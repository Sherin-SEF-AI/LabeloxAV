"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import { invalidate } from "@/lib/useQuery";
import { Badge, Stat } from "@/components/ui/kit";
import EmptyState from "@/components/ui/EmptyState";
import { num, pct, plural } from "@/lib/format";

// Migrating in from another tool, with the cost stated before it is paid.
//
// A team switching has years of labels and a taxonomy they argued about internally. The import will remap
// their class names into this ontology, and some of that remapping loses distinctions. Doing it and then
// reporting "imported, 4,000 unmapped" invites them to find out a month later that their
// two_wheeler_with_pillion and their motorcycle both became motorcycle.
//
// So the dry run comes first and writes nothing. It parses the source, runs the same remap the real import
// would, and shows which of their classes collapse onto one of ours and which fall into a fallback bucket.
// Then importing is a decision rather than a surprise.

const FORMATS = [
  // Labelbox first because it is the most common thing to be migrating off, and the one this page could
  // not answer for until now.
  ["labelbox", "Labelbox"],
  ["scale", "Scale AI"], ["superannotate", "SuperAnnotate"], ["encord", "Encord"],
  ["cvat", "CVAT"], ["labelstudio", "Label Studio"], ["coco", "COCO"],
] as const;

type Report = {
  source_classes: number;
  objects: number;
  mapped_cleanly: number;
  into_fallback: number;
  fallback_fraction: number;
  unmapped: { source_class: string; objects: number; falls_back_to: string }[];
  merges: { ontology_class: string; source_classes: string[]; objects: number }[];
  mapping: { source_class: string; ontology_class: string; objects: number; clean: boolean }[];
};

export default function MigratePage() {
  const router = useRouter();
  const [format, setFormat] = useState<string>("labelbox");
  const [source, setSource] = useState("");
  const [vehicle, setVehicle] = useState("IMPORT-01");
  const [city, setCity] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<{ frames: number; report: Report } | null>(null);

  const dryRun = useCallback(async () => {
    if (!source.trim()) { toast("Enter a source first", "error"); return; }
    setBusy(true);
    setRes(null);
    try {
      setRes(await api.importDryRun(format, source.trim()));
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  }, [format, source]);

  const commit = useCallback(async () => {
    setBusy(true);
    try {
      const r = await api.startImport({
        format, source_uri: source.trim(), target_vehicle: vehicle, city: city || undefined,
      });
      toast("Import started", "success");
      // The import page this lands on, the jobs table and home all list the job just created.
      void invalidate("listImports", "jobs", "/api/home");
      router.push(`/import?job=${r.job_id}`);
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  }, [format, source, vehicle, city, router]);

  const rep = res?.report;
  const lossy = rep ? rep.fallback_fraction > 0.1 : false;

  return (
    <PageShell active="MIGRATE" title="Migrate from another tool"
      subtitle="See what the class mapping costs before you import anything.">
      <div className="page-wrap-narrow space-y-4">

        <section className="panel">
          <div className="panel-head"><h2 className="section-title">Source export</h2></div>
          <div className="p-4 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-[200px_minmax(0,1fr)] gap-4">
              <Field label="From">
                <select value={format} onChange={(e) => setFormat(e.target.value)}
                  className="field">
                  {FORMATS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </Field>
              <Field label="Source" hint="An s3:// URI, a zip or a directory">
                <input value={source} onChange={(e) => setSource(e.target.value)} spellCheck={false}
                  placeholder="s3://uploads/labelbox-export.ndjson"
                  onKeyDown={(e) => { if (e.key === "Enter") void dryRun(); }}
                  className="field w-full font-mono text-[12px]" />
              </Field>
              <Field label="Vehicle">
                <input value={vehicle} onChange={(e) => setVehicle(e.target.value)}
                  className="field" />
              </Field>
              <Field label="City" hint="Optional">
                <input value={city} onChange={(e) => setCity(e.target.value)} placeholder="e.g. BLR"
                  className="field sm:w-48" />
              </Field>
            </div>
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button onClick={() => void dryRun()} disabled={busy || !source.trim()}
                data-tip={!source.trim() ? "Enter a source first" : undefined}
                className="btn btn-primary">
                {busy && !rep && <span className="running-dot mr-1" />}
                {busy && !rep ? "Reading export..." : "Run a dry run"}
              </button>
              <span className="section-desc">
                Writes nothing: it parses the export and applies the same class remap the import would.
              </span>
            </div>
          </div>
        </section>

        {!rep && !busy && (
          <div className="panel">
            <EmptyState icon="flask" title="Check the mapping first">
              A dry run shows which of your classes map cleanly, which collapse together, and which fall into
              a fallback class. You decide whether to import once you have seen it.
            </EmptyState>
          </div>
        )}

        {rep && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <Stat label="Frames" value={num(res.frames)} />
              <Stat label="Their classes" value={num(rep.source_classes)} />
              <Stat label="Objects" value={num(rep.objects)} />
              <Stat label="Mapped cleanly" value={num(rep.mapped_cleanly)} tone="pass" />
              <Stat label="Into fallback" value={pct(rep.fallback_fraction, 1)}
                tone={lossy ? "block" : undefined} />
            </div>

            {lossy && (
              // Said plainly and before the import, because this is the number a customer will care about
              // after it and cannot undo.
              <div role="alert" className="panel px-4 py-3 border-l-2 border-l-block text-[12.5px] text-ink-2 leading-relaxed">
                <span className="text-ink font-medium">{num(rep.into_fallback)} of {num(rep.objects)} objects
                ({pct(rep.fallback_fraction, 1)}) would land in a fallback class</span> rather than a
                real one. Those labels survive the import and lose their meaning. Widening the ontology or
                the mapping table first is usually cheaper than re-labelling afterwards.
              </div>
            )}

            <div className="grid gap-4 md:grid-cols-2">
              <section className="panel">
                <div className="panel-head">
                  <h2 className="section-title">Distinctions that would be lost</h2>
                </div>
                <div className="p-4 space-y-2">
                  {rep.merges.length === 0 && (
                    <div className="text-[12.5px] text-pass">
                      No two of their classes collapse onto one of ours.
                    </div>
                  )}
                  {rep.merges.slice(0, 12).map((m) => (
                    <div key={m.ontology_class} className="text-[12.5px] leading-relaxed">
                      <span className="text-block font-mono text-[12px]">{m.source_classes.join(" + ")}</span>
                      <span className="text-ink-3"> all become </span>
                      <span className="text-ink font-mono text-[12px]">{m.ontology_class}</span>
                      <span className="text-ink-3"> ({plural(m.objects, "object")})</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <h2 className="section-title">Classes with nowhere to go</h2>
                </div>
                <div className="p-4 space-y-2">
                  {rep.unmapped.length === 0 && (
                    <div className="text-[12.5px] text-pass">
                      Every class maps to a real one. Nothing falls back.
                    </div>
                  )}
                  {rep.unmapped.slice(0, 12).map((u) => (
                    <div key={u.source_class} className="flex justify-between gap-3 text-[12.5px]">
                      <span className="text-warn font-mono text-[12px] truncate">{u.source_class}</span>
                      <span className="text-ink-3 shrink-0">
                        to <span className="font-mono text-[12px] text-ink-2">{u.falls_back_to}</span> · {num(u.objects)}
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            </div>

            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Full mapping</h2>
                <div className="ml-auto flex items-center gap-3">
                  {lossy && <span className="text-[12px] text-block">Some labels lose their meaning</span>}
                  <button onClick={() => void commit()} disabled={busy}
                    className={lossy ? "btn btn-danger" : "btn btn-primary"}>
                    {lossy ? "Import anyway" : "Import"}
                  </button>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr><th className="text-left">Their class</th><th className="text-left">Becomes</th><th className="num">Objects</th></tr>
                  </thead>
                  <tbody>
                    {rep.mapping.slice(0, 40).map((m) => (
                      <tr key={m.source_class}>
                        <td className="font-mono text-[12px]">{m.source_class}</td>
                        <td className={m.clean ? "text-ink" : "text-warn"}>
                          <span className="font-mono text-[12px]">{m.ontology_class}</span>
                          {!m.clean && <span className="ml-2"><Badge state="warn">Fallback</Badge></span>}
                        </td>
                        <td className="num">{num(m.objects)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {rep.mapping.length > 40 && (
                  <div className="px-3 py-2 text-[12px] text-ink-3 border-t hairline">
                    Showing 40 of {num(rep.mapping.length)} classes.
                  </div>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </PageShell>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 min-w-0">
      <span className="text-[12px] font-medium text-ink-2">
        {label}{hint && <span className="font-normal text-ink-3"> · {hint}</span>}
      </span>
      {children}
    </label>
  );
}
