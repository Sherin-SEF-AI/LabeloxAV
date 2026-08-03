"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";

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
    if (!source.trim()) { toast("point at a source first", "error"); return; }
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
      toast("import started", "success");
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
      subtitle="see what the taxonomy costs before importing it">
      <div className="p-4 space-y-3 max-w-6xl">

        <div className="panel px-3 py-3 space-y-2">
          <div className="flex flex-wrap items-end gap-2">
            <Field label="from">
              <select value={format} onChange={(e) => setFormat(e.target.value)}
                className="input font-mono text-[11px]">
                {FORMATS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </Field>
            <Field label="source (s3:// uri, zip or directory)">
              <input value={source} onChange={(e) => setSource(e.target.value)} spellCheck={false}
                placeholder="s3://uploads/labelbox-export.ndjson"
                className="input font-mono text-[11px] w-[26rem]" />
            </Field>
            <Field label="vehicle">
              <input value={vehicle} onChange={(e) => setVehicle(e.target.value)}
                className="input font-mono text-[11px] w-32" />
            </Field>
            <Field label="city">
              <input value={city} onChange={(e) => setCity(e.target.value)} placeholder="optional"
                className="input font-mono text-[11px] w-28" />
            </Field>
            <button onClick={() => void dryRun()} disabled={busy}
              className="border border-accent text-accent px-3 py-1 hover:bg-accent/10 disabled:opacity-30 font-mono text-[11px]">
              {busy ? "reading..." : "dry run"}
            </button>
          </div>
          <div className="font-mono text-[10px] text-ink-3">
            A dry run writes nothing. It parses the export and applies the same remap the import would.
          </div>
        </div>

        {rep && (
          <>
            <div className="flex flex-wrap gap-2 font-mono text-[11px]">
              <Stat label="frames" value={res.frames.toLocaleString()} />
              <Stat label="their classes" value={String(rep.source_classes)} />
              <Stat label="objects" value={rep.objects.toLocaleString()} />
              <Stat label="mapped cleanly" value={rep.mapped_cleanly.toLocaleString()} tone="text-accent" />
              <Stat label="into fallback" value={`${(rep.fallback_fraction * 100).toFixed(1)}%`}
                tone={lossy ? "text-block" : "text-ink"} />
            </div>

            {lossy && (
              // Said plainly and before the import, because this is the number a customer will care about
              // after it and cannot undo.
              <div className="panel px-3 py-2 border-l-2 border-block font-mono text-[11px] text-ink-2">
                {rep.into_fallback.toLocaleString()} of {rep.objects.toLocaleString()} objects
                ({(rep.fallback_fraction * 100).toFixed(1)}%) would land in a fallback class rather than a
                real one. Those labels survive the import and lose their meaning. Widening the ontology or
                the mapping table first is usually cheaper than re-labelling afterwards.
              </div>
            )}

            <div className="grid gap-3 md:grid-cols-2">
              <section className="panel">
                <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                  distinctions that would be lost
                </div>
                <div className="p-3 space-y-1">
                  {rep.merges.length === 0 && (
                    <div className="font-mono text-[10px] text-ink-3">
                      No two of their classes collapse onto one of ours.
                    </div>
                  )}
                  {rep.merges.slice(0, 12).map((m) => (
                    <div key={m.ontology_class} className="font-mono text-[10px]">
                      <span className="text-block">{m.source_classes.join(" + ")}</span>
                      <span className="text-ink-3"> all become </span>
                      <span className="text-ink">{m.ontology_class}</span>
                      <span className="text-ink-3"> ({m.objects.toLocaleString()} objects)</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="panel">
                <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                  classes with nowhere to go
                </div>
                <div className="p-3 space-y-1">
                  {rep.unmapped.length === 0 && (
                    <div className="font-mono text-[10px] text-accent">
                      Every class maps to a real one. Nothing falls back.
                    </div>
                  )}
                  {rep.unmapped.slice(0, 12).map((u) => (
                    <div key={u.source_class} className="flex justify-between font-mono text-[10px]">
                      <span className="text-warn truncate">{u.source_class}</span>
                      <span className="text-ink-3">
                        {u.falls_back_to} · {u.objects.toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            </div>

            <section className="panel">
              <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2 flex items-center justify-between">
                <span>the full mapping</span>
                <button onClick={() => void commit()} disabled={busy}
                  className="border border-accent text-accent px-2 py-0.5 hover:bg-accent/10 disabled:opacity-30">
                  import anyway
                </button>
              </div>
              <div className="p-3 overflow-x-auto">
                <table className="w-full font-mono text-[10px]">
                  <thead className="text-ink-3 text-left">
                    <tr><th className="py-1">their class</th><th>becomes</th><th className="text-right">objects</th></tr>
                  </thead>
                  <tbody>
                    {rep.mapping.slice(0, 40).map((m) => (
                      <tr key={m.source_class} className="border-t border-line">
                        <td className="py-0.5">{m.source_class}</td>
                        <td className={m.clean ? "text-ink" : "text-warn"}>
                          {m.ontology_class}{m.clean ? "" : "  (fallback)"}
                        </td>
                        <td className="text-right tabular-nums text-ink-3">{m.objects.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </div>
    </PageShell>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="font-mono text-[10px] uppercase text-ink-3">{label}</span>
      {children}
    </label>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="panel px-3 py-2 min-w-[110px]">
      <div className="text-[10px] uppercase text-ink-3">{label}</div>
      <div className={`text-[18px] tabular-nums ${tone ?? "text-ink"}`}>{value}</div>
    </div>
  );
}
