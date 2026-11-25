"use client";

import { Suspense, useEffect, useState } from "react";
import { useJobStream } from "@/lib/useEventStream";
import { api , humanizeError } from "@/lib/api";
import type { DatasetDetail, DatasetRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { ago, num, plural } from "@/lib/format";
import { useQueryParam } from "@/lib/useQueryParam";
import { invalidate, useQuery } from "@/lib/useQuery";

// Datasets + delivery: seal a versioned dataset (a background export job) and download the formats.
// This is the "product out" surface - how labeled data leaves the engine.

const FORMATS = ["coco", "yolo", "parquet", "openlabel", "nuscenes"];
// How each format is written. The value sent to the export is the code.
const FORMAT_NAMES: Record<string, string> = {
  coco: "COCO", yolo: "YOLO", parquet: "Parquet", openlabel: "OpenLABEL", nuscenes: "nuScenes",
};
const fmtName = (f: string) => FORMAT_NAMES[f] ?? f;

function Section({ title, desc, right, flush = false, children }: {
  title: string; desc?: string; right?: React.ReactNode; flush?: boolean; children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">{title}</h2>
        {right && <span className="ml-auto flex items-center gap-2">{right}</span>}
      </div>
      <div className={flush ? "" : "p-4"}>
        {desc && <p className="section-desc mb-4">{desc}</p>}
        {children}
      </div>
    </section>
  );
}

function FieldLabel({ children, hint }: { children: React.ReactNode; hint?: string }) {
  return (
    <span className="text-[12px] font-medium text-ink-2">
      {children}{hint && <span className="font-normal text-ink-3"> · {hint}</span>}
    </span>
  );
}

export default function DatasetsPage() {
  // useSearchParams (via useQueryParam) forces a CSR bailout Next requires be under Suspense.
  return <Suspense fallback={null}><DatasetsBody /></Suspense>;
}

function DatasetsBody() {
  // A failed list says so, with a retry, rather than reading as "none sealed yet".
  const list = useQuery(["datasets"], () => api.datasets());
  const rows: DatasetRow[] = list.data ?? [];
  const refresh = list.reload;
  const [open, setOpen] = useState<string | null>(null);
  const opened = useQuery(open ? ["dataset", open] : null, () => api.dataset(open as string));
  const detail: DatasetDetail | null = opened.error ? null : opened.data ?? null;
  const [msg, setMsg] = useState<string | null>(null);

  // export form
  const [name, setName] = useState("delivery-v1");
  const [states, setStates] = useState("accepted");
  const qsFormat = useQueryParam("format");
  // the File > Export menu deep-links a target format; preselect it alongside the lossless sidecar
  const [fmts, setFmts] = useState<string[]>(
    qsFormat ? [qsFormat, "parquet"] : ["coco", "parquet"]);
  // Zero is no split, which is what every export did before this existed. A delivered dataset with no
  // split gets split by whoever receives it, and the obvious way to do that (per frame) is the wrong way:
  // consecutive dashcam frames are near-duplicates, so it puts the same road on both sides of the line.
  const [valFrac, setValFrac] = useState(0);
  const [testFrac, setTestFrac] = useState(0);

  // Export jobs are on the shared stream, so a sealing dataset updates as it seals rather than on a timer.
  const jobStream = useJobStream();
  useEffect(() => { if (jobStream.data) void refresh(); }, [jobStream.data, refresh]);

  async function runExport() {
    try {
      await api.startExport({
        name, states: states.split(",").map((s) => s.trim()).filter(Boolean), formats: fmts,
        val_frac: valFrac, test_frac: testFrac, split_group_by: "session",
      });
      setMsg("Export queued. Follow it on Jobs; it appears here once sealed.");
      void refresh();
      // The export is a job: the jobs table and home's running count both change.
      void invalidate("jobs", "/api/home");
    } catch (e) {
      setMsg(humanizeError(e));
    }
  }

  const toggleFmt = (f: string) => setFmts((s) => (s.includes(f) ? s.filter((x) => x !== f) : [...s, f]));

  const queued = msg?.startsWith("Export queued");

  return (
    <PageShell
      active="DATASETS"
      title="Sealed Datasets"
      subtitle="Seal a versioned snapshot of your labels and download it in the formats you need."
      meta={<span className="text-[12px] text-ink-3">{plural(rows.length, "dataset")} sealed</span>}
    >
      <div className="page-wrap-narrow space-y-4">
        <Section title="New export">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-[12.5px]">
            <label className="flex flex-col gap-1.5">
              <FieldLabel>Name</FieldLabel>
              <input value={name} onChange={(e) => setName(e.target.value)} className="field" />
            </label>
            <label className="flex flex-col gap-1.5">
              <FieldLabel hint="comma separated">Object states to include</FieldLabel>
              <input value={states} onChange={(e) => setStates(e.target.value)} placeholder="accepted" className="field font-mono text-[12px]" />
            </label>
          </div>

          <div className="mt-4 flex flex-col gap-1.5">
            <FieldLabel>Formats</FieldLabel>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Export formats">
              {FORMATS.map((f) => (
                <button key={f} type="button" onClick={() => toggleFmt(f)} aria-pressed={fmts.includes(f)}
                  className={`chip ${fmts.includes(f) ? "chip-on" : ""}`}>
                  {fmts.includes(f) && <span aria-hidden>✓</span>}
                  {fmtName(f)}
                </button>
              ))}
            </div>
            <span className="text-[12px] text-ink-3">A lossless Parquet sidecar is always included.</span>
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-4">
            {/* Split at the session boundary. A session is one drive, so holding out whole sessions is
                what keeps the same road out of both train and val. */}
            <label className="flex flex-col gap-1.5">
              <FieldLabel>Validation %</FieldLabel>
              <input type="number" min={0} max={90} step={5} value={Math.round(valFrac * 100)}
                onChange={(e) => setValFrac(Math.max(0, Math.min(90, Number(e.target.value))) / 100)}
                className="field w-24" />
            </label>
            <label className="flex flex-col gap-1.5">
              <FieldLabel>Test %</FieldLabel>
              <input type="number" min={0} max={90} step={5} value={Math.round(testFrac * 100)}
                onChange={(e) => setTestFrac(Math.max(0, Math.min(90, Number(e.target.value))) / 100)}
                className="field w-24" />
            </label>
            <span className="text-[12px] text-ink-3 pb-1.5 max-w-sm">
              {valFrac || testFrac
                ? "Held out by session, so no drive appears in two splits."
                : "No split: the whole slice is delivered as train."}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3 mt-5 pt-4 border-t hairline">
            <button onClick={runExport} disabled={!fmts.length}
              data-tip={!fmts.length ? "Pick at least one format" : undefined}
              className="btn btn-primary">
              Seal and export
            </button>
            {msg && (
              <span className={`text-[12.5px] ${queued ? "text-pass" : "text-block"}`} role="status">
                {msg}
                {queued && <> <a href="/jobs" className="link">Open Jobs →</a></>}
              </span>
            )}
          </div>
        </Section>

        <Section title="Sealed datasets" flush
          right={rows.length ? <span className="text-[12px] text-ink-3 tabular-nums">{num(rows.length)}</span> : undefined}>
          <LoadState {...list.loadState} rows={4}
            empty={!rows.length}
            emptyIcon="layers"
            emptyTitle="No datasets sealed yet"
            emptyText="A sealed dataset is a fixed, versioned copy of your labels. Seal one above and its files appear here to download.">
            <ul>
              {rows.map((d) => {
                const isOpen = open === d.commit_id;
                return (
                <li key={d.commit_id} className="border-b hairline last:border-0">
                  <button onClick={() => setOpen(isOpen ? null : d.commit_id)} aria-expanded={isOpen}
                    className="w-full flex items-center gap-4 px-4 py-2.5 text-[12.5px] text-left hover:bg-head/50 transition-colors">
                    <span className="text-ink-3 w-3 text-[11px]" aria-hidden>{isOpen ? "▾" : "▸"}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-ink font-medium truncate">{d.name ?? "Untitled"}</span>
                      <span className="block text-[11.5px] text-ink-3 truncate">
                        <span className="font-mono">{d.commit_id.slice(0, 12)}</span>
                        {d.created_at && <> · {ago(d.created_at)}</>}
                      </span>
                    </span>
                    <span className="text-ink-2 tabular-nums shrink-0">{plural(d.object_count, "object")}</span>
                    <span className="hidden sm:flex gap-1 shrink-0">
                      {(d.formats || []).map((f) => (
                        <span key={f} className="px-1.5 py-px rounded border border-line text-[11px] text-ink-2">{fmtName(f)}</span>
                      ))}
                    </span>
                    <span className="text-ink-3 tabular-nums shrink-0 w-16 text-right">{plural(d.n_files, "file")}</span>
                  </button>
                  {isOpen && (
                    <div className="px-4 pb-3 pl-11">
                      {opened.loading ? (
                        <div className="space-y-2 py-1"><span className="skeleton h-3 w-1/2 block" /><span className="skeleton h-3 w-1/3 block" /></div>
                      ) : opened.error ? (
                        <div className="text-[12px] text-block">Could not list the files: {humanizeError(opened.error)}</div>
                      ) : detail && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1 text-[12px]">
                          {detail.files.map((f) => (
                            f.url ? (
                              <a key={f.path} href={f.url} className="link truncate inline-flex items-center gap-1.5" title={f.path} download>
                                <span aria-hidden>↓</span>{f.path.split("/").slice(-2).join("/")}
                              </a>
                            ) : (
                              <span key={f.path} className="text-ink-3 truncate" title={`${f.path} (file not found in storage)`}>
                                {f.path.split("/").slice(-2).join("/")} <span className="text-block">(missing)</span>
                              </span>
                            )
                          ))}
                          {!detail.files.length && <span className="text-ink-3">This dataset has no files.</span>}
                        </div>
                      )}
                    </div>
                  )}
                </li>
                );
              })}
            </ul>
          </LoadState>
        </Section>
      </div>
    </PageShell>
  );
}
