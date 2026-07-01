"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { DatasetDetail, DatasetRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";

// Datasets + delivery: seal a versioned dataset (a background export job) and download the formats.
// This is the "product out" surface - how labeled data leaves the engine.

const FORMATS = ["coco", "yolo", "parquet", "openlabel", "nuscenes"];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">{title}</div>
      <div className="p-3">{children}</div>
    </section>
  );
}

export default function DatasetsPage() {
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  // export form
  const [name, setName] = useState("delivery-v1");
  const [states, setStates] = useState("accepted");
  const [fmts, setFmts] = useState<string[]>(["coco", "parquet"]);

  async function refresh() {
    try {
      setRows(await api.datasets());
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (open) api.dataset(open).then(setDetail).catch(() => setDetail(null));
    else setDetail(null);
  }, [open]);

  async function runExport() {
    try {
      await api.startExport({ name, states: states.split(",").map((s) => s.trim()).filter(Boolean), formats: fmts });
      setMsg("export queued - watch it on Jobs, it appears here when sealed");
      refresh();
    } catch (e) {
      setMsg(String(e));
    }
  }

  const toggleFmt = (f: string) => setFmts((s) => (s.includes(f) ? s.filter((x) => x !== f) : [...s, f]));

  return (
    <PageShell
      active="DATASETS"
      title="Sealed Datasets"
      right={<span className="text-ink-3">{rows.length} sealed</span>}
    >
      <div className="p-4 space-y-4 max-w-4xl w-full mx-auto">
        <Section title="new export (seal a versioned dataset)">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 font-mono text-xs">
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">name</span>
              <input value={name} onChange={(e) => setName(e.target.value)} className="bg-panel border border-line px-2 py-1 text-ink" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">states (comma)</span>
              <input value={states} onChange={(e) => setStates(e.target.value)} placeholder="accepted" className="bg-panel border border-line px-2 py-1 text-ink" />
            </label>
            <div className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">formats</span>
              <div className="flex flex-wrap gap-1">
                {FORMATS.map((f) => (
                  <button key={f} onClick={() => toggleFmt(f)}
                    className={`px-1.5 py-0.5 border ${fmts.includes(f) ? "border-accent text-ink" : "border-line text-ink-3"}`}>
                    {f}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3 mt-3">
            <button onClick={runExport} disabled={!fmts.length}
              className="font-mono text-xs border border-pass text-pass px-3 py-1 hover:bg-pass/10 disabled:opacity-50">
              export
            </button>
            {msg && <span className="font-mono text-[11px] text-warn">{msg}</span>}
            <span className="font-mono text-[10px] text-ink-3 ml-auto">parquet sidecar is always included (lossless)</span>
          </div>
        </Section>

        <Section title={`sealed datasets (${rows.length})`}>
          {rows.length ? (
            <div className="space-y-1">
              {rows.map((d) => (
                <div key={d.commit_id} className="border-b hairline pb-1">
                  <button onClick={() => setOpen(open === d.commit_id ? null : d.commit_id)}
                    className="w-full flex items-center gap-3 px-1 py-1 font-mono text-[11px] text-left hover:text-ink">
                    <span className="text-ink-2 w-40 truncate">{d.name ?? "?"}</span>
                    <span className="text-ink-3">{d.commit_id.slice(0, 14)}</span>
                    <span className="text-ink-3">{d.object_count} obj</span>
                    <span className="text-ink-3">{(d.formats || []).join(", ")}</span>
                    <span className="ml-auto text-ink-3">{d.n_files} files {open === d.commit_id ? "▾" : "▸"}</span>
                  </button>
                  {open === d.commit_id && detail && (
                    <div className="pl-2 pb-2 grid grid-cols-2 md:grid-cols-3 gap-x-4 gap-y-0.5 font-mono text-[10px]">
                      {detail.files.map((f) => (
                        f.url ? (
                          <a key={f.path} href={f.url} className="text-info hover:text-accent truncate" title={f.path} download>
                            {f.path.split("/").slice(-2).join("/")}
                          </a>
                        ) : (
                          <span key={f.path} className="text-ink-3 truncate" title={`${f.path} (file not found in storage)`}>
                            {f.path.split("/").slice(-2).join("/")} <span className="text-block">(missing)</span>
                          </span>
                        )
                      ))}
                      {!detail.files.length && <span className="text-ink-3">no files</span>}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">no datasets sealed yet</div>
          )}
        </Section>
      </div>
    </PageShell>
  );
}
