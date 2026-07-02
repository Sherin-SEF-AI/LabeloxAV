"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type ImportJob } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { StateBadge, ConfBar } from "@/components/StateBadge";

// Upload a dataset (any size) straight to storage, then import it. The file goes browser -> MinIO via
// presigned multipart; the API only signs and runs the import as a background job. PII (Gate A) runs
// on every imported frame; objects land in triage as source=imported / state=review.

const FORMATS = [
  "coco",
  "yolo",
  "pascalvoc",
  "openlabel",
  "nuscenes",
  "parquet",
  "mapillary",
  "images",
  "video",
  "mcap",
];

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

export default function ImportPage() {
  const router = useRouter();
  const [format, setFormat] = useState("coco");
  const [vehicle, setVehicle] = useState("IMPORT-01");
  const [city, setCity] = useState("BLR");
  const [files, setFiles] = useState<File[]>([]);
  const [drag, setDrag] = useState(false);
  const [phase, setPhase] = useState<"idle" | "uploading" | "importing">("idle");
  const [uploadFrac, setUploadFrac] = useState(0);
  const [batchIdx, setBatchIdx] = useState(0);
  const [jobs, setJobs] = useState<ImportJob[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      setJobs(await api.listImports());
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, []);

  async function onGo() {
    if (!files.length) return;
    setErr(null);
    try {
      // Upload + start an import per file, so a folder of videos becomes one session each.
      for (let i = 0; i < files.length; i++) {
        setBatchIdx(i);
        setPhase("uploading");
        setUploadFrac(0);
        const uri = await api.uploadMultipart(files[i], setUploadFrac);
        setPhase("importing");
        await api.startImport({ format, source_uri: uri, target_vehicle: vehicle, city });
        refresh();
      }
      setPhase("idle");
      setFiles([]);
      refresh();
    } catch (e) {
      setErr(String(e));
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";

  return (
    <PageShell active="IMPORT" title="Import Dataset">
      <div className="p-4 space-y-4 max-w-4xl w-full mx-auto">
        <Section title="upload a dataset (any format, any size)">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              if (e.dataTransfer.files.length) setFiles(Array.from(e.dataTransfer.files));
            }}
            onClick={() => inputRef.current?.click()}
            className={`border border-dashed ${
              drag ? "border-accent" : "border-line"
            } p-8 text-center cursor-pointer font-mono text-xs text-ink-3`}
          >
            <input
              ref={inputRef}
              type="file"
              multiple
              className="hidden"
              onChange={(e) => e.target.files?.length && setFiles(Array.from(e.target.files))}
            />
            {files.length === 1 ? (
              <span className="text-ink">
                {files[0].name}{" "}
                <span className="text-ink-3">({(files[0].size / 1e6).toFixed(1)} MB)</span>
              </span>
            ) : files.length > 1 ? (
              <span className="text-ink">
                {files.length} files{" "}
                <span className="text-ink-3">
                  ({(files.reduce((s, f) => s + f.size, 0) / 1e6).toFixed(0)} MB total)
                </span>
                <span className="block text-ink-3 text-[10px] mt-1 truncate">
                  {files.slice(0, 4).map((f) => f.name).join(", ")}{files.length > 4 ? ` +${files.length - 4} more` : ""}
                </span>
              </span>
            ) : (
              "drag .zip / .mp4 / .mcap / .parquet here (multiple allowed), or click to browse"
            )}
          </div>

          <div className="grid grid-cols-3 gap-3 mt-3 font-mono text-xs">
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">format</span>
              <select
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                className="bg-panel border border-line px-2 py-1 text-ink"
              >
                {FORMATS.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">vehicle</span>
              <input
                value={vehicle}
                onChange={(e) => setVehicle(e.target.value)}
                className="bg-panel border border-line px-2 py-1 text-ink"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">city</span>
              <input
                value={city}
                onChange={(e) => setCity(e.target.value)}
                className="bg-panel border border-line px-2 py-1 text-ink"
              />
            </label>
          </div>

          <div className="flex items-center gap-3 mt-3">
            <button
              onClick={onGo}
              disabled={!files.length || busy}
              className="font-mono text-xs border border-line px-3 py-1 hover:border-accent disabled:opacity-50"
            >
              {phase === "uploading"
                ? `uploading ${files.length > 1 ? `${batchIdx + 1}/${files.length} · ` : ""}${(uploadFrac * 100).toFixed(0)}%`
                : phase === "importing"
                  ? `starting import${files.length > 1 ? ` ${batchIdx + 1}/${files.length}` : ""}...`
                  : `upload + import${files.length > 1 ? ` (${files.length})` : ""}`}
            </button>
            {phase === "uploading" && (
              <div className="flex-1 h-2 bg-line relative">
                <div
                  className="absolute left-0 top-0 h-full bg-accent"
                  style={{ width: `${uploadFrac * 100}%` }}
                />
              </div>
            )}
            {err && <span className="font-mono text-[11px] text-block">{err}</span>}
          </div>
          <div className="font-mono text-[11px] text-ink-3 mt-2">
            PII (faces/plates) is blurred on every imported frame before storage. Objects arrive in
            triage as imported / review.
          </div>
        </Section>

        <Section title={`import jobs (${jobs.length})`}>
          {jobs.length ? (
            <div className="space-y-1">
              {jobs.map((j) => (
                <div key={j.job_id} className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="w-16 truncate text-ink-2">{j.format}</span>
                  <StateBadge state={j.status} />
                  <div className="flex-1 flex justify-center">
                    <ConfBar conf={j.progress || 0} />
                  </div>
                  <span className="w-40 text-right text-ink-3 truncate">
                    {j.counts?.frames ?? 0}fr / {j.counts?.objects ?? 0}obj
                    {j.error ? ` · ${j.error.slice(0, 30)}` : ""}
                  </span>
                  {j.session_id && (
                    <button
                      onClick={() => router.push(`/?session=${j.session_id}`)}
                      className="border border-line px-1.5 hover:border-accent"
                    >
                      view
                    </button>
                  )}
                  {j.session_id && (
                    <button
                      onClick={async () => {
                        try {
                          const r = await api.inspectorLichtblick(j.session_id!);
                          window.open(r.url, "_blank", "noopener");
                        } catch (e) {
                          setErr("Open in Lichtblick failed (no MCAP for this session, or Lichtblick is not running): " + String(e));
                        }
                      }}
                      title="full-power MCAP inspection in the self-hosted Lichtblick"
                      className="border border-line px-1.5 hover:border-accent"
                    >
                      lichtblick
                    </button>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">no imports yet</div>
          )}
        </Section>
      </div>
    </PageShell>
  );
}
