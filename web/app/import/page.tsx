"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type ImportJob } from "@/lib/api";
import TopNav from "@/components/TopNav";

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
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [phase, setPhase] = useState<"idle" | "uploading" | "importing">("idle");
  const [uploadFrac, setUploadFrac] = useState(0);
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
    if (!file) return;
    setErr(null);
    try {
      setPhase("uploading");
      setUploadFrac(0);
      const uri = await api.uploadMultipart(file, setUploadFrac);
      setPhase("importing");
      await api.startImport({ format, source_uri: uri, target_vehicle: vehicle, city });
      setPhase("idle");
      setFile(null);
      refresh();
    } catch (e) {
      setErr(String(e));
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="IMPORT" />

      <main className="flex-1 overflow-auto p-4 space-y-4 max-w-4xl w-full mx-auto">
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
              if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
            }}
            onClick={() => inputRef.current?.click()}
            className={`border border-dashed ${
              drag ? "border-accent" : "border-line"
            } p-8 text-center cursor-pointer font-mono text-xs text-ink-3`}
          >
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])}
            />
            {file ? (
              <span className="text-ink">
                {file.name}{" "}
                <span className="text-ink-3">({(file.size / 1e6).toFixed(1)} MB)</span>
              </span>
            ) : (
              "drag a .zip / .mp4 / .mcap / .parquet here, or click to browse"
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
              disabled={!file || busy}
              className="font-mono text-xs border border-line px-3 py-1 hover:border-accent disabled:opacity-50"
            >
              {phase === "uploading"
                ? `uploading ${(uploadFrac * 100).toFixed(0)}%`
                : phase === "importing"
                  ? "starting import..."
                  : "upload + import"}
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
                  <span
                    className={`w-16 ${
                      j.status === "done"
                        ? "text-pass"
                        : j.status === "error"
                          ? "text-block"
                          : "text-warn"
                    }`}
                  >
                    {j.status}
                  </span>
                  <div className="flex-1 h-2 bg-line relative">
                    <div
                      className="absolute left-0 top-0 h-full bg-accent"
                      style={{ width: `${(j.progress || 0) * 100}%` }}
                    />
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
                </div>
              ))}
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">no imports yet</div>
          )}
        </Section>
      </main>
    </div>
  );
}
