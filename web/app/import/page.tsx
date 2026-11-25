"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useJobStream } from "@/lib/useEventStream";
import { api, type ImportJob , humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { StateBadge } from "@/components/StateBadge";
import { Badge } from "@/components/ui/kit";
import EmptyState from "@/components/ui/EmptyState";
import LoadState from "@/components/shell/LoadState";
import Icon from "@/components/shell/Icon";
import { ago, humanize, num, pct } from "@/lib/format";
import { useQueryParam } from "@/lib/useQueryParam";
import { invalidate, useQuery } from "@/lib/useQuery";
import { pressable } from "@/components/ui/pressable";

// Upload a dataset (any size) straight to storage, then import it. The file goes browser -> MinIO via
// presigned multipart; the API only signs and runs the import as a background job. PII (Gate A) runs
// on every imported frame; objects land in triage as source=imported / state=review.

// The starting list, replaced by the server's the moment it answers. It exists only so the select is not
// empty on the first paint; it is deliberately not the source of truth. A hardcoded list here is what let
// this page offer ten formats while the importer registered nineteen, and silently rewrite the other nine
// to COCO: deep-linking File > Import > CVAT XML landed you on the COCO importer with no message.
const FALLBACK_FORMATS = ["coco", "yolo", "pascalvoc", "images", "video", "mcap"];

function Section({ title, right, children, flush = false }: {
  title: string; right?: React.ReactNode; children: React.ReactNode; flush?: boolean;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">{title}</h2>
        {right && <span className="ml-auto flex items-center gap-2">{right}</span>}
      </div>
      <div className={flush ? "" : "p-4"}>{children}</div>
    </section>
  );
}

// Format codes as people say them. Only the label changes; the value sent to the importer is the code.
const FORMAT_NAMES: Record<string, string> = {
  coco: "COCO", yolo: "YOLO", pascalvoc: "Pascal VOC", mcap: "MCAP", cvat: "CVAT", cvat_xml: "CVAT XML",
  kitti: "KITTI", nuscenes: "nuScenes", waymo: "Waymo", bdd: "BDD100K", bdd100k: "BDD100K",
  labelme: "LabelMe", labelbox: "Labelbox", rosbag: "ROS bag", parquet: "Parquet", openlabel: "OpenLABEL",
  images: "Images", video: "Video", cityscapes: "Cityscapes", argoverse: "Argoverse", av2: "Argoverse 2",
  scale: "Scale AI", supervisely: "Supervisely", datumaro: "Datumaro", tfrecord: "TFRecord",
};
function formatName(code: string): string {
  return FORMAT_NAMES[code.toLowerCase()] ?? humanize(code);
}

// The file a job was made from, for a row that otherwise only had a format and a uuid to go by.
function sourceName(uri: string | null): string | null {
  if (!uri) return null;
  const base = uri.split(/[\\/]/).filter(Boolean).pop();
  if (!base) return null;
  try { return decodeURIComponent(base); } catch { return base; }
}

const HEALTH: Record<string, { label: string; state: "pass" | "warn" | "block" }> = {
  pass: { label: "Healthy", state: "pass" },
  warn: { label: "Warnings", state: "warn" },
  fail: { label: "Failed checks", state: "block" },
};

export default function ImportPage() {
  // useSearchParams (via useQueryParam) forces a CSR bailout Next requires be under Suspense.
  return <Suspense fallback={null}><ImportBody /></Suspense>;
}

function ImportBody() {
  const router = useRouter();
  const qsFormat = useQueryParam("format");
  // What the importer registers. Until it answers (or if it does not) the fallback list stands, as before.
  const formatsQ = useQuery(["importFormats"], () => api.importFormats());
  const formats: string[] = formatsQ.data?.formats ?? FALLBACK_FORMATS;
  const rawFormats: string[] = formatsQ.data?.raw ?? [];
  // The File > Import menu deep-links a format per entry. Honoured as given: whether the importer knows it
  // is the server's question, and answering it from a stale local list is what silently sent nine formats
  // to the wrong importer.
  const [format, setFormat] = useState(qsFormat || "coco");
  const [unknownFormat, setUnknownFormat] = useState<string | null>(null);

  useEffect(() => {
    // Said out loud rather than corrected behind the user's back.
    if (qsFormat && formatsQ.data && !formatsQ.data.formats.includes(qsFormat)) setUnknownFormat(qsFormat);
  }, [qsFormat, formatsQ.data]);
  const [vehicle, setVehicle] = useState("IMPORT-01");
  const [city, setCity] = useState("BLR");
  const [files, setFiles] = useState<File[]>([]);
  const [drag, setDrag] = useState(false);
  const [phase, setPhase] = useState<"idle" | "uploading" | "importing">("idle");
  const [uploadFrac, setUploadFrac] = useState(0);
  const [batchIdx, setBatchIdx] = useState(0);
  // A failed list says so (LoadState below) rather than reading as "no imports yet".
  const jobsQ = useQuery(["listImports"], () => api.listImports());
  const jobs: ImportJob[] = useMemo(() => jobsQ.data ?? [], [jobsQ.data]);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Inspector health verdict per session: the chip that answers "did my recording work" before opening a
  // panel. One read per imported session, after the list; the chips on screen stay while they refresh.
  const sids = useMemo(
    () => Array.from(new Set(jobs.map((j) => j.session_id).filter(Boolean))) as string[], [jobs]);
  const healthQ = useQuery(jobsQ.data ? ["inspectorHealth", sids] : null, async () => {
    const pairs = await Promise.all(sids.map(async (sid) => {
      try { return [sid, (await api.inspectorHealth(sid)).verdict] as const; } catch { return [sid, null] as const; }
    }));
    return Object.fromEntries(pairs) as Record<string, "pass" | "warn" | "fail" | null>;
  }, { keepPrevious: true });
  const health: Record<string, "pass" | "warn" | "fail" | null> = healthQ.data ?? {};
  const reloadJobs = jobsQ.reload;
  const reloadHealth = healthQ.reload;
  const refresh = useCallback(() => { void reloadJobs(); void reloadHealth(); }, [reloadJobs, reloadHealth]);

  // The import job list arrives on the shared stream; refreshing on each push replaces the two-second
  // poll that ran on every open tab regardless of whether anything was importing.
  const jobStream = useJobStream();
  useEffect(() => { if (jobStream.data) refresh(); }, [jobStream.data, refresh]);

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
      // Each import is a job and will become a session: the jobs table, home and the session list change.
      void invalidate("jobs", "/api/home", "sessionsPage", "sessions");
    } catch (e) {
      setErr(humanizeError(e));
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";

  return (
    <PageShell active="IMPORT" title="Import Dataset"
      subtitle="Bring in labelled datasets or raw recordings; each file becomes its own session.">
      <div className="page-wrap space-y-4">
        <Section title="Upload a dataset">
          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px] gap-5">
          <div
            {...pressable(() => inputRef.current?.click())}
            aria-label="Choose dataset files, or drop them here"
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
            className={`flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-lg px-6 py-10 text-center cursor-pointer transition-colors ${
              drag ? "border-accent bg-accent/5" : "border-line bg-bg-2/40 hover:border-accent/60"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              multiple
              className="hidden"
              onChange={(e) => e.target.files?.length && setFiles(Array.from(e.target.files))}
            />
            <span className="flex items-center justify-center w-11 h-11 rounded-full border border-line bg-head text-ink-2 mb-1" aria-hidden>
              <Icon name="upload" size={20} />
            </span>
            {files.length === 1 ? (
              <span className="text-[14px] font-medium text-ink break-all">
                {files[0].name}{" "}
                <span className="text-ink-3 font-normal">({(files[0].size / 1e6).toFixed(1)} MB)</span>
              </span>
            ) : files.length > 1 ? (
              <span className="text-[14px] font-medium text-ink">
                {files.length} files{" "}
                <span className="text-ink-3 font-normal">
                  ({(files.reduce((s, f) => s + f.size, 0) / 1e6).toFixed(0)} MB total)
                </span>
                <span className="block text-ink-3 text-[12px] font-normal mt-1 truncate max-w-md">
                  {files.slice(0, 4).map((f) => f.name).join(", ")}{files.length > 4 ? ` +${files.length - 4} more` : ""}
                </span>
              </span>
            ) : (
              <>
                <span className="text-[14.5px] font-medium text-ink">Drop dataset files here</span>
                <span className="text-[12.5px] text-ink-3">or <span className="text-accent-2">click to browse</span>. Several at once is fine.</span>
                <span className="text-[12px] text-ink-3 mt-1">.zip, .mp4, .mcap, .parquet and more, any size</span>
              </>
            )}
          </div>

          <div className="flex flex-col gap-3 text-[12.5px]">
            <label className="flex flex-col gap-1.5">
              <span className="text-[12px] font-medium text-ink-2">Format</span>
              <select
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                className="field"
              >
                {formats.map((f) => (
                  <option key={f} value={f}>
                    {formatName(f)}{rawFormats.includes(f) ? " (media only)" : ""}
                  </option>
                ))}
              </select>
              {/* A deep link the importer does not recognise used to be rewritten to COCO in silence, so
                  the run started, read nothing it understood, and reported success. */}
              {unknownFormat && (
                <span className="text-block text-[12px]">
                  The importer does not know the format &quot;{unknownFormat}&quot;; pick one above.
                </span>
              )}
            </label>
            <div className="grid grid-cols-2 gap-3">
              <label className="flex flex-col gap-1.5">
                <span className="text-[12px] font-medium text-ink-2">Vehicle</span>
                <input
                  value={vehicle}
                  onChange={(e) => setVehicle(e.target.value)}
                  className="field"
                />
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="text-[12px] font-medium text-ink-2">City</span>
                <input
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  className="field"
                />
              </label>
            </div>
            <button
              onClick={onGo}
              disabled={!files.length || busy}
              className="btn btn-primary w-full mt-1"
              data-tip={!files.length ? "Add files first" : undefined}
            >
              {busy && <span className="running-dot mr-1" />}
              {phase === "uploading"
                ? `Uploading ${files.length > 1 ? `${batchIdx + 1} of ${files.length} · ` : ""}${pct(uploadFrac)}`
                : phase === "importing"
                  ? `Starting import${files.length > 1 ? ` ${batchIdx + 1} of ${files.length}` : ""}...`
                  : `Upload and import${files.length > 1 ? ` ${files.length} files` : ""}`}
            </button>
            {phase === "uploading" && (
              <div className="h-1.5 bg-bg-2 rounded-full overflow-hidden" role="progressbar"
                aria-valuenow={Math.round(uploadFrac * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="upload">
                <div
                  className="h-full bg-accent transition-[width] duration-300"
                  style={{ width: `${uploadFrac * 100}%` }}
                />
              </div>
            )}
            {err && <div className="text-[12px] text-block break-words">{err}</div>}
            <p className="section-desc">
              Faces and plates are blurred on every imported frame before storage. Imported objects wait in
              the review queue.
            </p>
          </div>
          </div>
        </Section>

        <Section title="Import jobs" flush
          right={jobs.length ? <span className="text-[12px] text-ink-3 tabular-nums">{num(jobs.length)}</span> : undefined}>
          <LoadState {...jobsQ.loadState} rows={4}>
          {jobs.length ? (
            <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="text-left">Source</th>
                  <th className="text-left">Format</th>
                  <th className="text-left">Status</th>
                  <th className="text-left">Progress</th>
                  <th className="num">Frames</th>
                  <th className="num">Objects</th>
                  <th className="text-left">Health</th>
                  <th className="text-right"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
              {jobs.map((j) => {
                const src = sourceName(j.source_uri);
                const h = j.session_id ? health[j.session_id] : null;
                const finished = j.status === "done" || j.status === "error" || j.status === "failed";
                return (
                <tr key={j.job_id}>
                  <td className="max-w-[28rem]">
                    <div className="text-ink truncate" title={j.source_uri ?? undefined}>
                      {src ?? j.target_vehicle ?? formatName(j.format)}
                    </div>
                    <div className="text-[11.5px] text-ink-3 truncate">
                      {[src ? j.target_vehicle : null, j.city, j.created_at ? ago(j.created_at) : null].filter(Boolean).join(" · ")}
                    </div>
                    {/* The whole reason, wrapped: "cannot open vid..." told nobody what to fix. */}
                    {j.error && (
                      <div className="text-[12px] text-block mt-1 whitespace-normal break-words">{j.error}</div>
                    )}
                  </td>
                  <td>{formatName(j.format)}</td>
                  <td><StateBadge state={j.status} /></td>
                  <td>
                    {finished && j.status !== "done" ? <span className="text-ink-3">-</span> : (
                      <span className="inline-flex items-center gap-2">
                        <span className="w-20 h-1.5 rounded-full bg-bg-2 overflow-hidden">
                          <span className={`block h-full ${j.status === "done" ? "bg-pass" : "bg-accent"}`}
                            style={{ width: `${Math.min(1, j.progress || 0) * 100}%` }} />
                        </span>
                        <span className="tabular-nums text-[12px]">{pct(j.progress || 0)}</span>
                      </span>
                    )}
                  </td>
                  <td className="num">{num(j.counts?.frames ?? 0)}</td>
                  <td className="num">{num(j.counts?.objects ?? 0)}</td>
                  <td>
                    {h ? (
                      <span title="Inspector health verdict for this recording">
                        <Badge state={HEALTH[h]?.state ?? "muted"}>{HEALTH[h]?.label ?? humanize(h)}</Badge>
                      </span>
                    ) : <span className="text-ink-3">-</span>}
                  </td>
                  <td className="text-right whitespace-nowrap">
                  {j.session_id && (
                    <span className="inline-flex items-center gap-1.5">
                    <button
                      onClick={() => router.push(`/?session=${j.session_id}`)}
                      className="btn"
                    >
                      View
                    </button>
                    <button
                      onClick={async () => {
                        try {
                          const r = await api.inspectorLichtblick(j.session_id!);
                          window.open(r.url, "_blank", "noopener");
                        } catch (e) {
                          setErr("Could not open the recording viewer (no MCAP for this session, or the viewer is not running): " + humanizeError(e));
                        }
                      }}
                      title="full-power MCAP inspection in the self-hosted Lichtblick"
                      data-tip="Open the full recording in the Lichtblick MCAP viewer (new tab)"
                      className="btn"
                    >
                      Inspect recording
                    </button>
                    </span>
                  )}
                  </td>
                </tr>
                );
              })}
              </tbody>
            </table>
            </div>
          ) : (
            <EmptyState compact icon="upload" title="No imports yet">
              Files you upload above are imported here as jobs, one per file, with their progress and result.
            </EmptyState>
          )}
          </LoadState>
        </Section>
      </div>
    </PageShell>
  );
}
