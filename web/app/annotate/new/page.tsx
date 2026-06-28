"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import TopNav from "@/components/TopNav";

// New Annotation: upload a folder of images (zip), a video, or an mcap; import it into a fresh
// session, then jump straight into the frame editor on the first frame. The bytes go browser ->
// storage via presigned multipart; the API only signs and runs the import as a background job.

// Only the source formats that make sense for "start annotating from raw media". Richer dataset
// imports (coco/yolo/nuscenes/...) live on the IMPORT page.
const FORMATS = [
  { value: "images", label: "images (zip / folder of images)" },
  { value: "video", label: "video (.mp4 / .mov / .mkv / .avi)" },
  { value: "mcap", label: "mcap (robotics log)" },
];

const VIDEO_EXTS = ["mp4", "mov", "mkv", "avi", "webm", "m4v"];
const IMAGE_EXTS = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];

// Guess the import format from a chosen file's extension. The user can still override the select.
function formatForFile(file: File): string {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "mcap") return "mcap";
  if (VIDEO_EXTS.includes(ext)) return "video";
  if (ext === "zip" || IMAGE_EXTS.includes(ext)) return "images";
  // Fall back on the broad mime type when the extension is missing or unknown.
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("image/")) return "images";
  return "images";
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

type Phase = "idle" | "uploading" | "importing" | "opening";

export default function NewAnnotationPage() {
  const router = useRouter();
  const [format, setFormat] = useState("images");
  const [vehicle, setVehicle] = useState("ANNO-01");
  const [city, setCity] = useState("BLR");
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [uploadFrac, setUploadFrac] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [emptyHint, setEmptyHint] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Choosing a file auto-picks a sensible format; the select still lets the user override it.
  function pickFile(f: File) {
    setFile(f);
    setFormat(formatForFile(f));
    setErr(null);
    setEmptyHint(false);
  }

  async function onGo() {
    if (!file) return;
    setErr(null);
    setEmptyHint(false);
    setProgress(0);

    // a. Upload the file straight to storage, surfacing the per-part progress fraction.
    let uri: string;
    try {
      setPhase("uploading");
      setUploadFrac(0);
      setStatus("uploading file to storage...");
      uri = await api.uploadMultipart(file, setUploadFrac);
    } catch (e) {
      setErr("upload failed: " + String(e));
      setPhase("idle");
      return;
    }

    // b. Kick off the import job for the uploaded object.
    let jobId: string;
    try {
      setPhase("importing");
      setStatus("starting import...");
      const res = await api.startImport({
        format,
        source_uri: uri,
        target_vehicle: vehicle,
        city,
      });
      jobId = res.job_id;
    } catch (e) {
      setErr("could not start import: " + String(e));
      setPhase("idle");
      return;
    }

    // c. Poll the import job until it is done or errors out.
    let sessionId: string;
    try {
      sessionId = await new Promise<string>((resolve, reject) => {
        const t = setInterval(async () => {
          try {
            const job = await api.importStatus(jobId);
            setProgress(job.progress || 0);
            const frames = job.counts?.frames ?? 0;
            const objects = job.counts?.objects ?? 0;
            setStatus(
              `import ${job.status} - ${((job.progress || 0) * 100).toFixed(0)}% - ${frames} frames / ${objects} objects`,
            );
            if (job.status === "error") {
              clearInterval(t);
              reject(new Error(job.error || "import failed"));
              return;
            }
            if (job.status === "done") {
              clearInterval(t);
              if (!job.session_id) {
                reject(new Error("import done but no session was created"));
                return;
              }
              resolve(job.session_id);
            }
          } catch (e) {
            clearInterval(t);
            reject(e);
          }
        }, 2000);
      });
    } catch (e) {
      setErr("import error: " + String(e));
      setPhase("idle");
      return;
    }

    // d. Open the editor on the first frame of the new session.
    try {
      setPhase("opening");
      setStatus("opening editor...");
      const { frame_id } = await api.firstFrame(sessionId);
      router.push("/frame/" + frame_id);
    } catch (e) {
      const msg = String(e);
      // A 404 means the session imported but has no frames to open yet.
      if (msg.includes("404")) {
        setStatus("session created, but it has no frames to open.");
        setEmptyHint(true);
      } else {
        setErr("could not open editor: " + msg);
      }
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="NEW" />

      <main className="flex-1 overflow-auto p-4">
        <div className="max-w-2xl w-full mx-auto space-y-4">
          <Section title="new annotation - upload images, video, or mcap">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]);
              }}
              onClick={() => !busy && inputRef.current?.click()}
              className={`border border-dashed ${
                drag ? "border-accent" : "border-line"
              } p-8 text-center cursor-pointer font-mono text-xs text-ink-3`}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*,.zip,.mp4,.mov,.mkv,.avi,.mcap"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && pickFile(e.target.files[0])}
              />
              {file ? (
                <span className="text-ink">
                  {file.name}{" "}
                  <span className="text-ink-3">({(file.size / 1e6).toFixed(1)} MB)</span>
                </span>
              ) : (
                "drag a .zip of images / a video / a .mcap here, or click to browse"
              )}
            </div>

            <div className="grid grid-cols-3 gap-3 mt-3 font-mono text-xs">
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 uppercase text-[11px]">format</span>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value)}
                  disabled={busy}
                  className="bg-panel border border-line px-2 py-1 text-ink"
                >
                  {FORMATS.map((f) => (
                    <option key={f.value} value={f.value}>
                      {f.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 uppercase text-[11px]">vehicle</span>
                <input
                  value={vehicle}
                  onChange={(e) => setVehicle(e.target.value)}
                  disabled={busy}
                  className="bg-panel border border-line px-2 py-1 text-ink"
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 uppercase text-[11px]">city</span>
                <input
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  disabled={busy}
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
                    ? "importing..."
                    : phase === "opening"
                      ? "opening..."
                      : "Create annotation"}
              </button>
              {(phase === "uploading" || phase === "importing") && (
                <div className="flex-1 h-2 bg-line relative">
                  <div
                    className="absolute left-0 top-0 h-full bg-accent"
                    style={{
                      width: `${(phase === "uploading" ? uploadFrac : progress) * 100}%`,
                    }}
                  />
                </div>
              )}
            </div>

            {status && !err && (
              <div className="font-mono text-[11px] text-ink-2 mt-2">{status}</div>
            )}
            {err && <div className="font-mono text-[11px] text-block mt-2">{err}</div>}
            {emptyHint && (
              <div className="font-mono text-[11px] text-ink-3 mt-1">
                <button
                  onClick={() => router.push("/annotations")}
                  className="text-accent hover:underline"
                >
                  browse existing annotations -&gt;
                </button>
              </div>
            )}

            <div className="font-mono text-[11px] text-ink-3 mt-3 leading-relaxed">
              Your file uploads straight to storage, imports into a new session (PII faces and plates
              are blurred on every frame), and the editor opens on the first frame so you can start
              annotating. Larger videos take longer to decode.
            </div>
          </Section>

          <div className="font-mono text-[11px] text-ink-3 px-1">
            <button
              onClick={() => router.push("/annotations")}
              className="text-accent hover:underline"
            >
              browse existing annotations -&gt;
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
