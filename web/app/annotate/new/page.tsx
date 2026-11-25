"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { watchAutolabelJob, watchImportJob } from "@/lib/useEventStream";
import { type QueueItem, humanSize, shouldAutoOpen, summarize } from "@/lib/uploadQueue";
import { pressable } from "@/components/ui/pressable";
import {
  clearUploads,
  enqueue,
  getUploadState,
  startAutolabel,
  startUploads,
  subscribeUploads,
} from "@/lib/uploadManager";
import { importedSessions } from "@/lib/uploadQueue";
import ImportedPanel from "@/components/annotate/ImportedPanel";
import { T } from "@/lib/i18n";
import Icon from "@/components/shell/Icon";
import { humanize, pct, plural } from "@/lib/format";

// New Annotation: upload clips, a folder of them, or an mcap; import each into its own session, then jump
// into the editor when there was exactly one. The bytes go browser -> storage via presigned multipart; the
// API only signs and runs each import as a background job.
//
// This took one file at a time, which is the wrong unit for how the footage arrives. A dashcam session is a
// folder of clips and the corpus was built from 186 of them, so ingesting it singly meant sitting through an
// upload, a decode and an editor open before picking the next. It now takes a multi-select or a whole
// directory, and runs them one at a time: each import decodes video on a machine with a single GPU slot, so
// ten at once would compete rather than finish sooner.
//
// The one behaviour that had to change rather than extend is the navigation. The old page pushed to
// /frame/<id> the moment its import finished, which unmounts everything, and with a queue behind it that
// abandons uploads mid-transfer. Auto-open is now a property of a queue of exactly one.

const FORMATS = [
  { value: "auto", label: "Detect per file" },
  { value: "images", label: "Images (zip or folder of images)" },
  { value: "video", label: "Video (.mp4, .mov, .mkv, .avi, .webm, .m4v)" },
  { value: "mcap", label: "MCAP (robotics log)" },
];

const ACCEPT = "image/*,video/*,.zip,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mcap";

function Section({ title, step, right, children }: {
  title: string; step?: number; right?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        {step !== undefined && (
          <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-accent/15 text-accent-2 text-[11px] font-semibold tabular-nums shrink-0"
            aria-hidden>{step}</span>
        )}
        <h2 className="section-title">{title}</h2>
        {right && <span className="ml-auto flex items-center gap-2">{right}</span>}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

// The queue's statuses as words. `labeled` and `autolabeling` come from the second (autolabel) pass.
const STATUS_LABEL: Record<string, string> = {
  pending: "Waiting", uploading: "Uploading", importing: "Importing", autolabeling: "Autolabelling",
  done: "Imported", labeled: "Labelled", error: "Failed", skipped: "Skipped",
};

const DOT: Record<string, string> = {
  pending: "bg-line", uploading: "bg-info", importing: "bg-warn",
  done: "bg-pass", error: "bg-block", skipped: "bg-line",
};

/** One queue row. The bar is the item's own phase fraction, so it moves during a long single upload. */
function Row({ item, index, onOpen }: { item: QueueItem; index: number; onOpen: (i: QueueItem) => void }) {
  const active = item.status === "uploading" || item.status === "importing";
  return (
    <div
      // Staggered so a folder of forty clips arrives as a list being filled rather than as a flash of forty
      // rows. Capped, because past about ten the stagger stops reading as motion and starts reading as lag.
      // `.fade-up` and its `--d` delay are the shell's existing entrance, so this inherits the
      // prefers-reduced-motion handling instead of adding a second animation nobody would remember to
      // disable.
      style={{ "--d": `${Math.min(index, 10) * 25}ms` } as React.CSSProperties}
      className="fade-up flex items-center gap-3 px-3 py-2 border-b hairline last:border-0">
      {active
        ? <span className="running-dot shrink-0" />
        : <span className={`w-2 h-2 rounded-full shrink-0 ${DOT[item.status] ?? "bg-line"}`} />}
      <div className="flex-1 min-w-0">
        <div className="text-[12.5px] text-ink truncate" title={item.name}>{item.name}</div>
        {/* The failure reason on its own line, wrapped, rather than cut to "cannot open vid..." beside the name. */}
        {item.detail && item.status === "error" && (
          <div className="text-[12px] text-block mt-0.5 break-words">{item.detail}</div>
        )}
      </div>
      <span className="text-[12px] text-ink-3 shrink-0 tabular-nums">{humanSize(item.size)}</span>
      <span className="text-[12px] text-ink-3 w-16 text-right shrink-0">{humanize(item.format)}</span>
      <span className="w-28 shrink-0">
        {active ? (
          <span className="flex items-center gap-2">
            <span className="flex-1 h-1.5 bg-bg-2 rounded-full overflow-hidden">
              <span className="block h-full bg-accent transition-[width] duration-300 ease-out"
                style={{ width: `${Math.round(item.progress * 100)}%` }} />
            </span>
            <span className="text-[11px] text-ink-3 tabular-nums w-8 text-right">{pct(item.progress)}</span>
          </span>
        ) : (
          <span className={`text-[12px] ${item.status === "error" ? "text-block" : item.status === "done" || item.status === "labeled" ? "text-pass" : "text-ink-3"}`}>
            {STATUS_LABEL[item.status] ?? humanize(item.status)}
          </span>
        )}
      </span>
      {item.sessionId ? (
        // A button, not a link. This was an <a href>, so opening a finished clip navigated away, unmounted
        // the page and abandoned every upload still running, which is the same bug the auto-open rule
        // exists to prevent, left in a second place. It opens a panel beside the queue instead.
        <button onClick={() => onOpen(item)}
          className="link text-[12px] shrink-0 w-10 text-right">
          <T k="action.open">Open</T>
        </button>
      ) : <span className="w-10 shrink-0" aria-hidden />}
    </div>
  );
}

export default function NewAnnotationPage() {
  const router = useRouter();
  const [format, setFormat] = useState("auto");
  const [vehicle, setVehicle] = useState("ANNO-01");
  const [city, setCity] = useState("BLR");
  // The queue itself lives in lib/uploadManager, module-scoped, so it survives this page unmounting. A
  // 41GB batch is over an hour of transfer, and holding it in component state meant any navigation
  // abandoned it. This component is a view over that store, not its owner.
  const [state, setState] = useState(getUploadState);
  const [skipped, setSkipped] = useState<string[]>([]);
  const [duplicates, setDuplicates] = useState(0);
  const [drag, setDrag] = useState(false);
  const [opened, setOpened] = useState<QueueItem | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const dirRef = useRef<HTMLInputElement>(null);

  useEffect(() => subscribeUploads(setState), []);

  const items = state.items;
  const running = state.running;
  const summary = useMemo(() => summarize(items), [items]);

  // Keep the open panel pointed at the live row, so its counts follow the import rather than freezing at
  // whatever they were when it was clicked.
  const openedLive = opened ? items.find((i) => i.id === opened.id) ?? opened : null;
  const labelable = importedSessions(items).length;

  const take = useCallback((list: FileList | File[] | null) => {
    if (!list) return;
    const r = enqueue(Array.from(list));
    setSkipped(r.skipped);
    setDuplicates(r.duplicates);
    setErr(null);
  }, []);

  // Deps are assembled once here rather than inside each handler, so the import pass and the label pass
  // cannot drift apart on which client they call.
  const deps = {
    upload: api.uploadMultipart,
    startImport: api.startImport,
    watchImport: watchImportJob,
    firstFrame: api.firstFrame,
    humanizeError,
    startAutolabel: (sessionId: string) => api.startAutolabel(sessionId),
    watchAutolabel: watchAutolabelJob,
  };

  async function onAutolabel() {
    if (running) return;
    await startAutolabel(deps);
  }

  async function onGo() {
    if (!items.length || running) return;
    setErr(null);
    await startUploads({ format, vehicle, city }, deps);
    // Only ever for a queue of one. Navigating unmounts the page, which is now survivable, but it is still
    // not what somebody importing a folder asked for.
    const after = getUploadState().items;
    if (shouldAutoOpen(after)) router.push("/frame/" + after[0].frameId);
  }

  const label = running
    ? `Working ${summary.done + summary.failed} of ${summary.total}`
    : items.length > 1 ? `Import ${items.length} files` : "Create annotation";

  const primaryAction = (
    <button onClick={onGo} disabled={!items.length || running}
      className="btn btn-primary transition-colors"
      data-tip={!items.length ? "Add files first" : undefined}>
      {running && <span className="running-dot mr-1.5 align-middle" />}
      {label}
    </button>
  );

  return (
    <PageShell active="NEW" title="New Annotation"
      subtitle="Upload drive footage; each file becomes a session you can autolabel and review."
      primaryAction={primaryAction}>
      <div className="page-wrap-narrow space-y-4">
        <Section
          step={1}
          title="Add footage"
          right={items.length ? (
            <button onClick={() => { clearUploads(); setSkipped([]); setDuplicates(0); setOpened(null); }}
              disabled={running}
              className="btn"><T k="action.clear">Clear</T></button>
          ) : null}
        >
          {/* Outside the dropzone, deliberately. These were inside it, and calling .click() on one
              dispatches a click that bubbles up to the dropzone's own onClick, which opens a second
              chooser in the same gesture. The browser refuses that and says so:
                "File chooser dialog can only be shown with a user activation."
              A hidden input does not need to live inside the drop target, and keeping it out means the
              two handlers can never trigger each other. */}
          <input ref={fileRef} type="file" accept={ACCEPT} multiple className="hidden"
            onChange={(e) => take(e.target.files)} />
          {/* A second input carrying webkitdirectory: one element cannot offer both a file picker and a
              folder picker, and a folder is how these clips sit on disk. The prop is not in React's
              HTMLInputElement types but is what every browser implements, so it is spread through a
              typed record rather than cast away with `any`. */}
          <input ref={dirRef} type="file" className="hidden" multiple
            {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
            onChange={(e) => take(e.target.files)} />
          <div
            {...pressable(() => fileRef.current?.click(), { disabled: running })}
            aria-label="Choose files to upload, or drop them here"
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); if (!running) take(e.dataTransfer.files); }}
            className={`flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-lg px-6 py-12 text-center cursor-pointer transition-all duration-150
              ${drag ? "border-accent bg-accent/5 scale-[1.01]" : "border-line bg-bg-2/40 hover:border-accent/60 hover:bg-accent/[0.03]"}
              ${running ? "opacity-60 cursor-default" : ""}`}>
            <span className={`flex items-center justify-center w-12 h-12 rounded-full border mb-1 ${drag ? "border-accent text-accent-2 bg-accent/10" : "border-line bg-head text-ink-2"}`}
              aria-hidden>
              <Icon name="upload" size={22} />
            </span>
            {items.length ? (
              <>
                <span className="text-[15px] font-medium text-ink">
                  {plural(items.length, "file")} ready
                  <span className="text-ink-3 font-normal"> ({humanSize(items.reduce((a, i) => a + i.size, 0))})</span>
                </span>
                <span className="text-[12.5px] text-ink-3">Drop more here or click to add files</span>
              </>
            ) : (
              <>
                <span className="text-[15px] font-medium text-ink">Drop clips, a folder or an MCAP file here</span>
                <span className="text-[12.5px] text-ink-3">or <span className="text-accent-2">click to browse</span></span>
                <span className="text-[12px] text-ink-3 mt-1">Video, images (or a zip of them) and .mcap logs</span>
              </>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3 mt-3">
            <button onClick={() => !running && dirRef.current?.click()} disabled={running}
              className="btn">
              <Icon name="folder" size={14} />
              Choose a folder
            </button>
            <span className="text-[12px] text-ink-3">
              Every clip becomes its own session, imported one at a time.
            </span>
          </div>

          {(skipped.length > 0 || duplicates > 0) && (
            <div className="reveal text-[12px] text-ink-3 mt-3">
              {/* Said rather than silent: a folder pick sweeps up sidecars and dotfiles, and uploading
                  those would produce one failed import each and bury the real ones. */}
              {skipped.length > 0 && <>{skipped.length} non-media file{skipped.length === 1 ? "" : "s"} skipped ({skipped.slice(0, 3).join(", ")}{skipped.length > 3 ? ", ..." : ""}). </>}
              {duplicates > 0 && <>{duplicates} duplicate{duplicates === 1 ? "" : "s"} ignored.</>}
            </div>
          )}
        </Section>

        <Section step={2} title="Describe the drive">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-[12.5px]">
            <label className="flex flex-col gap-1.5">
              <span className="text-[12px] font-medium text-ink-2">Format</span>
              <select value={format} onChange={(e) => setFormat(e.target.value)} disabled={running}
                className="field">
                {FORMATS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-[12px] font-medium text-ink-2">Vehicle</span>
              <input value={vehicle} onChange={(e) => setVehicle(e.target.value)} disabled={running}
                className="field" />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-[12px] font-medium text-ink-2">City</span>
              <input value={city} onChange={(e) => setCity(e.target.value)} disabled={running}
                className="field" />
            </label>
          </div>

          {err && <div className="text-[12px] text-block mt-3">{err}</div>}

          <p className="section-desc mt-4">
            Faces and number plates are blurred on every frame. A single file opens the editor when it finishes.
          </p>
          <details className="mt-2 text-[12px] text-ink-3">
            <summary className="cursor-pointer hover:text-ink-2">How uploads work</summary>
            <p className="mt-1.5 leading-relaxed">
              Each file uploads straight to storage and imports into its own session. A batch keeps running if
              you move around the app and notifies you when it is done, so you can carry on working; closing the
              tab is the only thing that stops it. Larger videos take longer to decode.
            </p>
          </details>
        </Section>

        {items.length > 0 && (
          <Section
            step={3}
            title={running || !summary.finished ? "Upload and import" : "Imported"}
            right={
              <span className="text-[12px] text-ink-3 tabular-nums">
                {summary.done} done · {summary.failed > 0 ? <span className="text-block">{summary.failed} failed</span> : "0 failed"} · {summary.pending} waiting
                <span className="ml-2 text-ink font-medium">{pct(summary.progress)}</span>
              </span>
            }
          >
            <div className="h-1.5 bg-bg-2 rounded-full overflow-hidden mb-3">
              <div className="h-full bg-accent transition-[width] duration-500 ease-out"
                style={{ width: `${summary.progress * 100}%` }} />
            </div>
            <div className="flex gap-3 items-start">
              <div className="flex-1 min-w-0 border hairline rounded overflow-hidden max-h-[36rem] overflow-y-auto">
                {items.map((i, n) => <Row key={i.id} item={i} index={n} onOpen={setOpened} />)}
              </div>
              {openedLive?.sessionId && (
                <ImportedPanel item={openedLive} onClose={() => setOpened(null)} />
              )}
            </div>
            {summary.finished && !running && (
              <div className="reveal flex flex-wrap items-center gap-3 mt-3">
                <span className="text-[12.5px] text-ink-2">
                  {plural(summary.done, "session")} imported
                  {summary.failed > 0 && <span className="text-block"> · {summary.failed} failed</span>}
                </span>
                {/* Offered only once every import has finished. Autolabel is GPU work on a machine with one
                    slot, so starting it while clips are still decoding would put two jobs in flight and
                    lose the ordering that makes a batch followable. */}
                {labelable > 0 && (
                  <button onClick={onAutolabel} className="btn btn-on">
                    Autolabel {plural(labelable, "session")}
                  </button>
                )}
                <button onClick={() => router.push("/annotations")}
                  className="ml-auto link text-[12.5px]">
                  Browse sessions →
                </button>
              </div>
            )}
          </Section>
        )}

        <div className="text-[12.5px] px-1 flex items-center gap-5">
          <button onClick={() => router.push("/annotations")} className="link">
            Browse existing sessions →
          </button>
          {/* Added because the batch now keeps running after you leave, and nothing pointed at where to
              watch it. An import is a job like any other, and the jobs page already lists them all. */}
          <button onClick={() => router.push("/jobs")} className="link">
            Watch import jobs →
          </button>
        </div>
      </div>
    </PageShell>
  );
}
