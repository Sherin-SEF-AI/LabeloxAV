"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { watchImportJob } from "@/lib/useEventStream";
import { type QueueItem, humanSize, shouldAutoOpen, summarize } from "@/lib/uploadQueue";
import {
  clearUploads,
  enqueue,
  getUploadState,
  startUploads,
  subscribeUploads,
} from "@/lib/uploadManager";
import ImportedPanel from "@/components/annotate/ImportedPanel";

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
  { value: "auto", label: "auto (detect per file)" },
  { value: "images", label: "images (zip / folder of images)" },
  { value: "video", label: "video (.mp4 / .mov / .mkv / .avi)" },
  { value: "mcap", label: "mcap (robotics log)" },
];

const ACCEPT = "image/*,video/*,.zip,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mcap";

function Section({ title, right, children }: {
  title: string; right?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        <span>{title}</span>
        {right && <span className="ml-auto normal-case">{right}</span>}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

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
      className="fade-up flex items-center gap-2 px-2 py-1.5 border-b hairline last:border-0">
      {active
        ? <span className="running-dot shrink-0" />
        : <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT[item.status]}`} />}
      <span className="font-mono text-[11px] text-ink truncate flex-1 min-w-0" title={item.name}>
        {item.name}
      </span>
      <span className="font-mono text-[10px] text-ink-3 shrink-0">{humanSize(item.size)}</span>
      <span className="font-mono text-[10px] text-ink-3 w-14 text-right shrink-0">{item.format}</span>
      <span className="w-24 shrink-0">
        {active ? (
          <span className="block h-1 bg-line rounded overflow-hidden">
            <span className="block h-full bg-accent transition-[width] duration-300 ease-out"
              style={{ width: `${Math.round(item.progress * 100)}%` }} />
          </span>
        ) : (
          <span className={`font-mono text-[10px] ${item.status === "error" ? "text-block" : item.status === "done" ? "text-pass" : "text-ink-3"}`}>
            {item.status}
          </span>
        )}
      </span>
      {item.sessionId && (
        // A button, not a link. This was an <a href>, so opening a finished clip navigated away, unmounted
        // the page and abandoned every upload still running, which is the same bug the auto-open rule
        // exists to prevent, left in a second place. It opens a panel beside the queue instead.
        <button onClick={() => onOpen(item)}
          className="font-mono text-[10px] text-accent hover:underline shrink-0">
          open
        </button>
      )}
      {item.detail && item.status === "error" && (
        <span className="font-mono text-[10px] text-block truncate max-w-[14rem]" title={item.detail}>
          {item.detail}
        </span>
      )}
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

  const take = useCallback((list: FileList | File[] | null) => {
    if (!list) return;
    const r = enqueue(Array.from(list));
    setSkipped(r.skipped);
    setDuplicates(r.duplicates);
    setErr(null);
  }, []);

  async function onGo() {
    if (!items.length || running) return;
    setErr(null);
    await startUploads({ format, vehicle, city }, {
      upload: api.uploadMultipart,
      startImport: api.startImport,
      watchImport: watchImportJob,
      firstFrame: api.firstFrame,
      humanizeError,
    });
    // Only ever for a queue of one. Navigating unmounts the page, which is now survivable, but it is still
    // not what somebody importing a folder asked for.
    const after = getUploadState().items;
    if (shouldAutoOpen(after)) router.push("/frame/" + after[0].frameId);
  }

  const label = running
    ? `working ${summary.done + summary.failed}/${summary.total}`
    : items.length > 1 ? `Import ${items.length} files` : "Create annotation";

  const primaryAction = (
    <button onClick={onGo} disabled={!items.length || running}
      className="font-mono text-xs border border-line px-3 py-1 hover:border-accent disabled:opacity-50 transition-colors">
      {running && <span className="running-dot mr-1.5 align-middle" />}
      {label}
    </button>
  );

  return (
    <PageShell active="NEW" title="New Annotation" primaryAction={primaryAction}>
      <div className="p-4">
        <div className="max-w-3xl w-full mx-auto space-y-4">
          <Section
            title="new annotation - upload images, video, or mcap"
            right={items.length ? (
              <button onClick={() => { clearUploads(); setSkipped([]); setDuplicates(0); setOpened(null); }}
                disabled={running}
                className="font-mono text-[10px] text-ink-3 hover:text-ink disabled:opacity-40">clear</button>
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
              onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); if (!running) take(e.dataTransfer.files); }}
              onClick={() => !running && fileRef.current?.click()}
              className={`border border-dashed rounded p-8 text-center cursor-pointer font-mono text-xs transition-all duration-150
                ${drag ? "border-accent bg-accent/5 text-ink scale-[1.01]" : "border-line text-ink-3 hover:border-ink-3"}`}>
              {items.length ? (
                <span className="text-ink">
                  {items.length} file{items.length === 1 ? "" : "s"} ready
                  <span className="text-ink-3"> ({humanSize(items.reduce((a, i) => a + i.size, 0))})</span>
                </span>
              ) : (
                <>drag clips, a folder, or a .mcap here, or click to browse</>
              )}
            </div>

            <div className="flex items-center gap-3 mt-2">
              <button onClick={() => !running && dirRef.current?.click()} disabled={running}
                className="font-mono text-[11px] border border-line rounded px-2 py-1 text-ink-2 hover:border-accent hover:text-ink disabled:opacity-40 transition-colors">
                choose a folder
              </button>
              <span className="font-mono text-[10px] text-ink-3">
                every clip becomes its own session, imported one at a time
              </span>
            </div>

            {(skipped.length > 0 || duplicates > 0) && (
              <div className="reveal font-mono text-[10px] text-ink-3 mt-2">
                {/* Said rather than silent: a folder pick sweeps up sidecars and dotfiles, and uploading
                    those would produce one failed import each and bury the real ones. */}
                {skipped.length > 0 && <>{skipped.length} non-media file{skipped.length === 1 ? "" : "s"} skipped ({skipped.slice(0, 3).join(", ")}{skipped.length > 3 ? ", ..." : ""}). </>}
                {duplicates > 0 && <>{duplicates} duplicate{duplicates === 1 ? "" : "s"} ignored.</>}
              </div>
            )}

            <div className="grid grid-cols-3 gap-3 mt-3 font-mono text-xs">
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 uppercase text-[11px]">format</span>
                <select value={format} onChange={(e) => setFormat(e.target.value)} disabled={running}
                  className="bg-panel border border-line px-2 py-1 text-ink">
                  {FORMATS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 uppercase text-[11px]">vehicle</span>
                <input value={vehicle} onChange={(e) => setVehicle(e.target.value)} disabled={running}
                  className="bg-panel border border-line px-2 py-1 text-ink" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 uppercase text-[11px]">city</span>
                <input value={city} onChange={(e) => setCity(e.target.value)} disabled={running}
                  className="bg-panel border border-line px-2 py-1 text-ink" />
              </label>
            </div>

            {err && <div className="font-mono text-[11px] text-block mt-2">{err}</div>}

            <div className="font-mono text-[11px] text-ink-3 mt-3 leading-relaxed">
              Each file uploads straight to storage and imports into its own session (PII faces and plates
              are blurred on every frame). One file opens the editor when it finishes. A batch keeps running
              if you move around the app, and notifies you when it is done, so you can carry on working;
              closing the tab is the only thing that stops it. Larger videos take longer to decode.
            </div>
          </Section>

          {items.length > 0 && (
            <Section
              title={`queue - ${summary.done} done, ${summary.failed} failed, ${summary.pending} waiting`}
              right={
                <span className="font-mono text-[10px] text-ink-3">
                  {Math.round(summary.progress * 100)}%
                </span>
              }
            >
              <div className="h-1 bg-line rounded overflow-hidden mb-2">
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
              {summary.finished && (
                <div className="reveal font-mono text-[11px] text-ink-2 mt-2">
                  {summary.done} session{summary.done === 1 ? "" : "s"} imported
                  {summary.failed > 0 && <span className="text-block"> · {summary.failed} failed</span>}
                  {" · "}
                  <button onClick={() => router.push("/annotations")} className="text-accent hover:underline">
                    browse annotations -&gt;
                  </button>
                </div>
              )}
            </Section>
          )}

          <div className="font-mono text-[11px] text-ink-3 px-1">
            <button onClick={() => router.push("/annotations")} className="text-accent hover:underline">
              browse existing annotations -&gt;
            </button>
          </div>
        </div>
      </div>
    </PageShell>
  );
}
