// The upload queue behind "new annotation", which used to take exactly one file.
//
// One file per upload is the wrong unit for how this footage arrives. A dashcam session is a folder of
// clips, and the corpus was built from 186 of them; ingesting that a file at a time means sitting through
// upload, decode and editor-open before choosing the next one. So the panel now takes a multi-select or a
// whole folder, and this module is the state machine behind it.
//
// Three decisions here are about correctness rather than convenience, and each is easy to get wrong.
//
// **The editor must not open on the first finished file.** The old flow navigated to /frame/<id> the moment
// its single import completed, which is right for one file and destroys a queue: the page unmounts and
// everything still uploading is abandoned mid-transfer. Auto-open is therefore a property of a queue of
// exactly one, not of an item finishing.
//
// **A folder is not a list of media.** Picking a directory hands over .DS_Store, sidecar .json, thumbnails
// and whatever else is in there. Uploading those produces one failed import per file and buries the real
// ones, so unsupported entries are dropped at selection time and counted, rather than being allowed to fail
// later where the failure looks like the importer's.
//
// **Work is sequential, not parallel.** Each import decodes video on the server, and the machine has one
// GPU slot that everything else already contends for. Ten parallel imports would not finish sooner; they
// would compete, and a browser uploading ten multipart streams at once is its own problem. The queue runs
// one at a time and says which one.

export type ItemStatus = "pending" | "uploading" | "importing" | "done" | "error" | "skipped";

export type QueueItem = {
  id: string;
  name: string;
  size: number;
  format: string;
  status: ItemStatus;
  /** 0..1 within the current phase, so the row's bar means something during both upload and import. */
  progress: number;
  detail?: string;
  sessionId?: string;
  frameId?: string;
};

const VIDEO_EXTS = ["mp4", "mov", "mkv", "avi", "webm", "m4v"];
const IMAGE_EXTS = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"];

/** The format an import would use for this file, or null when it is not media this page can ingest. */
export function formatForName(name: string, mime = ""): string | null {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "mcap") return "mcap";
  if (ext === "zip") return "images";
  if (VIDEO_EXTS.includes(ext)) return "video";
  if (IMAGE_EXTS.includes(ext)) return "images";
  if (mime.startsWith("video/")) return "video";
  if (mime.startsWith("image/")) return "images";
  return null;
}

/** Files a folder pick throws in that are never media, matched before the extension check for clarity. */
function isJunk(name: string): boolean {
  const base = name.split("/").pop() ?? name;
  return base.startsWith(".") || base === "Thumbs.db";
}

export type BuildResult = { items: QueueItem[]; skipped: string[]; duplicates: number };

/**
 * Turn a file selection into a queue.
 *
 * Deduplicates on name and size, because dragging a folder and then dragging it again is a normal accident
 * and importing the same clip twice creates two sessions that are hard to tell apart afterwards.
 */
export function buildQueue(files: { name: string; size: number; type?: string }[]): BuildResult {
  const items: QueueItem[] = [];
  const skipped: string[] = [];
  const seen = new Set<string>();
  let duplicates = 0;

  for (const f of files) {
    const key = `${f.name}:${f.size}`;
    if (seen.has(key)) {
      duplicates++;
      continue;
    }
    seen.add(key);

    if (isJunk(f.name)) {
      skipped.push(f.name);
      continue;
    }
    const format = formatForName(f.name, f.type ?? "");
    if (!format) {
      skipped.push(f.name);
      continue;
    }
    items.push({
      id: key,
      name: f.name,
      size: f.size,
      format,
      status: "pending",
      progress: 0,
    });
  }
  return { items, skipped, duplicates };
}

/** Replace one item, by id, returning a new array. */
export function patchItem(items: QueueItem[], id: string, patch: Partial<QueueItem>): QueueItem[] {
  return items.map((i) => (i.id === id ? { ...i, ...patch } : i));
}

/** The next item to work, or null when the queue is finished. */
export function nextPending(items: QueueItem[]): QueueItem | null {
  return items.find((i) => i.status === "pending") ?? null;
}

export type Summary = {
  total: number;
  done: number;
  failed: number;
  active: number;
  pending: number;
  /** Overall completion across the whole queue, counting the in-flight item's own fraction. */
  progress: number;
  finished: boolean;
};

export function summarize(items: QueueItem[]): Summary {
  const total = items.length;
  const done = items.filter((i) => i.status === "done").length;
  const failed = items.filter((i) => i.status === "error").length;
  const active = items.filter((i) => i.status === "uploading" || i.status === "importing").length;
  const pending = items.filter((i) => i.status === "pending").length;
  // Partial credit for the item in flight, so a single large video does not leave the bar frozen at 0 for
  // several minutes and look stuck.
  const partial = items
    .filter((i) => i.status === "uploading" || i.status === "importing")
    .reduce((a, i) => a + Math.max(0, Math.min(1, i.progress)), 0);
  return {
    total,
    done,
    failed,
    active,
    pending,
    progress: total ? Math.min(1, (done + failed + partial) / total) : 0,
    finished: total > 0 && done + failed === total,
  };
}

/**
 * Whether finishing this queue should open the editor.
 *
 * True only for a single successful item. Navigating away unmounts the page, and with anything else still
 * in flight that abandons it mid-upload, which is the bug this whole module exists around.
 */
export function shouldAutoOpen(items: QueueItem[]): boolean {
  return items.length === 1 && items[0].status === "done" && !!items[0].frameId;
}

/** Human file size, for a row that has to fit a filename beside it. */
export function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = bytes / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[u]}`;
}
