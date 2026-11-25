// Turning what the API stores into what a person reads.
//
// Pages were printing the storage form: `adjust_geometry`, `CUT_IN`, a progress of `0.05`, a window of
// `1786533661949`, a job named after the first eight characters of a uuid. Each is accurate and none of them is
// what somebody looking at the screen wants to know. The same three or four conversions were also written out
// again per page (`ago` existed four times, with three different thresholds), so the same instant read "21 h
// ago" on Home and "21h ago" in the bell.
//
// Pure functions, so they are tested without a DOM like the rest of web/lib.

/** `adjust_geometry` / `CUT_IN` / `two-wheeler` -> "Adjust geometry" / "Cut in" / "Two wheeler". */
// Words that keep their own case when a code is turned into a phrase: "suv" read as "Suv", "vlm_verify" as
// "Vlm verify". Lower-case keys, written form as the value.
const KEEP_CASE: Record<string, string> = {
  suv: "SUV", vlm: "VLM", pii: "PII", gpu: "GPU", cpu: "CPU", lidar: "LiDAR", odd: "ODD", id: "ID", ids: "IDs",
  ai: "AI", sam: "SAM", mcap: "MCAP", coco: "COCO", yolo: "YOLO", kitti: "KITTI", api: "API", url: "URL",
  anpr: "ANPR", ocr: "OCR", nms: "NMS", iou: "IoU", map: "mAP", fps: "FPS", rgb: "RGB", "3d": "3D", "2d": "2D",
  slo: "SLO", vru: "VRU", tp: "TP", fp: "FP", fn: "FN", llm: "LLM", ttc: "TTC",
};

export function humanize(code: string | null | undefined): string {
  if (!code) return "";
  const words = String(code).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().toLowerCase();
  if (!words) return "";
  const out = words.split(" ").map((w, i) => {
    const kept = KEEP_CASE[w];
    // "map" is mAP only as a metric name on its own; "map fusion" stays a map.
    if (kept && !(w === "map" && words !== "map")) return kept;
    return i === 0 ? w[0].toUpperCase() + w.slice(1) : w;
  });
  return out.join(" ");
}

/** A 0..1 fraction as a whole percentage: 0.05 -> "5%", 1 -> "100%". Values above 1 are read as percentages. */
export function pct(frac: number | null | undefined, digits = 0): string {
  if (frac === null || frac === undefined || !Number.isFinite(frac)) return "-";
  const p = frac > 1 ? frac : frac * 100;
  // A job at 0.3% is running, not at zero: show the first decimal rather than rounding it away.
  if (digits === 0 && p > 0 && p < 1) return "<1%";
  return `${p.toFixed(digits)}%`;
}

/** 12345 -> "12,345"; null -> "-". */
export function num(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "-";
  return n.toLocaleString("en-US");
}

/** 882079 -> "882k", 1250000 -> "1.3M". For tiles and chips where width matters more than the last digit. */
export function compact(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "-";
  const a = Math.abs(n);
  if (a < 10_000) return n.toLocaleString("en-US");
  if (a < 1_000_000) return `${Math.round(n / 1000)}k`;
  return `${(n / 1_000_000).toFixed(a < 10_000_000 ? 1 : 0)}M`;
}

/** Seconds since an ISO instant (or epoch ms), as "just now", "4 min ago", "3 h ago", "2 days ago", or a date. */
export function ago(when: string | number | Date | null | undefined, now: number = Date.now()): string {
  if (when === null || when === undefined || when === "") return "-";
  const t = when instanceof Date ? when.getTime() : typeof when === "number" ? when : Date.parse(when);
  if (!Number.isFinite(t)) return "-";
  const s = Math.max(0, (now - t) / 1000);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`;
  if (s < 129600) return `${Math.round(s / 3600)} h ago`;
  if (s < 30 * 86400) return `${Math.round(s / 86400)} days ago`;
  return new Date(t).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/** A nanosecond timestamp (the frame clock) as a local date and time; "-" when absent or not a real instant. */
export function fromNs(ns: number | string | bigint | null | undefined): string {
  if (ns === null || ns === undefined || ns === "") return "-";
  const ms = Number(BigInt(String(ns).split(".")[0]) / 1_000_000n);
  // Frame clocks of synthetic or relative recordings start near zero; those are offsets, not dates.
  if (ms < Date.UTC(2000, 0, 1)) return duration(ms / 1000);
  return new Date(ms).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** 3725 -> "1 h 2 min", 95 -> "1 min 35 s", 4.2 -> "4.2 s". */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "-";
  const s = Math.max(0, seconds);
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) {
    const m = Math.floor(s / 60), r = Math.round(s % 60);
    return r ? `${m} min ${r} s` : `${m} min`;
  }
  const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

/** The first 8 characters of a uuid, for when an id is all there is and it still needs to be told apart. */
export function shortId(id: string | null | undefined): string {
  return id ? String(id).slice(0, 8) : "-";
}

/** "1 frame" / "3 frames". */
export function plural(n: number, one: string, many = `${one}s`): string {
  return `${num(n)} ${n === 1 ? one : many}`;
}
