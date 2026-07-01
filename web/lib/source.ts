// Make the origin of data and annotations obvious: what came from a public dataset import (Mapillary, IDD,
// BDD) versus what your own fleet + annotators produced. Used by badges and filters across the app.

export type SourceKind = "imported" | "app" | "other";

const FORMAT_LABEL: Record<string, string> = {
  mapillary: "Mapillary",
  idd: "IDD",
  bdd: "BDD100K",
  bdd100k: "BDD100K",
  kitti: "KITTI",
  nuscenes: "nuScenes",
  coco: "COCO",
  openlabel: "OpenLABEL",
};

// One object's annotation source.
export function objectSource(source?: string, importFormat?: string | null): { label: string; kind: SourceKind; cls: string } {
  if (source === "imported") {
    const ds = importFormat ? (FORMAT_LABEL[importFormat] ?? importFormat) : "dataset";
    return { label: `Imported · ${ds}`, kind: "imported", cls: "border-info/60 text-info" };
  }
  const label =
    source === "human" ? "Your label"
      : source === "fused" ? "Autolabel"
        : source === "auto_accept" ? "Auto-accepted"
          : source === "interpolated" ? "Interpolated"
            : source || "unknown";
  return { label, kind: "app", cls: "border-pass/60 text-pass" };
}

// A session's data origin (from the /sessions origin field).
export function sessionOrigin(origin?: string): { label: string; kind: SourceKind; cls: string } {
  if (origin === "imported") return { label: "Imported dataset", kind: "imported", cls: "border-info/60 text-info" };
  if (origin === "fleet") return { label: "Your fleet", kind: "app", cls: "border-pass/60 text-pass" };
  return { label: "Other", kind: "other", cls: "border-line text-ink-3" };
}
