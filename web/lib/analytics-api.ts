// Analytics dashboard client. Kept separate from lib/api.ts to avoid integration conflicts.
// Same-origin: next.config rewrites /api/* to the FastAPI backend.

export type Overview = {
  sessions: number;
  frames: number;
  objects: number;
  tracks: number;
  scenarios: number;
  auto_accepted_pct: number;
  human_touched_pct: number;
  long_tail: { covered_classes: number; total_classes: number; coverage_pct: number };
  source_mix: SourceMix;
};

export type ClassRow = {
  class_id: number;
  name: string;
  l0: string;
  l1: string;
  india: boolean;
  count: number;
};

export type SourceMix = {
  total: number;
  by_state: Record<string, number>;
  by_source: Record<string, number>;
  auto_accepted_pct: number;
  human_touched_pct: number;
};

export type ScenarioCoverage = { type: string; count: number; mean_criticality: number };

export type GeoPoint = { lat: number; lon: number };

export type ReviewAgreement = {
  total_reviews: number;
  confirmed: number;
  reclassified: number;
  confirmed_pct: number;
  reclassified_pct: number;
  mean_time_spent_ms: number;
  per_class: {
    class_id: number;
    name: string;
    reviews: number;
    confirmed: number;
    confirmed_pct: number;
  }[];
};

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`);
  return r.json();
}

function qs(session_id?: string): string {
  return session_id ? "?" + new URLSearchParams({ session_id }).toString() : "";
}

export type PiiCoverage = {
  total_frames: number;
  frames_anonymized: number;
  coverage_pct: number;
  faces_blurred: number;
  plates_blurred: number;
  method_versions: Record<string, number>;
};

export const analyticsApi = {
  overview: (session_id?: string) => get<Overview>("/api/analytics/overview" + qs(session_id)),
  classes: (session_id?: string) => get<ClassRow[]>("/api/analytics/classes" + qs(session_id)),
  sourceMix: (session_id?: string) => get<SourceMix>("/api/analytics/source-mix" + qs(session_id)),
  scenarios: (session_id?: string) =>
    get<ScenarioCoverage[]>("/api/analytics/scenarios" + qs(session_id)),
  geo: (session_id?: string) => get<GeoPoint[]>("/api/analytics/geo" + qs(session_id)),
  reviewAgreement: () => get<ReviewAgreement>("/api/analytics/review-agreement"),
  pii: (session_id?: string) => get<PiiCoverage>("/api/analytics/pii" + qs(session_id)),
  sceneSplits: (session_id?: string) => get<Record<string, Record<string, number>>>("/api/analytics/scene-splits" + qs(session_id)),
  dedupRate: (session_id?: string) => get<DedupRate>("/api/analytics/dedup-rate" + qs(session_id)),
  growth: () => get<GrowthPoint[]>("/api/analytics/growth"),
  clusterMap: (limit = 1500) => get<ClusterMap>(`/api/analytics/cluster-map?limit=${limit}`),
};

export type DedupRate = { total: number; redundant: number; groups: number; rate: number };
export type GrowthPoint = { date: string; frames: number; cumulative: number };
export type ClusterPoint = { frame_id: string; x: number; y: number; cluster: number; time_of_day: string | null; road_type: string | null };
export type ClusterMap = { points: ClusterPoint[]; n: number; clusters: number };
