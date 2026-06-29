export type TriageRow = {
  object_id: string;
  frame_id: string;
  session_id: string;
  class_id: number;
  class_name: string;
  conf: number;
  state: string;
  why: string;
  priority: number;
};

// COCO-style keypoints/skeleton: points are [x, y, v] with v in {0 not-labeled, 1 occluded, 2 visible}.
export type Keypoints = { skeleton: string; points: number[][] };

export type ObjectDetail = {
  object_id: string;
  frame_id: string;
  session_id: string;
  ts_ns: number;
  cam_id: string;
  image_url: string;
  width: number;
  height: number;
  class_id: number;
  class_name: string;
  bbox: number[];
  mask_polygons: number[][];
  attrs: Record<string, unknown>;
  conf: number;
  state: string;
  source: string;
  provenance: Record<string, unknown>;
  version?: number;
  rot_deg?: number;
  keypoints?: Keypoints | null;
  polyline?: number[][] | null;
};

export type OntologyClass = { id: number; name: string; l0: string; l1: string; india: boolean };
export type Ontology = {
  version: string;
  hierarchy_levels: number;
  attributes: Record<string, { type: string; values: unknown[] | null; range: number[] | null }>;
  classes: OntologyClass[];
  // per-subclass (l1) applicable-attribute allowlist; a subclass absent here means all attributes apply
  attribute_scope?: Record<string, string[]>;
};

export type SessionRow = {
  session_id: string;
  vehicle_id: string;
  city: string | null;
  route: string | null;
  start_ts_ns: number;
  end_ts_ns: number;
};

export type SegmentResult = { polygons: number[][]; bbox: number[] | null };

export type UserRow = { user_id: string; name: string; role: string; reviews: number };

export type DatasetRow = {
  commit_id: string;
  name: string | null;
  object_count: number;
  formats: string[];
  ontology_version: string;
  n_files: number;
  created_at: string | null;
};

export type CurationSummary = {
  total_frames: number;
  embedded: number;
  embedded_pct: number;
  mean_nn_sim: number | null;
  duplicate_frames: number;
  novel: { frame_id: string; novelty: number; image_url: string }[];
  duplicates: { a: string; b: string; sim: number; a_url: string; b_url: string }[];
};

export type CorrectionCandidate = {
  object_id: string;
  frame_id: string;
  class_name: string;
  current: string | number | boolean | null;
  conf: number;
  state: string;
  score: number;
  crop_url: string;
  already: boolean;
};

export type CorrectionSuggestion = {
  kind: "class" | "attr";
  change: Record<string, unknown>;
  count: number;
  candidates: CorrectionCandidate[];
  reason?: string;
};

export type ConfusionRow = { old_class: string; new_class: string; count: number; group?: string };
export type Confusions = { by: string; total_corrections: number; confusions: ConfusionRow[] };
export type CorrectionCoverage = { embedded: number; total: number; pct: number };

export type AlItem = { object_id: string; frame_id: string; class_name: string; conf: number; value: number; scores: { uncertainty: number; diversity: number; rarity: number; error_prone: number } };
export type ErrorCandidateRow = { candidate_id: string; object_id: string; kind: string; score: number; proposed_label: { class_name?: string } | null; detail: Record<string, unknown>; status: string };
export type GovState = { loop_enabled: boolean; auto_accept_enabled: boolean; auto_promote_enabled: boolean; champion_version: string | null; paused_reason: string | null; updated_at: string | null };
export type RegistryRow = { model_version: string; task: string; is_champion: boolean; promoted_from: string | null; gold_metrics: Record<string, unknown>; dataset_commit: string | null; created_at: string | null };
export type AuditRow = { audit_id: string; actor: string; decision: string; subject: string | null; rationale: Record<string, unknown>; created_at: string | null };

export type AssignmentRow = { assignment_id: string; item_id: string; user: string; branch: string; status: string };
export type MergeRequestRow = { mr_id: string; title: string; source_branch: string; target_branch: string; status: string; merge_commit: string | null; created_at: string | null };

export type MapCommitRow = { commit_id: string; region: string; element_count: number; session_ids: string[]; formats: Record<string, string>; calibration_version: string | null; created_at: string | null };
export type MapFeature = { type: "Feature"; geometry: { type: string; coordinates: number[] | number[][] } | null; properties: Record<string, unknown> & { element_id: string; kind: string; confidence: number } };
export type MapProvenance = { found: boolean; element_id?: string; kind?: string; attrs?: Record<string, unknown>; confidence?: number; calibration_version?: string | null; commit_id?: string | null; fusion_job_id?: string | null; source_sessions?: string[] | null; source_frames?: { frame_id: string; session_id: string; cam_id: string; ts_ns: number; vehicle_id: string | null }[] };

export type MulticamGroups = {
  cameras: string[];
  multicamera: boolean;
  n_groups: number;
  groups: { ts_ns: number; frames: Record<string, { frame_id: string; img_uri: string }> }[];
};

export type CalibFovCheck = { implied_fov_deg: number; expected_fov_deg: number | null; diff_deg: number | null; tolerance_deg: number; ok: boolean };
export type CalibCamera = { cam_id: string; model: string; lens?: string; reproj_error_px: number | null; fov_check: CalibFovCheck; time_offset_ns: number | null; status: string };
export type CalibDetail = { session_id: string; cameras_in_session: string[]; validations: CalibCamera[]; overall: string };
export type CalibSession = { session_id: string; vehicle_id: string; cameras: number; fail: number; overall: string };

export type LaneRow = {
  lane_id: string;
  frame_id: string;
  track_ref: string | null;
  control_points: number[][];
  lane_type: string;
  is_ego: boolean;
  source: string;
  model_version: string | null;
};

export type DiscoveryCandidate = {
  candidate_id: string;
  frame_id: string;
  session_id: string;
  vehicle_id: string;
  kind: "embedding_outlier" | "sparse_cluster" | "rare_class";
  score: number;
  cluster_id: number | null;
  rare_classes: string[];
  state: string;
  tag: string | null;
  image_url: string;
};

export type SimilarFrameResult = { frame_id: string; image_url: string; scene: Record<string, unknown> | null; score: number };
export type SimilarObjectResult = { object_id: string; frame_id: string; class_name: string; crop_url: string; score: number };
export type SimilarResponse = {
  kind: "frame" | "object";
  mode?: string;
  reason?: string;
  results: Array<Partial<SimilarFrameResult> & Partial<SimilarObjectResult> & { score: number }>;
};

export type DatasetDetail = DatasetRow & {
  slice_spec: Record<string, unknown>;
  files: { path: string; url: string | null }[];
};

export type JobRow = {
  job_id: string;
  kind: string; // import | training | autolabel
  status: string;
  progress: number;
  label: string;
  detail: string;
  link: string;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type ObjectDynamicsRow = { object_id: string; track_id: string | null; distance_m: number | null; lateral_m: number | null; speed_kmh: number | null; closing_speed_kmh: number | null; heading_deg: number | null; ttc_s: number | null; risk_level: string | null; confidence: number };

export type FrameMeta = {
  frame_id: string;
  session_id: string;
  width: number;
  height: number;
  ts_ns: number;
  cam_id: string;
  image_url: string;
  n_objects: number;
  prev_frame_id: string | null;
  next_frame_id: string | null;
  is_lidar?: boolean;
  lidar_points?: number | null;
  lidar_res?: number | null;
};

export type Relationship = { relationship_id: string; from_object_id: string; to_object_id: string; kind: string };
export type AdverseRegion = { region_id: string; frame_id: string; geometry: number[]; condition: string; source: string; confidence: number };

export type FrameObject = {
  object_id: string;
  track_id: string | null;
  class_id: number;
  class_name: string;
  bbox: number[]; // xyxy
  conf: number;
  state: string;
  mask_polygons: number[][];
  version?: number;
  rot_deg?: number;
  keypoints?: Keypoints | null;
  polyline?: number[][] | null;
};

export type TrackItem = {
  object_id: string;
  frame_id: string;
  ts_ns: number;
  class_id: number;
  class_name: string;
  bbox: number[];
  state: string;
  conf: number;
  source?: string;
  is_keyframe?: boolean;
  interp_source?: string | null;
  crop_url: string;
};

export type Track = {
  track_id: string;
  n_frames: number;
  classes: Record<string, number>;
  dominant: string;
  flips: boolean;
  items: TrackItem[];
};

export type Scenario = {
  scenario_id: string;
  session_id: string;
  type: string;
  t_in_ns: number;
  t_out_ns: number;
  actors: string[];
  criticality: number;
  tags: string[];
  meta: Record<string, unknown>;
  city: string | null;
  vehicle_id: string | null;
};
