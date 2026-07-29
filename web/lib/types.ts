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
  source?: string;
  import_format?: string | null;
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
  cuboid_3d?: { center: number[]; size: number[]; yaw: number } | null;
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
  origin?: string;
};

export type SegmentResult = { polygons: number[][]; bbox: number[] | null };

// ---- Integrations ----

export type WebhookRow = {
  webhook_id: string; url: string; events: string[]; active: boolean;
  last_status: number | null; last_error: string | null; failure_count: number;
  last_delivery_at: string | null;
};

export type StorageSourceRow = {
  source_id: string; name: string; provider: string; bucket: string; prefix: string | null;
  uri: string; credential_profile: string | null; last_object_count: number | null;
};

// ---- Multi-modal project spine (Asset -> Annotation), Label-Studio style ----

export type LabelDef = { name: string; color: string | null; kinds: string[] };
export type FieldDef = { name: string; type: "enum" | "float" | "int" | "bool" | "text"; values: string[]; required: boolean };
export type LabelConfig = { labels?: LabelDef[]; fields?: FieldDef[]; allow_kinds?: string[]; media?: string };

export type AssetRow = {
  asset_id: string; project_id: string; media_type: string;
  uri: string | null; text: string | null; external_id: string | null;
  frame_id: string | null; session_id: string | null;
  meta: Record<string, unknown>; state: string; created_at: string | null;
};

export type AnnotationRow = {
  annotation_id: string; asset_id: string; kind: string; label: string | null;
  payload: Record<string, unknown>; fields: Record<string, unknown>;
  conf: number | null; source: string; state: string; version: number;
  provenance: Record<string, unknown>; created_at: string | null;
};

export type AssetDetail = AssetRow & { label_config: LabelConfig; annotations: AnnotationRow[] };

// ---- Labeling operations (CVAT-style projects, jobs, issues, scorecards) ----

export type LabelProjectRow = {
  project_id: string; name: string; description: string | null; modality: string;
  honeypot_frac: number; min_honeypot_accuracy: number; gold_id: string | null;
  label_config: Record<string, unknown>; created_at: string | null;
};

// stage is where in the pipeline the work sits; state is how far along it is within that stage.
export type LabelJobRow = {
  job_id: string; task_id: string; assignee_id: string | null;
  stage: string; state: string; version: number;
  n_frames: number; frame_ids: string[]; n_honeypots: number;
  honeypot_accuracy: number | null; honeypot_detail: Record<string, unknown>;
  started_at: string | null; submitted_at: string | null; created_at: string | null;
};

export type BoardCell = { stage: string; state: string; count: number };

export type IssueComment = { comment_id: string; body: string; author: string | null; created_at: string | null };
export type IssueRow = {
  issue_id: string; kind: string; status: string;
  object_id: string | null; frame_id: string | null; job_id: string | null;
  region: number[] | null; created_at: string | null; resolved_at: string | null;
  comments?: IssueComment[]; n_comments?: number;
};

export type ScorecardRow = {
  user_id: string; name: string; role: string; reviews: number;
  total_time_min: number; mean_time_ms: number; median_time_ms: number;
  jobs: number; honeypot_accuracy: number | null;
};

// ---- Explore workspace (embeddings map, facets, tags, saved views, eval drill-down) ----

// The predicate is the same shape a CurationSlice stores, so a filter built in the explorer saves as a
// cohort and exports unchanged (services/explore/query.py documents the full vocabulary).
export type ExplorePredicate = {
  weather?: string[]; time_of_day?: string[]; road_type?: string[]; density?: string[];
  cities?: string[]; class_names?: string[]; states?: string[]; sources?: string[];
  min_conf?: number; max_conf?: number; tags?: string[]; frame_tags?: string[];
  session_id?: string; object_ids?: string[]; frame_ids?: string[];
};

export type FacetBucket = { value: string; count: number; class_id?: number; lo?: number; hi?: number };
export type Facets = {
  total: number;
  classes: FacetBucket[]; states: FacetBucket[]; sources: FacetBucket[]; cities: FacetBucket[];
  scene: Record<string, FacetBucket[]>; conf: FacetBucket[]; tags: FacetBucket[];
};

export type ProjectionRow = {
  projection_id: string; kind: string; space: string; method: string; n: number;
  session_id: string | null; params: Record<string, unknown>; notes: string | null; created_at: string | null;
};
export type ProjectionPoint = {
  id: string; x: number; y: number; cluster: number;
  class_id?: number; state?: string; source?: string; conf?: number | null; tags?: string[];
  frame_id?: string; session_id?: string; quality?: number | null; scene?: Record<string, string>;
};
export type ProjectionPoints = {
  projection_id: string; kind: string; space?: string; method: string; n?: number; points: ProjectionPoint[];
};

export type SavedView = {
  slice_id: string; name: string; predicate: ExplorePredicate; description: string | null;
  created_at: string | null;
};

export type EvalRun = { eval_id: string; gold_id: string | null; tp: number; fp: number; fn: number; created_at: string | null };
export type ConfusionCell = {
  gt_class_id: number | null; pred_class_id: number | null; gt_class: string | null; pred_class: string | null;
  outcome: string; count: number;
};
export type EvalPatchRow = {
  patch_id: string; object_id: string | null; frame_id: string | null; outcome: string;
  gt_class_id: number | null; pred_class_id: number | null; iou: number | null; conf: number | null;
  crop_url: string | null;
};

export type UserRow = { user_id: string; name: string; role: string; reviews: number };
// Create/re-issue returns the signed token once (not present on the list endpoint).
export type UserCreated = UserRow & { token: string };

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

export type AlItem = { object_id: string; frame_id: string; class_name: string; conf: number; quality_score?: number | null; value: number; scores: { uncertainty: number; diversity: number; rarity: number; error_prone: number } };
export type ErrorCandidateRow = { candidate_id: string; object_id: string; kind: string; score: number; proposed_label: { class_name?: string } | null; detail: Record<string, unknown>; status: string };
export type GovState = { loop_enabled: boolean; auto_accept_enabled: boolean; auto_promote_enabled: boolean; champion_version: string | null; paused_reason: string | null; updated_at: string | null };
export type RegistryRow = { model_version: string; task: string; is_champion: boolean; promoted_from: string | null; gold_metrics: Record<string, unknown>; dataset_commit: string | null; created_at: string | null };
export type AuditRow = { audit_id: string; actor: string; decision: string; subject: string | null; rationale: Record<string, unknown>; created_at: string | null };

export type AssignmentRow = { assignment_id: string; item_id: string; user: string; branch: string; status: string };
export type MergeRequestRow = { mr_id: string; title: string; source_branch: string; target_branch: string; status: string; merge_commit: string | null; created_at: string | null };

export type MapCommitRow = { commit_id: string; region: string; element_count: number; session_ids: string[]; formats: Record<string, string>; calibration_version: string | null; created_at: string | null };
export type MapFeature = { type: "Feature"; geometry: { type: string; coordinates: number[] | number[][] } | null; properties: Record<string, unknown> & { element_id: string; kind: string; confidence: number } };
export type MapProvenance = { found: boolean; element_id?: string; kind?: string; attrs?: Record<string, unknown>; confidence?: number; calibration_version?: string | null; commit_id?: string | null; fusion_job_id?: string | null; source_sessions?: string[] | null; source_frames?: { frame_id: string; session_id: string; cam_id: string; ts_ns: number; vehicle_id: string | null }[] };

// M-F.4 productivity + QA analytics
export type ReviewerMetric = { reviewer: string; reviews: number; correction_rate: number | null; avg_review_ms: number; objects_per_hour: number | null; agreement: number | null };
export type ProductivityReport = {
  reviewers: ReviewerMetric[];
  n_reviewers: number;
  interannotator: { shared_objects: number; agreed: number; agreement_rate: number | null };
  trend: { day: number; reviews: number; agreement: number | null }[];
  cost: { human_hours: number; human_cost_usd: number; gpu_hours: number; n_objects: number; n_frames: number; n_auto_accept: number; cost_per_object_usd: number | null; cost_per_frame_usd: number | null; auto_accept_saved_hours: number; auto_accept_saved_usd: number; assumptions: { human_usd_per_hour: number; manual_label_seconds: number } };
  note: string;
};

// M-F.0 explainable auto-labeling
export type ExplainPath = { path: string; label: string; class_name: string | null; conf: number; verdict: string | null; model_version: string | null };
export type ObjectExplanation = {
  object_id: string;
  class_name: string;
  state: string;
  source: string;
  rare: boolean;
  paths: ExplainPath[];
  agreement: boolean;
  mask_box_disagree: boolean;
  vlm: { confirmed: boolean; class_name: string | null; verdict: string | null } | null;
  overruled_classes: string[];
  calibration: { raw: number | null; calibrated: number | null; auto_accept_floor: number | null };
  quality_flags: string[];
  machine_decision: string;
  deciding_reason: string;
  summary: string[];
};

export type MulticamGroups = {
  cameras: string[];
  multicamera: boolean;
  n_groups: number;
  groups: { ts_ns: number; frames: Record<string, { frame_id: string; img_uri: string }> }[];
};

// M-MC.0 persisted frame groups
export type FrameGroup = {
  group_id: string;
  ts_ns: number;
  frame_ids: Record<string, string>;   // cam_id -> frame_id
  missing_cams: string[];
  sync_spread_ns: number;
  n_cams: number;
  confirmed: boolean;
};
export type PersistedGroups = { session_id: string; cameras: string[]; multicamera: boolean; n_groups: number; groups: FrameGroup[] };

// M-MC.2 rig identity
export type RigMember = { object_id: string; cam: string; class_id: number | null; class_name: string; state: string };
export type RigObjectItem = { rig_object_id: string; class_id: number | null; class_name: string | null; conflict: boolean; cameras: string[]; members: RigMember[] };
export type RigObjectsResponse = { group_id: string; rig_objects: RigObjectItem[]; singletons: RigMember[] };
export type LinkSuggestion = { a: string; b: string; cam_a: string; cam_b: string; class_a: number | null; class_b: number | null; cos: number };
export type SuggestResponse = { group_id: string; suggestions: LinkSuggestion[]; appearance_cos: number };

// M-MC.4 rig tracks
export type RigTrackRow = { rig_track_id: string; instants: number; cameras: string[]; ts_start: number | null; ts_end: number | null; class_name: string | null; inconsistent: boolean };
export type RigTracksResponse = { session_id: string; n_tracks: number; tracks: RigTrackRow[] };
export type RigTrackInstant = { rig_object_id: string; group_id: string; ts_ns: number | null; class_name: string | null; conflict: boolean; cameras: string[]; members: { object_id: string; cam: string | null; class_name: string }[] };
export type RigTrackTimeline = { rig_track_id: string; n_instants: number; instants: RigTrackInstant[] };
export type ConsistencyResult = { session_id: string; n_tracks: number; inconsistent_objects: number };

export type CalibFovCheck = { implied_fov_deg: number; expected_fov_deg: number | null; diff_deg: number | null; tolerance_deg: number; ok: boolean };
export type CalibCamera = { cam_id: string; model: string; lens?: string; reproj_error_px: number | null; fov_check: CalibFovCheck; time_offset_ns: number | null; status: string };
export type CalibDetail = { session_id: string; cameras_in_session: string[]; validations: CalibCamera[]; overall: string };
export type CalibSession = { session_id: string; vehicle_id: string; cameras: number; fail: number; overall: string };
export type ResolvedCalibCam = { cam_id: string; source: string; quality: number; fx: number; fy: number; cx: number; cy: number; pitch_deg: number; yaw_deg: number; height_m: number };
export type CalibTrust = { level: string; mean_quality: number; n_cameras: number };
export type CalibResolved = { session_id: string; cameras: ResolvedCalibCam[]; trust: CalibTrust };
// Plane-4 inertial (M-IMU.1/.3/.4)
export type EgoSample = { ts_ns: number; speed_mps: number | null; heading_deg: number | null; yaw_rate: number | null; long_accel: number | null; lat_accel: number | null; jerk: number | null };
export type EgoState = { session_id: string; source: string; n_samples: number; n_with_motion: number; series: EgoSample[] };
export type InertialEvent = { kind: string; t_in_ns: number; t_out_ns: number; peak: number; severity: number };
export type InertialAnomaly = { ts_ns: number; metric: string; value: number; z: number; status: string };
export type Maneuver = { kind: string; t_in_ns: number; t_out_ns: number };
export type InertialEvents = { session_id: string; source: string; n_samples: number; events: InertialEvent[]; anomalies: InertialAnomaly[]; maneuvers: Maneuver[] };

export type LaneRow = {
  lane_id: string;
  frame_id: string;
  track_ref: string | null;
  control_points: number[][];
  lane_type: string;
  is_ego: boolean;
  source: string;
  model_version: string | null;
  // How strongly the paint supported the type. Null means nobody measured it, which is not the same as
  // measured and uncertain: every lane written before the classifier existed carries a hardcoded default.
  marking_conf?: number | null;
  measured?: boolean;
};

export type LaneTypeResult = {
  session_id: string;
  lanes: number;
  measured?: number;
  changed_type?: number;
  unreadable_frames_lanes?: number;
  frames_decoded?: number;
  by_type?: Record<string, number>;
  mean_confidence?: number | null;
  distinct_types?: number;
  dry_run?: boolean;
  detail?: string;
};

export type LaneTypeCoverage = {
  total: number;
  measured: number;
  unmeasured: number;
  by_type: Record<string, { count: number; mean_confidence: number | null }>;
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
  has_mcap?: boolean;
  annotation_source?: string | null;
  import_format?: string | null;
  prev_frame_id: string | null;
  next_frame_id: string | null;
  is_lidar?: boolean;
  lidar_points?: number | null;
  lidar_res?: number | null;
};

export type Relationship = { relationship_id: string; from_object_id: string; to_object_id: string; kind: string };
export type AdverseRegion = { region_id: string; frame_id: string; geometry: number[]; condition: string; source: string; confidence: number };
export type ProjectedCuboid = { object_id: string; corners_uv: number[][]; edges: number[][]; any_in_image: boolean };

export type FrameObject = {
  object_id: string;
  track_id: string | null;
  class_id: number;
  class_name: string;
  bbox: number[]; // xyxy
  conf: number;
  quality_score?: number | null;
  state: string;
  mask_polygons: number[][];
  version?: number;
  rot_deg?: number;
  keypoints?: Keypoints | null;
  polyline?: number[][] | null;
  cuboid_3d?: { center: number[]; size: number[]; yaw: number } | null;
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

export type TrackIntent = { intent: string; kind: string; source: string; status: string; confidence: number; evidence: Record<string, unknown> };
export type Track = {
  track_id: string;
  n_frames: number;
  classes: Record<string, number>;
  dominant: string;
  flips: boolean;
  items: TrackItem[];
  intents: TrackIntent[];
};
export type IntentVocab = { ontology_version: string; vru: string[]; vehicle: string[]; trajectory_vru: string[]; trajectory_vehicle: string[]; vlm_vru: string[] };

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

// Warm cloud-GPU session (the connect/disconnect control). Mirrors the backend status snapshot.
export type CloudStatus = {
  state: string;            // disconnected | provisioning | connected | running_job | pausing | terminating
  connected: boolean;
  pod_id: string | null;
  gpu_type: string | null;
  uptime_s: number;
  gpu_seconds: number;
  est_cost: number;
  hourly_usd: number;
  idle_remaining_s: number | null;
  session_remaining_s: number | null;
  last_job_id: string | null;
  cold_start_s: number;
  configured: boolean;      // is RUNPOD_API_KEY set on the backend
};
export type CloudOrphan = { pod_id: string; gpu_type: string | null; uptime_s: number; est_cost: number };

// LabeloxSec: the security domain. These mirror services/api/routers/security.py, which is capability-gated
// rather than role-gated alone: reading a registration mark is lawful for an authorised security deployment
// and is exactly what the AV pack must never do, because under DPDPA a plate is personal data the privacy
// plane blurs.
export type SecPack = {
  pack_id: string;
  name: string;
  capabilities: string[];
  anpr_authorised: boolean;
  static_camera: boolean;
  safety_classes: string[];
  available_packs: string[];
};

export type SecSession = {
  session_id: string;
  camera_id: string | null;
  city: string | null;
  start_ts_ns: number | null;
  pack_id: string | null;
  plate_reads: number;
};

export type WatchlistEntry = {
  entry_id: string;
  plate: string;            // the normalised mark, which is what matching runs on
  plate_raw: string;
  reason: string | null;
  severity: string;         // info | warn | critical
  active: boolean;
  added_by: string | null;
  created_at: string | null;
};

export type PlateReadRow = {
  read_id: string;
  session_id: string | null;
  frame_id: string | null;
  camera_id: string | null;
  plate: string;
  plate_raw: string;
  plate_type: string;
  state_code: string | null;
  rto_district: string | null;
  valid: boolean;
  det_conf: number;
  // null when the reader exposes no calibrated score. The console says so rather than showing a number that
  // would make a confidence filter look meaningful when it is not.
  ocr_conf: number | null;
  format_confidence: number;
  bbox: number[] | null;
  watchlist_hit: boolean;
  watchlist_severity: string | null;
  created_at: string | null;
};

export type SecStats = {
  reads: number;
  watchlist_hits: number;
  valid_format: number;
  watchlist_size: number;
  unscored_reads: number;
  top_states: Record<string, number>;
};


// The inbox: what happened, who did what, and who looked at personal data. Mirrors services/notify.py,
// services/activity.py, and services/govern/pii_access.py.
export type NotificationRow = {
  notification_id: string;
  kind: string;
  severity: string;              // info | warn | critical
  title: string;
  body: string | null;
  href: string | null;
  subject_type: string | null;
  subject_id: string | null;
  role: string | null;
  user_id: string | null;
  meta: Record<string, unknown>;
  created_at: string | null;
  read: boolean;
};

export type ActivityEvent = {
  event_id: string;
  user_id: string | null;
  user_name: string | null;
  verb: string;
  label: string;                 // the human phrasing, so the feed is not raw identifiers
  subject_type: string | null;
  subject_id: string | null;
  summary: string | null;
  href: string | null;
  meta: Record<string, unknown>;
  created_at: string | null;
};

export type ActivitySummary = {
  hours: number;
  total: number;
  by_verb: Record<string, number>;
  labels: Record<string, string>;
  active_people: number;
};

export type PiiAccessRow = {
  access_id: string;
  user_id: string | null;
  user_name: string | null;
  subject_type: string;
  subject_id: string;
  session_id: string | null;
  action: string;
  pii_kinds: string[];
  // The number a policy is actually written about: viewing a blurred frame is ordinary work, viewing the
  // original is the thing being governed.
  redacted: boolean;
  route: string | null;
  pack_id: string | null;
  created_at: string | null;
};

export type Profile = {
  user_id: string;
  name: string;
  role: string;
  email: string | null;
  has_password: boolean;
  mfa_enabled: boolean;
  sso: boolean;
  sso_issuer: string | null;
  recovery_codes_left: number;
  created_at: string | null;
};

// Campaigns: the improvement loop run by the system. Mirrors services/flywheel/campaign.py.
export type Campaign = {
  campaign_id: string;
  name: string;
  class_name: string;
  task_type: string;
  target_metric: string;
  target_value: number;
  label_budget: number;
  labels_spent: number;
  max_iterations: number;
  patience: number;
  status: string;              // pending | running | blocked | succeeded | exhausted | stopped
  iteration: number;
  stalled_iterations: number;
  best_value: number | null;
  // True by default. A loop that can promote with no person in it is a different product with a
  // different risk profile, so autopilot is opted into one stage at a time.
  require_approval: boolean;
  autopilot_stages: string[];
  created_by: string | null;
  notes: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type CampaignStep = {
  step_id: string;
  iteration: number;
  stage: string;               // mine | judge | label | train | evaluate | promote
  status: string;
  detail: Record<string, unknown>;
  metrics: Record<string, number>;
  awaiting: string | null;
  job_id: string | null;
  started_at: string | null;
  finished_at: string | null;
};

export type CampaignDetail = Campaign & {
  steps: CampaignStep[];
  next_stage: string;
  halt_reason: string | null;
};

export type CampaignTick = {
  campaign: Campaign;
  action: string;              // ran | waiting | awaiting_approval | halted | failed | none
  stage?: string;
  awaiting?: string;
  detail?: string;
  step?: CampaignStep;
};

// The lineage DAG. `rank` is the column a renderer puts a node in, computed server side so the layout is
// the same in every client rather than reinvented per view.
export type LineageNode = {
  id: string;
  kind: string;                // session | dataset | gold | training_job | model | promotion | deployment
  label: string;
  meta: Record<string, unknown>;
  // A node an edge points at that is no longer there. Rendered as a break, because a missing gold set is
  // a fact about the lineage rather than an absence to smooth over.
  incomplete: boolean;
  rank: number;
};

export type LineageGraph = {
  nodes: LineageNode[];
  edges: { source: string; target: string; kind: string }[];
  kinds: string[];
  root: string;
  sessions_shown?: number;
  sessions_truncated?: boolean;
  detail?: string;
};

// Tracklets: a track as a timeline with its keyframes marked.
export type TrackletSample = {
  frame_id: string;
  object_id: string;
  ts_ns: number;
  bbox: number[];
  is_keyframe: boolean;
  source: string;
  state: string;
};

export type Tracklet = {
  track_id: string;
  class_id: number;
  class_name: string;
  first_ts_ns: number;
  last_ts_ns: number;
  length: number;
  keyframes: number;
  // The number an annotator is optimising: how many frames one correction covers.
  frames_per_keyframe: number | null;
  samples: TrackletSample[];
  intents: unknown[];
};

// LabeloxSec v2.
export type CameraZone = {
  zone_id: string;
  camera_id: string;
  session_id: string | null;
  name: string;
  kind: string;                // area | line
  points: number[][];
  rule: string;                // enter | exit | dwell | cross
  dwell_seconds: number | null;
  classes: string[];
  severity: string;
  active: boolean;
  created_by: string | null;
  created_at: string | null;
};

export type SecurityIncident = {
  incident_id: string;
  camera_id: string | null;
  session_id: string | null;
  zone_id: string | null;
  kind: string;
  severity: string;
  title: string;
  summary: string | null;
  start_ts_ns: number;
  end_ts_ns: number;
  duration_s: number;
  evidence: Record<string, unknown>;
  plate: string | null;
  person_identity: string | null;
  status: string;              // open | ack | closed
  acknowledged_by: string | null;
  created_at: string | null;
};

export type PersonIdentityRow = {
  identity_id: string;
  n_tracks: number;
  cameras: string[];
  first_ts_ns: number | null;
  last_ts_ns: number | null;
  // The signature itself is never returned: it is the only thing that could be matched against another
  // system's database, and exporting it would defeat the boundary re-identification is built around.
  signature_dim: number;
};

// Edge feedback: what the field says about what the bench passed.
export type EdgeDeviceRow = {
  device_id: string;
  name: string | null;
  hardware: string | null;
  runtime: string | null;
  artifact_id: string | null;
  model_version: string | null;
  fleet: string | null;
  last_seen_at: string | null;
  live: boolean;
  meta: Record<string, unknown>;
};

export type EdgeFieldReport = {
  artifact_id: string;
  hours: number;
  devices: number;
  live_devices: number;
  windows: number;
  inferences: number;
  dropped_frames: number;
  field: {
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
    worst_throttled_fraction: number;
    temp_c_max: number | null;
  };
  bench: Record<string, unknown>;
  // The product: either number alone is uninteresting, the gap between them is the finding.
  latency_ratio: number | null;
  confidence_drift: number | null;
  findings: { kind: string; severity: string; detail: string }[];
  fleet_significant: boolean;
  min_devices: number;
};

export type EdgeFleet = {
  hours: number;
  artifacts: {
    artifact_id: string;
    windows: number;
    devices: number;
    latency_p95_ms: number | null;
    latency_ratio: number | null;
    findings: number;
    fleet_significant: boolean;
  }[];
  devices: number;
  // A fleet whose devices have gone quiet looks identical to a healthy one in every average computed
  // over the devices still talking.
  silent_devices: number;
};


// The reasoning layer. Mirrors services/autolabel/reasoner/.
//
// A finding carries its own sentence as well as its weight, because every one of these ends up in front
// of a reviewer or an auditor, and a number nobody can read is a number nobody can trust.
export type ReasoningFinding = {
  check: string;               // physics | geometry | temporal | scene | cross_model | corpus_memory
  weight: number;              // positive supports the label, negative argues against it
  detail: string;
  suggests: string | null;
};

export type ReasoningTrace = {
  // accept | review | adjudicate | reject | abstain. Abstain means nothing could be assessed, which is
  // deliberately distinct from having assessed it and found nothing wanting.
  decision: string;
  score: number;
  conflict: number;
  detector_conf: number;
  suggested_class: string | null;
  question: string | null;
  findings: ReasoningFinding[];
};

export type ReasonerAttribution = {
  objects: number;
  reasoned: number;
  reviewed: number;
  decisions: Record<string, number>;
  checks: Record<string, {
    fired_against: number;
    fired_for: number;
    precision_against: number | null;
    precision_for: number | null;
    measured: boolean;
    correct_against: number;
    correct_for: number;
  }>;
  min_samples: number;
  caveat: string;
};

export type ReasonerRerun = {
  session_id: string;
  objects: number;
  frames: number;
  decisions: Record<string, number>;
  would_demote: number;
  auto_accepted: number;
  // The number that matters: dividing by every object understates the intervention by an order of
  // magnitude, because most objects were never auto-accepted in the first place.
  demote_rate_of_auto_accepted: number | null;
  demote_rate_of_all: number;
  applied: number;
  skipped_human_decisions: number;
  dry_run: boolean;
  examples: {
    object_id: string; frame_id: string; class_name: string; state: string;
    decision: string; suggested_class: string | null; reasons: string[];
  }[];
  truncated_examples: number;
};

// ---- Driving events ----------------------------------------------------------------------------------

export type EventKindSpec = {
  kind: string;
  modality: string;
  shape: "interval" | "point";
  anchor: "track" | "frame" | "session";
  severity: "info" | "notable" | "violation";
  derived: boolean;
  description: string;
  payload: string[];
};

export type EventTaxonomy = {
  version: string;
  kinds: EventKindSpec[];
  severities: string[];
  signal_phase_graph: Record<string, string[]>;
};

export type DrivingEvent = {
  event_id: string;
  session_id: string;
  kind: string;
  modality: string;
  t_start_ns: number;
  t_end_ns: number | null;
  track_id: string | null;
  frame_id: string | null;
  conf: number | null;
  severity: string;
  // Set when a person ruled on this and the evidence underneath has since changed. Their verdict stands;
  // this says the premise moved, so somebody can go and look.
  evidence_changed?: { was: string; would_now_be: string } | null;
  payload: Record<string, unknown>;
  source: string;
  state: string;
  version: number;
};

export type DrivingEventList = {
  session_id?: string;
  track_id?: string;
  count: number;
  // The session origin, so an absolute capture timestamp can be shown as an offset into the drive.
  session_start_ns?: number | null;
  events: DrivingEvent[];
};

export type DrivingEventSummary = {
  session_id: string;
  total: number;
  by_kind: Record<string, number>;
  by_state: Record<string, number>;
  by_severity: Record<string, number>;
  violations: number;
};

export type DeriveResult = {
  session_id: string;
  derived: number;
  inserted: number;
  updated: number;
  unchanged: number;
  pruned_stale: number;
  skipped_reviewed: number;
  rejected_by_taxonomy: number;
  by_kind: Record<string, number>;
};

export type DerivePreview = {
  session_id: string;
  derived: number;
  by_kind: Record<string, number>;
  events: Record<string, unknown>[];
  truncated: number;
};

export type LaneLinkResult = {
  session_id: string;
  lanes: number;
  already_linked?: number;
  linked: number;
  identities: number;
  multi_frame_identities: number;
  frame_width?: number;
  dry_run: boolean;
  detail?: string;
};

export type SegmentEditResult = {
  frame_id: string;
  kind: string;
  source: string;
  coverage: Record<string, number>;
  labelled_fraction: number;
  replaced: boolean;
  unknown_classes: string[];
};
