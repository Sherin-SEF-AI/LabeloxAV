// LabeloxAV API types, generated from the server's OpenAPI schema. Do not edit by hand.
//
// Regenerate with:  python -m scripts.generate_ts_types --out web/lib/api.generated.ts
//
// A response typed `unknown` is one the server has not declared a model for. Declare it on the route
// (a return annotation or response_model) and regenerate; do not cast it away here.

/* eslint-disable */

export type AboutOut = {
  version: string | null;
  commit: string | null;
  source_url: string;
  source_of_this_version: string | null;
  modified: boolean;
  own_licence: string;
  combined_licence: string;
  copyleft: CopyleftOut[];
  licensing_doc: string;
};

export type AcceptTrackIn = {
  state?: string | null;
  origin_object_id?: string | null;
  time_spent_ms?: number;
  reviewer?: string;
};

export type AdaptiveIn = {
  regressions: Record<string, unknown>[];
  odd_gaps: Record<string, unknown>[];
  total_label_budget: number;
  total_fleet_samples: number;
  safety_slices?: string[] | null;
  safety_floor?: number;
};

export type AdverseIn = {
  geometry: number[];
  condition: string;
  confidence?: number;
};

export type AgentPolicyIn = {
  auto_accept_conf?: number | null;
  review_low?: number | null;
  require_agreement?: boolean | null;
};

export type AggLabelIn = {
  center: number[];
  dims: number[];
  yaw?: number;
  class_id: number;
};

export type AggregateIn = {
  session_ids: string[];
  region?: string | null;
  compute_target?: string;
};

export type AgreementIn = {
  annotations: Record<string, unknown>[];
  iou_thr?: number;
};

export type AnnotationIn = {
  kind: string;
  label?: string | null;
  payload?: Record<string, unknown>;
  fields?: Record<string, unknown>;
  conf?: number | null;
  source?: string;
  state?: string;
  provenance?: Record<string, unknown>;
};

export type AnnotationPatch = {
  label?: string | null;
  payload?: Record<string, unknown> | null;
  fields?: Record<string, unknown> | null;
  state?: string | null;
  expected_version?: number | null;
};

export type ApproveIn = {
  name: string;
  l0?: string;
  l1?: string;
};

export type AskIn = {
  text: string;
  limit?: number;
};

export type AssetsIn = {
  project_id: string;
  items: Record<string, unknown>[];
};

export type AttachIn = {
  job_id: string;
  label?: string | null;
  baseline_run_id?: string | null;
  notes?: string | null;
};

/** What the sign-in page draws. Configuration only: nothing here says whether an account exists. */
export type AuthMethodsOut = {
  password: boolean;
  self_signup: boolean;
  signup_domains: string[];
  oidc: boolean;
  oidc_issuer: string | null;
  dev_login: boolean;
  bootstrap: boolean;
};

export type AutolabelJobOut = {
  job_id: string;
  session_id: string;
  status: string;
  executor: string;
  attempts: number;
  progress: number;
  counts: Record<string, unknown> | null;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
  queue_position: number | null;
};

export type AutolabelStartIn = {
  session_id: string;
  limit?: number | null;
  compute_target?: string;
  only_unlabelled?: boolean;
};

export type BatchCorrectIn = {
  object_3d_ids: string[];
  class_id?: number | null;
  dims?: number[] | null;
};

export type BatchFixIn = {
  object_ids: string[];
  to_class: number;
};

export type BenchIn = {
  model_version: string;
  target: string;
  latency_ms: Record<string, unknown>;
  throughput_fps?: number | null;
  power_w?: number | null;
  artifact_uri?: string | null;
};

export type BenchmarkIn = {
  session_ids?: string[] | null;
  rate?: number;
};

export type BevProjectIn = {
  points: unknown[];
  lane_type?: string;
  is_ego?: boolean;
};

export type BootstrapIn = {
  values: number[];
  n_boot?: number;
  alpha?: number;
  seed?: number;
};

export type BulkReviewIn = {
  object_ids: string[];
  action?: string;
  class_name?: string | null;
  attrs?: Record<string, unknown> | null;
  state?: string | null;
  reviewer?: string;
  time_spent_ms?: number;
  expected_versions?: Record<string, number> | null;
};

export type BulkReviewOut = {
  updated: number;
  action: string;
  skipped_missing: string[];
  skipped_stale: StaleObjectOut[];
  run_id: string | null;
};

export type BulkVerdictIn = {
  candidate_ids: string[];
  verdict: string;
  note?: string | null;
  apply_proposed?: boolean;
  reviewer?: string;
};

export type BuyerSpecIn = {
  text: string;
  name?: string | null;
  confirm?: boolean;
};

export type CalibrateFitIn = {
  gold_id?: string | null;
  session_id?: string | null;
};

export type CampaignIn = {
  name: string;
  class_name: string;
  target_metric?: string;
  target_value?: number;
  label_budget?: number;
  max_iterations?: number;
  patience?: number;
  task_type?: string;
  require_approval?: boolean;
  autopilot_stages?: string[];
  notes?: string | null;
};

export type CandidateStateIn = {
  state: string;
  tag?: string | null;
};

export type ChangePasswordIn = {
  current_password?: string | null;
  new_password: string;
};

export type ChatIn = {
  messages: Message[];
  session_id?: string | null;
  frame_id?: string | null;
};

/** One line of `scripts/doctor.sh`: services.doctor.Check, field for field. */
export type CheckOut = {
  name: string;
  status: string;
  detail: string;
  fix: string;
  group: string;
};

/** A save-as. The name is what a person will scan a list for, so it is required and the note is not. */
export type CheckpointIn = {
  name: string;
  note?: string | null;
};

export type ChecksOut = {
  status: string;
  checks: CheckOut[];
};

/** The class a create call landed on. `existed` is true when the name was already taken and the */
export type ClassCreatedOut = {
  id: number;
  name: string;
  l0: string;
  l1: string;
  india: boolean;
  existed: boolean;
};

/** One class and the regions the annotator painted for it, flattened [x,y,x,y,...] per ring. */
export type ClassPolygons = {
  class_name: string;
  polygons?: number[][];
};

export type ClassTemplateOut = {
  name: string;
  class_ids: number[];
  all: boolean;
};

export type ClassTemplatesOut = {
  templates: ClassTemplateOut[];
};

export type ClassesIn = {
  template?: string | null;
  class_ids?: number[] | null;
};

export type ClassifyIn = {
  frame_id: string;
  box: number[];
};

export type CleanupIn = {
  do_pii?: boolean;
  pii_limit?: number;
};

export type CliqueRewardIn = {
  clique: string;
  allocated: number;
  recall_before?: number | null;
  recall_after?: number | null;
};

export type CodeIn = {
  code: string;
  mfa_handle?: string | null;
};

export type CommandIn = {
  text: string;
  frame_id: string;
};

export type CommentIn = {
  body: string;
};

export type CommitWorkIn = {
  labels: Record<string, Record<string, unknown>>;
  message?: string;
};

export type ComposeIn = {
  polygons?: number[][];
  ops?: Record<string, unknown>[];
  width: number;
  height: number;
};

export type ConfirmIn = {
  apply_proposed?: boolean;
  reviewer?: string;
  user_id?: string | null;
};

export type ConnectIn = {
  ack_hourly_usd: number;
};

export type ConsensusIn = {
  fusion: Detection;
  auto_paths: Detection[];
  fusion_run_id?: string | null;
};

export type ConsentIn = {
  session_id: string;
  consent_status: string;
  legal_basis?: string | null;
  retention_until?: string | null;
};

export type CooptIn = {
  target: string;
  base_latency_ms: number;
  base_map50: number;
  prune_grid?: number[] | null;
};

export type CopyleftOut = {
  name: string;
  licence: string;
  where: string;
  why: string;
};

export type CostGateIn = {
  gpu_hours: number;
  hourly_usd: number;
  spent_usd?: number;
  per_job_cap_usd?: number | null;
  window_cap_usd?: number | null;
};

export type CreateObjectIn = {
  class_name: string;
  bbox: number[];
  attrs?: Record<string, unknown>;
  mask_polygons?: number[][] | null;
  state?: string;
  idem_key?: string | null;
  rot_deg?: number;
  keypoints?: Record<string, unknown> | null;
  bbox_amodal?: number[] | null;
  polyline?: number[][] | null;
  cuboid_3d?: Record<string, unknown> | null;
  job_id?: string | null;
};

export type CropSheetIn = {
  object_ids: string[];
  cell?: number;
  pad?: number;
  cols?: number;
};

export type CrossCamIn = {
  tol_ms?: number;
  high?: number;
  min_vis?: number;
};

export type Cuboid3DEdit = {
  class_id?: number | null;
  center?: number[] | null;
  dims?: number[] | null;
  yaw?: number | null;
  pitch?: number | null;
  roll?: number | null;
  attrs?: Record<string, unknown> | null;
  ground_snap?: boolean;
  expected_version?: number | null;
};

export type Cuboid3DIn = {
  class_id: number;
  center: number[];
  dims: number[];
  yaw?: number;
  pitch?: number;
  roll?: number;
  attrs?: Record<string, unknown>;
  object_id?: string | null;
  ground_snap?: boolean;
};

export type CuboidAtIn = {
  u: number;
  v: number;
  class_name: string;
};

export type CuboidIn = {
  min_iou?: number;
  high?: number;
};

/** A cuboid's eight corners projected onto the camera image, so it can be drawn over the 2D frame. */
export type CuboidProjectionOut = {
  object_id: string;
  corners_uv: number[][];
  edges: number[][];
  any_in_image: boolean;
};

export type CycleIn = {
  max_frames?: number;
  dry_run?: boolean;
  retrain?: boolean;
};

/** One dataset with its files. Not the list row plus two fields: the detail has no `n_files`, since */
export type DatasetDetailOut = {
  commit_id: string;
  name: string | null;
  object_count: number;
  formats: string[];
  ontology_version: string;
  created_at: string | null;
  slice_spec: Record<string, unknown> | null;
  files: DatasetFileOut[];
};

export type DatasetFileOut = {
  path: string;
  url: string | null;
  missing: boolean;
};

/** One sealed dataset in the list. `name` and `formats` are read out of the commit's slice spec. */
export type DatasetOut = {
  commit_id: string;
  name: string | null;
  object_count: number;
  formats: string[];
  ontology_version: string;
  n_files: number;
  created_at: string | null;
};

export type DatasheetIn = {
  gold_id?: string | null;
  title?: string | null;
};

export type DescribeIn = {
  phrase: string;
  class_id?: number | null;
  max_proposals?: number | null;
};

export type DetectIn = {
  session_id?: string | null;
  kinds?: string[] | null;
};

export type Detection = {
  path: string;
  class_id: number;
  bbox: number[];
  conf?: number;
};

export type DeviceIn = {
  device_id: string;
  name?: string | null;
  hardware?: string | null;
  runtime?: string | null;
  artifact_id?: string | null;
  model_version?: string | null;
  fleet?: string | null;
  meta?: Record<string, unknown>;
};

export type DisableMfaIn = {
  password: string;
};

export type DisconnectIn = {
  pause?: boolean;
};

export type DriftInvestigateIn = {
  ref_sessions?: string[] | null;
  cur_sessions?: string[] | null;
};

export type DrivableIn = {
  classes: Record<string, unknown>;
  width: number;
  height: number;
};

/** A human-authored driving event. Validated against the taxonomy before it is written. */
export type DrivingEventIn = {
  kind: string;
  t_start_ns: number;
  t_end_ns?: number | null;
  track_id?: string | null;
  frame_id?: string | null;
  payload?: Record<string, unknown>;
};

export type EfficiencyIn = {
  entries: Record<string, unknown>[];
};

export type EngageIn = {
  reason?: string;
  task?: string;
};

export type EnqueueIn = {
  item_ids: unknown[];
  user_id: string;
};

export type EraseIn = {
  session_id: string;
  reason?: string;
  dry_run?: boolean;
};

export type ErrorSweepIn = {
  max_sessions?: number;
  kinds?: string[] | null;
};

export type EventEdit = {
  kind?: string | null;
  t_start_ns?: number | null;
  t_end_ns?: number | null;
  payload?: Record<string, unknown> | null;
  state?: string | null;
  expected_version?: number | null;
};

export type EventIn = {
  kind: string;
  modality: string;
  t_start_ns: number;
  t_end_ns?: number | null;
  payload?: Record<string, unknown>;
};

export type ExperimentIn = {
  name: string;
  task_type?: string;
  description?: string | null;
  hypothesis?: string | null;
  tags?: string[];
};

/** One hypothetical detection, for trying the reasoner without running a session. */
export type ExplainIn = {
  class_name: string;
  bbox: number[];
  conf?: number;
  frame_w?: number;
  frame_h?: number;
  depth_m?: number | null;
  focal_px?: number | null;
  scene?: Record<string, unknown>;
  proposals?: Record<string, unknown>[];
};

export type Export3DIn = {
  session_ids?: string[];
  states?: string[];
  class_names?: string[];
  min_conf?: number;
  formats?: string[];
  export_clouds?: boolean;
};

export type ExternalModelIn = {
  model_version: string;
  task?: string;
  notes?: string | null;
};

export type ExternalRunIn = {
  model_version: string;
  predictions: Record<string, unknown>[];
  gold_id?: string | null;
  code_sha?: string;
  params?: Record<string, unknown> | null;
};

export type FeaturesOut = {
  optional_packs: Record<string, boolean>;
  gpu_worker: boolean;
  autolabel_executor: string;
  auth_enabled: boolean;
};

export type FilmstripOut = {
  frame_id: string;
  cam_id: string;
  frames: FilmstripTileOut[];
};

export type FilmstripTileOut = {
  frame_id: string;
  ts_ns: number;
  n_objects: number;
  n_confirmed: number;
  image_url: string;
  current: boolean;
};

export type FindOut = {
  q: string;
  results: FindResultOut[];
};

export type FindResultOut = {
  kind: string;
  id: string;
  label: string;
  detail: string;
  href: string;
};

export type FirstFrameOut = {
  frame_id: string;
};

export type FitCuboidIn = {
  contact?: number[] | null;
};

export type FitThresholdsIn = {
  run_id: string;
  min_support?: number | null;
  n_boot?: number | null;
  activate?: boolean;
};

export type FlywheelIn = {
  auto_accept_conf?: number | null;
  review_low?: number | null;
  require_agreement?: boolean | null;
  ticks?: number;
  max_frames?: number;
  session_id?: string | null;
  dry_run?: boolean;
};

/** Frame-level facts a person is asserting. Partial: only the keys sent are touched, so setting the */
export type FrameContextIn = {
  attrs: Record<string, unknown>;
};

export type FrameContextOut = {
  frame_id: string;
  context: Record<string, unknown>;
  context_provenance: Record<string, unknown>;
};

/** One object in the editor's frame list (GET /frames/{id}/objects). */
export type FrameObjectOut = {
  object_id: string;
  track_id: string | null;
  class_id: number;
  class_name: string;
  bbox: number[];
  conf: number;
  quality_score: number | null;
  state: string;
  mask_polygons: number[][];
  version: number;
  rot_deg: number;
  bbox_amodal: number[] | null;
  keypoints: Record<string, unknown> | null;
  polyline: number[][] | null;
  cuboid_3d: Record<string, unknown> | null;
};

/** Frame meta for the editor (GET /frames/{id}). */
export type FrameOut = {
  frame_id: string;
  session_id: string;
  width: number;
  height: number;
  ts_ns: number;
  cam_id: string;
  image_url: string;
  n_objects: number;
  has_mcap: boolean;
  annotation_source: string | null;
  import_format: string | null;
  prev_frame_id: string | null;
  next_frame_id: string | null;
  is_lidar: boolean;
  lidar_points: number | null;
  lidar_res: number | null;
  blind_audit_id: string | null;
  context: Record<string, unknown>;
  context_provenance: Record<string, unknown>;
};

export type FreshIn = {
  auto_accept_conf?: number | null;
  review_low?: number | null;
  require_agreement?: boolean | null;
  commit?: boolean;
};

export type GateBatchIn = {
  project_id: string;
  budget?: number;
  jobs_of?: number;
};

export type GateIn = {
  baseline_latency: Record<string, unknown>;
  candidate_latency: Record<string, unknown>;
  champion_eval: Record<string, unknown>;
  candidate_eval: Record<string, unknown>;
  protected_slices: string[];
};

export type GoldSealIn = {
  name?: string;
  cities?: string[] | null;
  session_id?: string | null;
  class_names?: string[] | null;
  limit?: number | null;
};

export type HTTPValidationError = {
  detail?: ValidationError[];
};

export type HeartbeatIn = {
  tab: string;
  acquire?: boolean;
};

export type HomeCorpusOut = {
  sessions: number;
  frames: number;
  objects: number;
  by_state: Record<string, number>;
  waiting: number;
  settled: number;
};

export type HomeFailedJobOut = {
  kind: string;
  job_id: string;
  error: string;
  at: string | null;
};

export type HomeJobsOut = {
  active: Record<string, number>;
  active_total: number;
  running: number;
  failed: HomeFailedJobOut[];
  failed_window_days: number;
};

export type HomeOut = {
  corpus: HomeCorpusOut;
  jobs: HomeJobsOut;
  recent_sessions: HomeRecentSessionOut[];
  waiting_for_people: HomeWaitingOut[];
  steps: HomeStepsOut;
};

export type HomeRecentSessionOut = {
  session_id: string;
  vehicle_id: string | null;
  city: string | null;
  start_ts_ns: number;
  frames: number;
  objects: number;
  waiting: number;
};

export type HomeStepsOut = {
  data: boolean;
  autolabel: boolean;
  review: boolean;
  export: boolean;
  model: boolean;
};

/** One thing that cannot finish without a person, in one of three kinds that share no field but `kind`. */
export type HomeWaitingOut = {
  kind: string;
  audit_id?: string | null;
  job_id?: string | null;
  frame_id?: string | null;
  n_frames?: number | null;
  n_labeled?: number | null;
  created_at?: string | null;
  count?: number | null;
  href?: string | null;
  lot_id?: string | null;
  class_name?: string | null;
  judged?: number | null;
  of?: number | null;
  population?: number | null;
};

export type IaaIn = {
  set_a: unknown[];
  set_b: unknown[];
  iou_thresh?: number;
};

export type ImportCalibIn = {
  cam_id: string;
  format: string;
  ref_width?: number;
  ref_height?: number;
  calib_text?: string | null;
  camera_intrinsic?: unknown[] | null;
  translation?: unknown[] | null;
  dist?: unknown[] | null;
};

export type ImportFormatsOut = {
  formats: string[];
  annotated: string[];
  raw: string[];
};

export type ImportJobOut = {
  job_id: string;
  status: string;
  format: string;
  source_uri: string | null;
  target_vehicle: string;
  city: string | null;
  progress: number;
  counts: Record<string, unknown> | null;
  error: string | null;
  session_id: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type ImportStartIn = {
  format: string;
  source_uri: string;
  target_vehicle?: string;
  city?: string | null;
  options?: Record<string, unknown>;
  project_id?: string | null;
};

export type IngestIn = {
  model_version: string;
  model_output: Record<string, unknown>[];
  session_ids?: string[] | null;
};

export type IntentSetIn = {
  intent: string;
  kind: string;
  status?: string;
};

export type InviteIn = {
  name: string;
  role?: string;
};

export type IssueIn = {
  kind?: string;
  body?: string | null;
  object_id?: string | null;
  frame_id?: string | null;
  job_id?: string | null;
  region?: unknown[] | null;
};

/** What a cancel answers: one of three shapes, each a 200 (services/job_control.cancel_job). */
export type JobCancelOut = {
  job_id: string;
  canceled: boolean;
  detail: string;
  status?: string | null;
  stopped?: boolean | null;
};

/** One row of the unified jobs list (GET /jobs): import, training, autolabel, export, map_fusion and */
export type JobOut = {
  job_id: string;
  kind: string;
  status: string;
  progress: number;
  label: string;
  detail: string;
  link: string;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type JudgeTubesIn = {
  track_ids?: string[];
  skip_judged?: boolean;
};

export type KeyframeIn = {
  bbox?: number[] | null;
  is_keyframe?: boolean;
};

export type LaneIn = {
  control_points: unknown[];
  lane_type?: string;
  is_ego?: boolean;
  marking_conf?: number | null;
};

export type LayoutIn = {
  name: string;
  panels: unknown[];
  is_default?: boolean;
};

export type LineageRevertIn = {
  from_name: string;
  to_name: string;
  limit?: number | null;
};

export type LinkBody = {
  session_id: string;
  group_id: string;
  object_ids: string[];
  source?: string;
};

export type LintIn = {
  open_issues?: boolean;
};

export type LoginIn = {
  name: string;
  password: string;
};

export type ManeuverIn = {
  trajectory: Record<string, unknown>[];
};

export type MarkAllReadOut = {
  marked: number;
};

/** Two shapes from one route, and no key in common but `marked`: a success names the notification, a */
export type MarkReadOut = {
  marked: boolean;
  notification_id?: string | null;
  reason?: string | null;
};

export type MaskIn = {
  polygons: number[][];
  width?: number | null;
  height?: number | null;
};

export type MaskSavedOut = {
  object_id: string;
  mask_polygons: number[][];
  version: number;
};

/** What adding or removing a member answers: the pair, and whether they are a member now. */
export type MembershipOut = {
  project_id: string;
  user_id: string;
  member: boolean;
};

export type MergeIn = {
  from_id: number;
  to_id: number;
};

export type MergeTrackIn = {
  from_track_id: string;
  force?: boolean;
};

export type Message = {
  role: string;
  content: string;
};

export type MetricElementIn = {
  kind: string;
  points_ego: unknown[];
  ref_lat: number;
  ref_lon: number;
  heading_rad?: number;
};

export type MfaIn = {
  mfa_handle: string;
  code: string;
};

export type MineIn = {
  session_id?: string | null;
  ttc_thresh?: number;
};

export type MintIn = {
  name: string;
  role?: string;
  description?: string | null;
  expires_in_days?: number | null;
  scopes?: string[] | null;
};

/** The runs recorded for one purpose, newest first, and which of them is live. */
export type ModelLineOut = {
  purpose: string;
  task_type: string;
  runs: ModelLineRunOut[];
  promoted: ModelLineRunOut | null;
};

export type ModelLineRunOut = {
  run_id: string;
  dataset_name: string;
  epochs: number;
  map50: number | null;
  safe_miou: number | null;
  promoted: boolean;
  weights_uri: string | null;
  created_at: string | null;
};

export type ModelRunOut = {
  run_id: string;
  dataset_name: string;
  purpose: string;
  epochs: number;
  n_train: number;
  n_val: number;
  promoted: boolean;
  base_weights: string;
  notes: string | null;
  baseline_metrics: Record<string, unknown>;
  metrics: Record<string, unknown>;
  gate: Record<string, unknown>;
  created_at: string | null;
};

export type MonoDepthIn = {
  box: number[];
  class_id: number;
  focal_px: number;
  cy: number;
  cam_height_m?: number | null;
  pitch_rad?: number;
};

export type MoveIn = {
  session_ids: string[];
};

export type MultipartAbortIn = {
  key: string;
  upload_id: string;
};

export type MultipartCompleteIn = {
  key: string;
  upload_id: string;
  parts: Record<string, unknown>[];
};

export type MultipartInitIn = {
  filename: string;
  content_type?: string;
};

export type MultipartSignIn = {
  key: string;
  upload_id: string;
  part_number: number;
};

export type NewClassIn = {
  name: string;
  l0?: string;
  l1?: string;
  india?: boolean;
};

export type NlEditApplyIn = {
  command: string;
  object_ids: string[];
};

export type NlEditPreviewIn = {
  command: string;
  frame_id?: string | null;
  session_id?: string | null;
  use_vlm?: boolean;
};

/** services/notify.py `_dict`, plus whether this caller has read it. */
export type NotificationOut = {
  notification_id: string;
  kind: string;
  severity: string;
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

export type NotificationsOut = {
  total: number;
  offset: number;
  limit: number;
  notifications: NotificationOut[];
};

export type ObjectDeletedOut = {
  deleted: string;
};

export type ObjectDetail = {
  object_id: string;
  frame_id: string;
  session_id: string;
  track_id?: string | null;
  ts_ns: number;
  cam_id: string;
  image_url: string;
  width: number;
  height: number;
  class_id: number;
  class_name: string;
  bbox: number[];
  bbox_amodal?: number[] | null;
  mask_polygons: number[][];
  attrs: Record<string, unknown>;
  conf: number;
  state: string;
  source: string;
  provenance: Record<string, unknown>;
  version?: number;
  rot_deg?: number;
  keypoints?: Record<string, unknown> | null;
  polyline?: number[][] | null;
  cuboid_3d?: Record<string, unknown> | null;
  sign_type?: string | null;
  sign_category?: string | null;
  ocr_text?: string | null;
  ocr_lang?: string | null;
  ocr_conf?: number | null;
};

export type OccupancyBuildIn = {
  t0?: number | null;
  t1?: number | null;
  limit?: number;
  voxel_m?: number | null;
};

export type OddIn = {
  fleet_counts: Record<string, number>;
  target_odd: Record<string, number>;
  min_gap?: number;
};

export type OntologyAttributeOut = {
  type: string;
  values: unknown[] | null;
  range: number[] | null;
  derived_from: string | null;
};

/** A class as GET /ontology lists it: OntologyClassOut plus its aliases. */
export type OntologyClassAliasedOut = {
  id: number;
  name: string;
  l0: string;
  l1: string;
  india: boolean;
  aliases: string[];
};

export type OntologyOut = {
  active_class_ids: number[] | null;
  class_set: "india" | "generic" | "custom";
  version: string;
  hierarchy_levels: number;
  attributes: Record<string, OntologyAttributeOut>;
  classes: OntologyClassAliasedOut[];
  attribute_scope: Record<string, string[]>;
  attribute_scope_class: Record<string, string[]>;
};

export type OntologyPromotion = {
  from_class: string;
  to_class: string;
};

export type OpenMRIn = {
  title: string;
  source_branch: string;
  author_id?: string | null;
  notes?: string | null;
};

export type OpsAskIn = {
  text: string;
  confirm?: boolean;
};

export type OverrideIn = {
  decision: string;
  reason: string;
};

export type ParityIn = {
  learned_metrics: Record<string, unknown>;
  heuristic_metrics: Record<string, unknown>;
  margin?: number | null;
};

export type PendingSetupIn = {
  mfa_handle: string;
};

export type PlanIn = {
  class_names: string[];
  n_frames?: number;
};

export type PlatformOut = {
  id: string;
  label: string;
  role: string;
  backend_prefix: string;
  legacy_prefixes: string[];
  gate: string | null;
  order: number;
  flywheel_stage: number | null;
};

export type PlatformsOut = {
  platforms: PlatformOut[];
};

export type PoseBuildIn = {
  cam_id?: string | null;
  limit?: number | null;
};

/** services.api.identity.public_state: what the profile page may know. No hash, secret or code. */
export type ProfileOut = {
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

export type ProjectChangedOut = {
  project_id: string;
  name: string;
  visibility: string;
  description: string | null;
};

export type ProjectClassesOut = {
  project_id: string;
  class_ids: number[] | null;
  class_set: "india" | "generic" | "custom";
  class_count: number;
  of: number;
};

export type ProjectCreatedOut = {
  project_id: string;
  name: string;
  visibility: string;
};

export type ProjectMemberOut = {
  user_id: string;
  name: string;
  role: string;
  added_at: string | null;
};

export type ProjectMembersOut = {
  project_id: string;
  members: ProjectMemberOut[];
};

/** One project in the list, with what it labels as services.project_classes.describe puts it. */
export type ProjectOut = {
  project_id: string;
  name: string;
  description: string | null;
  visibility: string;
  sessions: number;
  class_set: "india" | "generic" | "custom";
  class_count: number;
  of: number;
  members: number;
  is_member: boolean;
  created_at: string | null;
};

export type ProjectPatch = {
  visibility?: string | null;
  description?: string | null;
};

export type ProjectionIn = {
  kind?: string;
  space?: string;
  session_id?: string | null;
  method?: string;
  n_neighbors?: number;
  min_dist?: number;
  seed?: number;
  min_cluster_size?: number;
  limit?: number;
  notes?: string | null;
};

export type ProjectsOut = {
  projects: ProjectOut[];
};

export type Propagate4dIn = {
  keyframe_box: number[];
  n_frames: number;
  velocity?: number[] | null;
  known?: Record<string, number[]> | null;
};

export type PropagateIn = {
  span?: number;
  drift?: number;
  high?: number;
};

export type RadarIn = {
  track_box: number[];
  camera_velocity: number;
  radar_returns: Record<string, unknown>[];
  ego_speed?: number;
};

export type RankIn = {
  items: Record<string, unknown>[];
};

export type ReanalyzeAllIn = {
  session_id?: string | null;
  max_frames?: number;
};

export type RecognizeIn = {
  frame_id: string;
  regions: number[][];
  camera_id?: string | null;
};

export type ReconcileIn = {
  object_ids?: string[] | null;
  apply?: boolean;
  apply_min_conf?: number;
};

export type RecoverIn = {
  cam_id: string;
  nominal: number[][];
  drift_delta: number[][];
  residual_px: number;
  n_correspondences: number;
  severity?: string;
};

export type RedactionProofIn = {
  release_commit: string;
  frame_ids: string[];
  method_version?: string;
};

export type RegionOcrIn = {
  uri: string;
  bbox: number[];
  pad?: number;
};

export type RegisterIn = {
  model_version: string;
  task?: string;
  gold_metrics: Record<string, unknown>;
  dataset_commit?: string | null;
  weights_uri?: string | null;
  notes?: string | null;
};

export type RelabelAllIn = {
  max_frames?: number;
  session_id?: string | null;
  min_conf?: number;
  margin?: number;
};

export type RelabelIn = {
  min_conf?: number;
  margin?: number;
};

export type RelabelStartIn = {
  model_version: string;
  session_ids?: string[] | null;
  ontology_promotion?: OntologyPromotion | null;
  compute_target?: string;
};

export type RelabelTrackIn = {
  class_name: string;
  state?: string;
  origin_object_id?: string | null;
  force?: boolean;
  reviewer?: string;
  time_spent_ms?: number;
};

export type RelateIn = {
  to_object_id: string;
  kind: string;
};

/** One directed relationship on a frame (the India case is rider_of on a two-wheeler). */
export type RelationshipOut = {
  relationship_id: string;
  from_object_id: string;
  to_object_id: string;
  kind: string;
};

export type ReleaseIn = {
  tab: string;
  leave?: boolean;
};

export type RenameIn = {
  class_id: number;
  new_name: string;
};

export type ReproIn = {
  run_a: Record<string, unknown>;
  run_b: Record<string, unknown>;
};

export type ResetIn = {
  token: string;
  password: string;
};

export type ResetRequestIn = {
  name: string;
};

export type RetentionSweepIn = {
  dry_run?: boolean;
  limit?: number;
};

export type RetrainIn = {
  compute_target?: string;
  force?: boolean;
  base_weights?: string | null;
};

export type ReviewIn = {
  reviewer?: string;
  action: string;
  class_name?: string | null;
  bbox?: number[] | null;
  attrs?: Record<string, unknown> | null;
  state?: string | null;
  time_spent_ms?: number;
  expected_version?: number | null;
  rot_deg?: number | null;
  keypoints?: Record<string, unknown> | null;
  mask_polygons?: number[][] | null;
  polyline?: number[][] | null;
  cuboid_3d?: Record<string, unknown> | null;
  bbox_amodal?: number[] | null;
};

/** What a single review answers with (POST /objects/{id}/review). */
export type ReviewOut = {
  object_id: string;
  class_id: number;
  class_name: string;
  bbox: number[];
  attrs: Record<string, unknown> | null;
  state: string;
  source: string;
  version: number;
  rot_deg: number;
  bbox_amodal: number[] | null;
  keypoints: Record<string, unknown> | null;
};

export type ReviewerIn = {
  reviewer_id: string;
};

export type RolloutIn = {
  current_state: string;
  action: string;
};

export type RtspIn = {
  url: string;
  camera_id: string;
  city?: string | null;
  pack_id?: string;
  max_frames?: number;
  max_seconds?: number;
  motion_threshold?: number;
  heartbeat_seconds?: number;
};

/** A human decision on a candidate. note is free text kept on the event for the next reader. */
export type RulingIn = {
  note?: string | null;
  expected_version?: number | null;
};

export type SafetyIn = {
  objects: Record<string, unknown>[];
  ttc_horizon_s?: number;
  near_miss_ttc_s?: number;
};

export type SeedAuditIn = {
  run_id: string;
  n_frames?: number;
  stratify_by?: string;
  score_thr?: number;
  iou_thr?: number;
  project_id?: string | null;
  notes?: string | null;
};

/** A human edit to a frame's dense raster. */
export type SegmentEditIn = {
  kind?: string;
  classes?: ClassPolygons[];
  replace?: boolean;
};

export type SegmentIn = {
  frame_id: string;
  points?: number[][] | null;
  labels?: number[] | null;
  box?: number[] | null;
  precise?: boolean;
};

export type SelectIn = {
  budget_hours?: number;
  session_id?: string | null;
  dedup_cos?: number;
};

/** One capture session as the session lists serve it (GET /sessions and /sessions/page). */
export type SessionOut = {
  session_id: string;
  vehicle_id: string | null;
  city: string | null;
  route: string | null;
  start_ts_ns: number;
  end_ts_ns: number;
  ontology_version: string;
  origin: string;
  project_id: string | null;
};

export type SessionPageOut = {
  total: number;
  offset: number;
  limit: number;
  sessions: SessionOut[];
};

export type SessionStateOut = {
  session_id: string;
  frames: number;
  objects: number;
  reviewed_objects: number;
};

export type SessionStatsOut = {
  session_id: string;
  frames: number;
  objects: number;
  by_state: Record<string, number>;
  done: number;
  progress: number;
};

export type SessionsMovedOut = {
  project_id: string;
  moved: number;
};

export type SetCalibIn = {
  cam_specs: Record<string, unknown>;
  source?: string;
  ref_width?: number;
  ref_height?: number;
};

export type SignificanceIn = {
  champion: number[];
  challenger: number[];
  n_perm?: number;
  seed?: number;
};

export type SignupIn = {
  name: string;
  password: string;
  email?: string | null;
};

export type SimilarIn = {
  frame_id?: string | null;
  object_id?: string | null;
  image_b64?: string | null;
  mode?: string;
  k?: number;
  min_sim?: number;
  diversity?: boolean;
  same_class?: boolean;
  exclude_track?: boolean;
  city?: string | null;
  session_id?: string | null;
  target?: string;
};

export type SliceIn = {
  name: string;
  predicate?: Record<string, unknown>;
  description?: string | null;
};

export type SloIn = {
  plane: string;
  measurements: Record<string, unknown>;
  window_s?: number;
};

export type SourceIn = {
  name: string;
  provider: string;
  bucket: string;
  prefix?: string | null;
  region?: string | null;
  endpoint_url?: string | null;
  credential_profile?: string | null;
  project_id?: string | null;
};

export type SplitTrackIn = {
  at_ts_ns: number;
};

/** An object a bulk review skipped because it had moved on under the caller. */
export type StaleObjectOut = {
  object_id: string;
  expected: number;
  current: number;
};

export type StateIn = {
  state: string;
  expected_version?: number | null;
};

export type StewardScanIn = {
  min_cluster?: number;
  sample?: number;
  sim_thresh?: number;
};

export type StitchIn = {
  observations: Record<string, number[]>;
  n_frames: number;
  max_heal_gap?: number;
};

export type SubmitIn = {
  expected_version?: number | null;
};

export type SuggestIn = {
  object_id: string;
  kind?: string;
  old_class_name?: string | null;
  new_class_name?: string | null;
  attr_key?: string | null;
  old_value?: unknown | null;
  new_value?: unknown | null;
  filters?: Record<string, unknown>;
  limit?: number;
  threshold?: number;
  same_class?: boolean;
};

export type SweepApplyIn = {
  attr: string;
  value: unknown;
  unit?: string;
  ids: string[];
  overwrite?: boolean;
  state?: string | null;
  time_spent_ms?: number;
  reviewer?: string;
};

export type TabIn = {
  tab: string;
};

export type TagIn = {
  level?: string;
  predicate?: Record<string, unknown>;
  add?: string[];
  remove?: string[];
};

export type TargetlessIn = {
  vp1: number[];
  vp2: number[];
  horizon_y: number;
  cx: number;
  cy: number;
};

export type TaskIn = {
  project_id: string;
  name: string;
  session_id?: string | null;
  predicate?: Record<string, unknown>;
  jobs_of?: number;
  replicas?: number;
};

export type TelemetryIn = {
  device_id: string;
  window_start_ns: number;
  window_end_ns: number;
  n_inferences?: number;
  latency_p50_ms?: number | null;
  latency_p95_ms?: number | null;
  latency_max_ms?: number | null;
  fps?: number | null;
  temp_c_max?: number | null;
  throttled_fraction?: number | null;
  power_w_mean?: number | null;
  conf_histogram?: number[];
  detections_per_frame?: number | null;
  dropped_frames?: number;
  artifact_id?: string | null;
  model_version?: string | null;
  meta?: Record<string, unknown>;
};

export type TemporalRepairIn = {
  session_id?: string | null;
  min_majority?: number;
};

export type TerminatePodIn = {
  pod_id: string;
};

export type ThermalIn = {
  target: string;
  readings: Record<string, unknown>;
  cold_fps: number;
};

export type TrackAttrsIn = {
  attrs: Record<string, unknown>;
};

export type TrackEventIn = {
  event_type: string;
  start_frame_id: string;
  end_frame_id: string;
  notes?: string | null;
  state?: string;
};

export type TrackEventPatch = {
  state?: string | null;
  notes?: string | null;
  start_frame_id?: string | null;
  end_frame_id?: string | null;
};

/** A training job row. The four JSONB columns are passed through as stored: `counts` carries the */
export type TrainingJobOut = {
  job_id: string;
  status: string;
  purpose: string;
  task_type: string;
  compute_target: string;
  stage: string | null;
  progress: number;
  counts: Record<string, unknown> | null;
  metrics: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  error: string | null;
  run_id: string | null;
  config: Record<string, unknown> | null;
  created_at: string | null;
  updated_at: string | null;
};

export type TrainingStartIn = {
  purpose?: string;
  task_type?: string;
  compute_target?: string;
  dataset_spec?: Record<string, unknown>;
  base_weights?: string | null;
  hparams?: Record<string, unknown>;
  gate?: Record<string, unknown>;
  promote?: boolean;
  notes?: string | null;
};

export type TrainingTaskOut = {
  task_type: string;
  default_base_weights: string;
};

/** One reason an object is in the queue, with how loudly it should read. */
export type TriageFlag = {
  code: string;
  label: string;
  severity: string;
};

export type TriageIn = {
  baseline: Record<string, number>;
  current: Record<string, number>;
  margin?: number;
  protected?: string[] | null;
};

export type TriageRow = {
  object_id: string;
  frame_id: string;
  session_id: string;
  class_id: number;
  class_name: string;
  conf: number;
  state: string;
  why: string;
  flags?: TriageFlag[];
  priority: number;
  source?: string;
  import_format?: string | null;
};

export type UncertaintyIn = {
  consensus_score: number;
  n_views?: number;
  calib_confidence?: number;
  depth_uncertainty?: number;
};

export type UnreadCountOut = {
  unread: number;
};

export type UserCreateIn = {
  name: string;
  role?: string;
};

export type UserOut = {
  user_id: string;
  name: string;
  role: string;
  reviews: number;
};

export type UsersOut = {
  total: number;
  offset: number;
  limit: number;
  users: UserOut[];
};

export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
};

export type VerdictIn = {
  verdict: string;
};

export type ViewIn = {
  name: string;
  predicate?: Record<string, unknown>;
  description?: string | null;
};

export type WatchlistIn = {
  plate: string;
  reason?: string | null;
  severity?: string;
};

export type WebhookIn = {
  url: string;
  events?: string[];
  project_id?: string | null;
  secret?: string | null;
};

export type WorkforceIn = {
  name: string;
  kind?: string;
  endpoint?: string | null;
  capabilities?: Record<string, unknown>;
  capacity_jobs_per_day?: number;
  min_honeypot_accuracy?: number;
  contact?: string | null;
};

export type ZoneIn = {
  camera_id: string;
  name: string;
  points: number[][];
  kind?: string;
  rule?: string;
  classes?: string[];
  dwell_seconds?: number | null;
  severity?: string;
  session_id?: string | null;
  pack_id?: string;
};

export type services__api__deps__ExportIn = {
  name?: string;
  states?: string[] | null;
  class_names?: string[] | null;
  cities?: string[] | null;
  vehicle_ids?: string[] | null;
  session_id?: string | null;
  min_conf?: number | null;
  formats?: string[];
  limit?: number | null;
  val_frac?: number;
  test_frac?: number;
  split_group_by?: string;
  split_seed?: string | null;
};

export type services__api__routers__agent__AuditIn = {
  sample_size?: number;
  vlm_calls?: number;
  since_hours?: number;
};

export type services__api__routers__calyx__DriftIn = {
  cam_id: string;
  cam_points: number[][];
  inertial_points: number[][];
};

export type services__api__routers__collaborate__AssignIn = {
  item_id: string;
  user_id: string;
  branch?: string | null;
};

export type services__api__routers__dataset_query__BuildIn = {
  query: string;
  name?: string | null;
  version?: string | null;
  samples_per_shard?: number;
  limit?: number;
};

export type services__api__routers__explore__EvalIn = {
  gold_id: string;
  run_id?: string | null;
  model_version?: string | null;
  iou_thr?: number;
  score_thr?: number;
};

export type services__api__routers__flywheel__DispatchIn = {
  cycle_id?: string | null;
  slices?: string[] | null;
  per_slice_cap?: number;
};

export type services__api__routers__forgyx__ExportIn = {
  imgsz?: number;
  runs?: number;
  prefer_cuda?: boolean;
};

export type services__api__routers__govern__DriftIn = {
  ref_sessions?: string[] | null;
  cur_sessions?: string[] | null;
};

export type services__api__routers__labelops__AssignIn = {
  assignee_id?: string | null;
  expected_version?: number | null;
};

export type services__api__routers__labelops__ProjectIn = {
  name: string;
  description?: string | null;
  modality?: string;
  honeypot_frac?: number;
  min_honeypot_accuracy?: number;
  gold_id?: string | null;
};

export type services__api__routers__labelox__AuditIn = {
  predicted: Record<string, unknown>[];
  gold: Record<string, unknown>[];
  iou_thr?: number;
};

export type services__api__routers__lidar__BuildIn = {
  limit?: number;
  stride?: number | null;
  max_points?: number | null;
};

export type services__api__routers__project_settings__ProjectIn = {
  name: string;
  description?: string | null;
  visibility?: string;
};

export type services__api__routers__shadow__SweepIn = {
  challengers?: string[];
  frame_limit?: number | null;
};

export type services__api__routers__synth__BuildIn = {
  class_names: string[];
  n_frames?: number;
  seed?: string | null;
};

export type services__api__routers__training__SweepIn = {
  name?: string;
  space?: Record<string, unknown[]>;
  method?: string;
  max_trials?: number;
  seed?: number;
  base?: Record<string, unknown>;
};

export type services__api__routers__verdyx__EvalIn = {
  model_version: string;
  aggregate: Record<string, unknown>;
  per_slice?: Record<string, unknown> | null;
  run_id?: string | null;
  challenger_of?: string | null;
  release_commit?: string | null;
  gold_id?: string | null;
  protected?: string[] | null;
};

export type services__api__routers__workforce__DispatchIn = {
  job_id: string;
  workforce_id?: string | null;
  required_classes?: string[] | null;
  deliver?: boolean;
};

/** Every operation the server exposes, by method and path. */
export type Operations = {
  "GET /api/activelearn/batches": {
    response: unknown;
  };
  "POST /api/activelearn/champion-sweep": {
    query: { session_id?: string | null; conf_floor?: number; accept_threshold?: number; limit?: number; top_k?: number };
    response: unknown;
  };
  "GET /api/activelearn/false-negatives": {
    query: { session_id?: string | null; limit?: number; top_k?: number; accept_threshold?: number };
    response: unknown;
  };
  "GET /api/activelearn/loop": {
    response: unknown;
  };
  "POST /api/activelearn/loop/retrain": {
    body: RetrainIn;
    response: unknown;
  };
  "POST /api/activelearn/recall-reliability": {
    query: { apply?: boolean; min_verdicts?: number };
    response: unknown;
  };
  "GET /api/activelearn/score": {
    query: { session_id?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/activelearn/select": {
    body: SelectIn;
    response: unknown;
  };
  "GET /api/activity": {
    query: { user_id?: string | null; verb?: string | null; since_hours?: number | null; mine?: boolean; limit?: number; offset?: number };
    response: unknown;
  };
  "GET /api/activity/summary": {
    query: { hours?: number; mine?: boolean };
    response: unknown;
  };
  "DELETE /api/adverse/{region_id}": {
    path: { region_id: string };
    response: unknown;
  };
  "POST /api/agent/ask": {
    body: AskIn;
    response: unknown;
  };
  "GET /api/agent/audit/latest": {
    response: unknown;
  };
  "POST /api/agent/audit/run": {
    body: services__api__routers__agent__AuditIn | null;
    response: unknown;
  };
  "POST /api/agent/buyer/spec": {
    body: BuyerSpecIn;
    response: unknown;
  };
  "POST /api/agent/cleanup-sweep": {
    body: CleanupIn | null;
    response: unknown;
  };
  "POST /api/agent/command": {
    body: CommandIn;
    response: unknown;
  };
  "GET /api/agent/contamination": {
    query: { min_count?: number; refused_only?: boolean };
    response: unknown;
  };
  "GET /api/agent/contamination/objects": {
    query: { from_name: string; to_name: string; limit?: number; offset?: number };
    response: unknown;
  };
  "POST /api/agent/contamination/revert": {
    body: LineageRevertIn;
    response: unknown;
  };
  "POST /api/agent/copilot/batch-fix": {
    body: BatchFixIn;
    response: unknown;
  };
  "GET /api/agent/copilot/pattern": {
    query: { mine?: boolean };
    response: unknown;
  };
  "GET /api/agent/coverage": {
    response: unknown;
  };
  "POST /api/agent/disagreements/mine": {
    body: MineIn | null;
    response: unknown;
  };
  "POST /api/agent/docs/datasheet": {
    body: DatasheetIn | null;
    response: unknown;
  };
  "POST /api/agent/docs/model-card": {
    query: { model_version: string };
    response: unknown;
  };
  "POST /api/agent/docs/weekly": {
    response: unknown;
  };
  "POST /api/agent/drift/investigate": {
    body: DriftInvestigateIn | null;
    response: unknown;
  };
  "GET /api/agent/drift/latest": {
    response: unknown;
  };
  "GET /api/agent/errors/queue": {
    query: { status?: string; limit?: number };
    response: unknown;
  };
  "POST /api/agent/errors/sweep": {
    body: ErrorSweepIn;
    response: unknown;
  };
  "GET /api/agent/fleet/orders": {
    query: { status?: string };
    response: unknown;
  };
  "POST /api/agent/fleet/orders/{order_id}/{status}": {
    path: { order_id: string; status: string };
    response: unknown;
  };
  "POST /api/agent/fleet/plan": {
    response: unknown;
  };
  "POST /api/agent/flywheel": {
    body: FlywheelIn;
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/attributes": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/attributes/plan": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/cuboid/at": {
    body: CuboidAtIn;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/cuboids": {
    body: CuboidIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/cuboids/plan": {
    body: CuboidIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/fresh": {
    body: FreshIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/plan": {
    body: AgentPolicyIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/reanalyze": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/reanalyze/plan": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/reconcile": {
    body: ReconcileIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/relabel": {
    body: RelabelIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/relabel/plan": {
    body: RelabelIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/frames/{frame_id}/run": {
    body: AgentPolicyIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/agent/frames/{frame_id}/suggest": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/agent/gold-drift": {
    response: unknown;
  };
  "POST /api/agent/nl-edit/apply": {
    body: NlEditApplyIn;
    response: unknown;
  };
  "POST /api/agent/nl-edit/preview": {
    body: NlEditPreviewIn;
    response: unknown;
  };
  "POST /api/agent/objects/{object_id}/crosscam": {
    body: CrossCamIn | null;
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/agent/objects/{object_id}/crosscam/plan": {
    body: CrossCamIn | null;
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/agent/objects/{object_id}/cuboid/fit": {
    body: FitCuboidIn | null;
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/agent/objects/{object_id}/propagate": {
    body: PropagateIn | null;
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/agent/objects/{object_id}/propagate/plan": {
    body: PropagateIn | null;
    path: { object_id: string };
    response: unknown;
  };
  "GET /api/agent/ontology/proposals": {
    query: { status?: string };
    response: unknown;
  };
  "POST /api/agent/ontology/proposals/{proposal_id}/approve": {
    body: ApproveIn;
    path: { proposal_id: string };
    response: unknown;
  };
  "POST /api/agent/ontology/proposals/{proposal_id}/reject": {
    path: { proposal_id: string };
    response: unknown;
  };
  "POST /api/agent/ontology/scan": {
    body: StewardScanIn | null;
    response: unknown;
  };
  "POST /api/agent/ops/ask": {
    body: OpsAskIn;
    response: unknown;
  };
  "POST /api/agent/reanalyze/all": {
    body: ReanalyzeAllIn | null;
    response: unknown;
  };
  "POST /api/agent/relabel/all": {
    body: RelabelAllIn | null;
    response: unknown;
  };
  "GET /api/agent/report": {
    response: unknown;
  };
  "GET /api/agent/runs": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/agent/runs/interrupted": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/agent/runs/{run_id}": {
    path: { run_id: string };
    response: unknown;
  };
  "POST /api/agent/runs/{run_id}/resume": {
    path: { run_id: string };
    response: unknown;
  };
  "POST /api/agent/runs/{run_id}/revert": {
    path: { run_id: string };
    response: unknown;
  };
  "POST /api/agent/scenarios/mine": {
    body: MineIn | null;
    response: unknown;
  };
  "POST /api/agent/temporal-repair": {
    body: TemporalRepairIn | null;
    response: unknown;
  };
  "POST /api/agent/temporal-repair/plan": {
    body: TemporalRepairIn | null;
    response: unknown;
  };
  "POST /api/agent/training/cycle": {
    body: CycleIn | null;
    response: unknown;
  };
  "GET /api/analytics/classes": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/cluster-map": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/analytics/dedup-rate": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/geo": {
    query: { session_id?: string | null; limit?: number; epsilon?: number | null };
    response: unknown;
  };
  "GET /api/analytics/growth": {
    response: unknown;
  };
  "GET /api/analytics/label-value": {
    query: { run_id?: string | null };
    response: unknown;
  };
  "POST /api/analytics/label-value/snapshot": {
    query: { run_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/overview": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/pii": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/privacy-budget": {
    response: unknown;
  };
  "GET /api/analytics/productivity": {
    response: unknown;
  };
  "GET /api/analytics/report": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/review-agreement": {
    response: unknown;
  };
  "GET /api/analytics/scenarios": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/scene-splits": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/analytics/source-mix": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "PATCH /api/annotations/{annotation_id}": {
    body: AnnotationPatch;
    path: { annotation_id: string };
    response: unknown;
  };
  "DELETE /api/annotations/{annotation_id}": {
    path: { annotation_id: string };
    response: unknown;
  };
  "POST /api/assets": {
    body: AssetsIn;
    response: unknown;
  };
  "GET /api/assets/kinds": {
    response: unknown;
  };
  "GET /api/assets/{asset_id}": {
    path: { asset_id: string };
    response: unknown;
  };
  "GET /api/assets/{asset_id}/annotations": {
    path: { asset_id: string };
    query: { kind?: string | null };
    response: unknown;
  };
  "POST /api/assets/{asset_id}/annotations": {
    body: AnnotationIn;
    path: { asset_id: string };
    response: unknown;
  };
  "GET /api/assets/{asset_id}/media": {
    path: { asset_id: string };
    response: unknown;
  };
  "POST /api/assets/{asset_id}/state": {
    path: { asset_id: string };
    query: { state: string };
    response: unknown;
  };
  "POST /api/assistant/chat": {
    body: ChatIn;
    response: unknown;
  };
  "POST /api/assistant/eval": {
    response: unknown;
  };
  "GET /api/assistant/proposals": {
    query: { status?: string; limit?: number };
    response: unknown;
  };
  "POST /api/assistant/proposals/{proposal_id}/approve": {
    path: { proposal_id: string };
    response: unknown;
  };
  "POST /api/assistant/proposals/{proposal_id}/reject": {
    path: { proposal_id: string };
    response: unknown;
  };
  "GET /api/assistant/status": {
    response: unknown;
  };
  "POST /api/attrsweep/apply": {
    body: SweepApplyIn;
    response: unknown;
  };
  "GET /api/attrsweep/coverage": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/attrsweep/queue": {
    query: { attr: string; class_name?: string | null; session_id?: string | null; limit?: number; unit?: string };
    response: unknown;
  };
  "POST /api/auth/dev-login": {
    response: Record<string, unknown>;
  };
  "POST /api/auth/login": {
    body: LoginIn;
    response: Record<string, unknown>;
  };
  "POST /api/auth/login/mfa": {
    body: MfaIn;
    response: Record<string, unknown>;
  };
  "POST /api/auth/logout": {
    response: Record<string, unknown>;
  };
  "POST /api/auth/media-token": {
    response: Record<string, unknown>;
  };
  "GET /api/auth/methods": {
    response: AuthMethodsOut;
  };
  "POST /api/auth/mfa/confirm": {
    body: CodeIn;
    response: Record<string, unknown>;
  };
  "POST /api/auth/mfa/disable": {
    body: DisableMfaIn;
    response: Record<string, unknown>;
  };
  "POST /api/auth/mfa/setup": {
    response: Record<string, unknown>;
  };
  "POST /api/auth/mfa/setup-pending": {
    body: PendingSetupIn;
    response: Record<string, unknown>;
  };
  "GET /api/auth/oidc/callback": {
    query: { code?: string | null; state?: string | null; error?: string | null };
    response: unknown;
  };
  "GET /api/auth/oidc/start": {
    query: { next?: string };
    response: unknown;
  };
  "POST /api/auth/password/change": {
    body: ChangePasswordIn;
    response: Record<string, unknown>;
  };
  "POST /api/auth/password/reset": {
    body: ResetIn;
    response: Record<string, unknown>;
  };
  "POST /api/auth/password/reset-request": {
    body: ResetRequestIn;
    response: Record<string, unknown>;
  };
  "GET /api/auth/profile": {
    response: ProfileOut;
  };
  "POST /api/auth/refresh": {
    response: Record<string, unknown>;
  };
  "POST /api/auth/sessions/revoke": {
    response: Record<string, unknown>;
  };
  "POST /api/auth/signup": {
    body: SignupIn;
    response: Record<string, unknown>;
  };
  "POST /api/autolabel/ego-masks/estimate": {
    query: { force?: boolean };
    response: unknown;
  };
  "POST /api/autolabel/pii-backfill": {
    query: { limit?: number; session_id?: string | null };
    response: unknown;
  };
  "POST /api/autolabel/redetect-all": {
    query: { backfill_pii?: boolean };
    response: unknown;
  };
  "POST /api/autolabel/start": {
    body: AutolabelStartIn;
    response: unknown;
  };
  "GET /api/autolabel/{job_id}": {
    path: { job_id: string };
    response: AutolabelJobOut;
  };
  "GET /api/autonomy/state": {
    response: unknown;
  };
  "POST /api/billing/certify/{commit_id}": {
    path: { commit_id: string };
    query: { eval_id: string; gold_id: string; model_version: string };
    response: unknown;
  };
  "GET /api/billing/invoice": {
    query: { account?: string; since_ns?: number | null; until_ns?: number | null };
    response: unknown;
  };
  "GET /api/billing/usage": {
    query: { account?: string | null; since_ns?: number | null; until_ns?: number | null };
    response: unknown;
  };
  "GET /api/calibration/sessions": {
    response: unknown;
  };
  "POST /api/calibration/validate": {
    query: { session_id: string };
    response: unknown;
  };
  "POST /api/calibration/vehicle/{vehicle_id}/calibrate": {
    body: SetCalibIn;
    path: { vehicle_id: string };
    response: unknown;
  };
  "GET /api/calibration/{session_id}": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/calibration/{session_id}/calibrate": {
    body: SetCalibIn;
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/calibration/{session_id}/estimate": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/calibration/{session_id}/extrinsics": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/calibration/{session_id}/import": {
    body: ImportCalibIn;
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/calibration/{session_id}/resolved": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/calyx/rig/{vehicle_id}/consensus": {
    path: { vehicle_id: string };
    response: unknown;
  };
  "GET /api/calyx/rig/{vehicle_id}/history": {
    path: { vehicle_id: string };
    response: unknown;
  };
  "GET /api/calyx/session/{session_id}": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/calyx/session/{session_id}/drift": {
    body: services__api__routers__calyx__DriftIn;
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/calyx/session/{session_id}/recover": {
    body: RecoverIn;
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/calyx/targetless": {
    body: TargetlessIn;
    response: unknown;
  };
  "GET /api/campaigns": {
    query: { status?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/campaigns": {
    body: CampaignIn;
    response: unknown;
  };
  "GET /api/campaigns/{campaign_id}": {
    path: { campaign_id: string };
    response: unknown;
  };
  "POST /api/campaigns/{campaign_id}/approve": {
    path: { campaign_id: string };
    query: { stage: string };
    response: unknown;
  };
  "POST /api/campaigns/{campaign_id}/stop": {
    path: { campaign_id: string };
    query: { reason?: string };
    response: unknown;
  };
  "POST /api/campaigns/{campaign_id}/tick": {
    path: { campaign_id: string };
    query: { dry_run?: boolean };
    response: unknown;
  };
  "GET /api/checkpoints/{checkpoint_id}": {
    path: { checkpoint_id: string };
    response: unknown;
  };
  "DELETE /api/checkpoints/{checkpoint_id}": {
    path: { checkpoint_id: string };
    response: unknown;
  };
  "POST /api/checkpoints/{checkpoint_id}/restore": {
    path: { checkpoint_id: string };
    response: unknown;
  };
  "POST /api/cloud/connect": {
    body: ConnectIn;
    response: unknown;
  };
  "POST /api/cloud/disconnect": {
    body: DisconnectIn;
    response: unknown;
  };
  "GET /api/cloud/orphans": {
    response: unknown;
  };
  "POST /api/cloud/orphans/terminate": {
    body: TerminatePodIn;
    response: unknown;
  };
  "GET /api/cloud/status": {
    response: unknown;
  };
  "POST /api/collaborate/assign": {
    body: services__api__routers__collaborate__AssignIn;
    response: unknown;
  };
  "GET /api/collaborate/assignments": {
    query: { user_id?: string | null };
    response: unknown;
  };
  "POST /api/collaborate/assignments/{assignment_id}/commit": {
    body: CommitWorkIn;
    path: { assignment_id: string };
    response: unknown;
  };
  "GET /api/collaborate/branches": {
    response: unknown;
  };
  "GET /api/collaborate/merge_requests": {
    response: unknown;
  };
  "POST /api/collaborate/merge_requests/open": {
    body: OpenMRIn;
    response: unknown;
  };
  "POST /api/collaborate/merge_requests/{mr_id}/approve": {
    body: ReviewerIn;
    path: { mr_id: string };
    response: unknown;
  };
  "POST /api/collaborate/merge_requests/{mr_id}/merge": {
    body: ReviewerIn;
    path: { mr_id: string };
    response: unknown;
  };
  "POST /api/collaborate/merge_requests/{mr_id}/revert": {
    body: ReviewerIn;
    path: { mr_id: string };
    response: unknown;
  };
  "POST /api/collaborate/tasks/claim": {
    query: { user_id: string };
    response: unknown;
  };
  "POST /api/collaborate/tasks/enqueue": {
    body: EnqueueIn;
    response: unknown;
  };
  "GET /api/collaborate/tasks/stats": {
    query: { user_id: string };
    response: unknown;
  };
  "POST /api/collaborate/tasks/{assignment_id}/advance": {
    path: { assignment_id: string };
    query: { to_status: string };
    response: unknown;
  };
  "GET /api/corrections/confusions": {
    query: { by?: string; limit?: number };
    response: unknown;
  };
  "GET /api/corrections/coverage": {
    response: unknown;
  };
  "POST /api/corrections/embed": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "POST /api/corrections/suggest": {
    body: SuggestIn;
    response: unknown;
  };
  "POST /api/curation/dedup": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/curation/diverse": {
    query: { session_id?: string | null; k?: number };
    response: unknown;
  };
  "POST /api/curation/embed": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "POST /api/curation/extract": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/curation/slices": {
    response: unknown;
  };
  "POST /api/curation/slices": {
    body: SliceIn;
    response: unknown;
  };
  "GET /api/curation/slices/{slice_id}/materialize": {
    path: { slice_id: string };
    query: { sample?: number };
    response: unknown;
  };
  "GET /api/curation/summary": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/datasets": {
    query: { limit?: number };
    response: DatasetOut[];
  };
  "GET /api/datasets/blob": {
    query: { uri: string };
    response: unknown;
  };
  "GET /api/datasets/compile": {
    query: { q: string };
    response: unknown;
  };
  "POST /api/datasets/export": {
    body: services__api__deps__ExportIn;
    response: unknown;
  };
  "POST /api/datasets/lake/publish": {
    response: unknown;
  };
  "GET /api/datasets/lake/{table}": {
    path: { table: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/datasets/preview": {
    query: { q: string; version?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/datasets/shards": {
    body: services__api__routers__dataset_query__BuildIn;
    response: unknown;
  };
  "GET /api/datasets/versions/{name}": {
    path: { name: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/datasets/vocabulary": {
    response: unknown;
  };
  "GET /api/datasets/{a_id}/diff/{b_id}": {
    path: { a_id: string; b_id: string };
    query: { deep?: boolean; sample?: number };
    response: unknown;
  };
  "GET /api/datasets/{commit_id}": {
    path: { commit_id: string };
    response: DatasetDetailOut;
  };
  "GET /api/datasets/{commit_id}/download": {
    path: { commit_id: string };
    response: unknown;
  };
  "GET /api/datasets/{commit_id}/lineage": {
    path: { commit_id: string };
    response: unknown;
  };
  "GET /api/discovery/queue": {
    query: { state?: string; limit?: number };
    response: unknown;
  };
  "POST /api/discovery/run": {
    query: { session_id: string };
    response: unknown;
  };
  "POST /api/discovery/{candidate_id}/state": {
    body: CandidateStateIn;
    path: { candidate_id: string };
    response: unknown;
  };
  "POST /api/drivable/backfill": {
    query: { session_id?: string | null; max_frames?: number; redo?: boolean };
    response: unknown;
  };
  "GET /api/drivable/coverage": {
    response: unknown;
  };
  "POST /api/driving-events/{event_id}/confirm": {
    body: RulingIn;
    path: { event_id: string };
    response: unknown;
  };
  "POST /api/driving-events/{event_id}/reject": {
    body: RulingIn;
    path: { event_id: string };
    response: unknown;
  };
  "POST /api/dynamics/compute": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/dynamics/frame/{frame_id}": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/dynamics/object/{object_id}": {
    path: { object_id: string };
    response: unknown;
  };
  "GET /api/edge/artifacts/{artifact_id}/field": {
    path: { artifact_id: string };
    query: { hours?: number };
    response: unknown;
  };
  "GET /api/edge/artifacts/{artifact_id}/gate": {
    path: { artifact_id: string };
    query: { hours?: number };
    response: unknown;
  };
  "GET /api/edge/devices": {
    query: { fleet?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/edge/devices": {
    body: DeviceIn;
    response: unknown;
  };
  "GET /api/edge/fleet": {
    query: { hours?: number };
    response: unknown;
  };
  "POST /api/edge/telemetry": {
    body: TelemetryIn;
    response: unknown;
  };
  "GET /api/ego/coverage": {
    response: unknown;
  };
  "GET /api/ego/sessions/{session_id}/pose": {
    path: { session_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/ego/sessions/{session_id}/pose": {
    body: PoseBuildIn | null;
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/ego/tracks/{track_id}/lift3d": {
    path: { track_id: string };
    query: { dry_run?: boolean };
    response: unknown;
  };
  "POST /api/embeddings/compute": {
    body: Record<string, unknown>;
    response: unknown;
  };
  "POST /api/errordetect/benchmark": {
    body: BenchmarkIn;
    response: unknown;
  };
  "GET /api/errordetect/candidates": {
    query: { status?: string; limit?: number; kind?: string | null };
    response: unknown;
  };
  "POST /api/errordetect/candidates/bulk": {
    body: BulkVerdictIn;
    response: unknown;
  };
  "POST /api/errordetect/candidates/{candidate_id}/confirm": {
    body: ConfirmIn;
    path: { candidate_id: string };
    response: unknown;
  };
  "POST /api/errordetect/candidates/{candidate_id}/dismiss": {
    path: { candidate_id: string };
    query: { note?: string | null };
    response: unknown;
  };
  "GET /api/errordetect/precision": {
    response: unknown;
  };
  "POST /api/errordetect/run": {
    body: DetectIn;
    response: unknown;
  };
  "GET /api/errordetect/summary": {
    response: unknown;
  };
  "GET /api/eval/operations": {
    query: { window_days?: number | null };
    response: Record<string, unknown>;
  };
  "GET /api/eval/operations/{op_type}/latest": {
    path: { op_type: string };
    query: { window_days?: number | null };
    response: Record<string, unknown>;
  };
  "POST /api/eval/tracking": {
    query: { gold_id: string; run_id: string; iou_thr?: number };
    response: unknown;
  };
  "GET /api/events/corpus-summary": {
    response: unknown;
  };
  "GET /api/events/jobs": {
    response: unknown;
  };
  "GET /api/events/notifications": {
    response: unknown;
  };
  "GET /api/events/search": {
    query: { kind?: string | null; severity?: string | null; state?: string | null; city?: string | null; session_id?: string | null; min_conf?: number | null; with_kind?: string | null; with_state?: string | null; within_ms?: number; limit?: number; offset?: number };
    response: unknown;
  };
  "GET /api/events/system": {
    response: unknown;
  };
  "GET /api/events/taxonomy": {
    response: unknown;
  };
  "GET /api/events/training/{job_id}": {
    path: { job_id: string };
    response: unknown;
  };
  "PATCH /api/events/{event_id}": {
    body: EventEdit;
    path: { event_id: string };
    response: unknown;
  };
  "DELETE /api/events/{event_id}": {
    path: { event_id: string };
    query: { expected_version?: number | null };
    response: unknown;
  };
  "GET /api/experiments": {
    query: { task_type?: string | null };
    response: unknown;
  };
  "POST /api/experiments": {
    body: ExperimentIn;
    response: unknown;
  };
  "GET /api/experiments/runs/compare": {
    query: { a: string; b: string };
    response: unknown;
  };
  "GET /api/experiments/{name}": {
    path: { name: string };
    query: { metric?: string };
    response: unknown;
  };
  "POST /api/experiments/{name}/runs": {
    body: AttachIn;
    path: { name: string };
    response: unknown;
  };
  "POST /api/explore/eval": {
    body: services__api__routers__explore__EvalIn;
    response: unknown;
  };
  "DELETE /api/explore/eval/{eval_id}": {
    path: { eval_id: string };
    response: unknown;
  };
  "GET /api/explore/eval/{eval_id}/cells": {
    path: { eval_id: string };
    response: unknown;
  };
  "GET /api/explore/eval/{eval_id}/patches": {
    path: { eval_id: string };
    query: { gt_class_id?: number | null; pred_class_id?: number | null; outcome?: string | null; limit?: number };
    response: unknown;
  };
  "GET /api/explore/evals": {
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/explore/facets": {
    body: Record<string, unknown>;
    response: unknown;
  };
  "GET /api/explore/gold/{gold_id}/provenance": {
    path: { gold_id: string };
    response: unknown;
  };
  "POST /api/explore/projection": {
    body: ProjectionIn;
    response: unknown;
  };
  "DELETE /api/explore/projection/{projection_id}": {
    path: { projection_id: string };
    response: unknown;
  };
  "GET /api/explore/projection/{projection_id}/points": {
    path: { projection_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/explore/projections": {
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/explore/select": {
    body: Record<string, unknown>;
    query: { level?: string; limit?: number };
    response: unknown;
  };
  "POST /api/explore/tag": {
    body: TagIn;
    response: unknown;
  };
  "GET /api/explore/tags": {
    query: { level?: string };
    response: unknown;
  };
  "GET /api/explore/views": {
    response: unknown;
  };
  "POST /api/explore/views": {
    body: ViewIn;
    response: unknown;
  };
  "GET /api/explore/views/{slice_id}/export-spec": {
    path: { slice_id: string };
    response: unknown;
  };
  "POST /api/export": {
    body: services__api__deps__ExportIn;
    response: unknown;
  };
  "GET /api/export/scenario/event/{event_id}": {
    path: { event_id: string };
    query: { near_m?: number; pad_s?: number; road_network_file?: string; fmt?: string };
    response: unknown;
  };
  "GET /api/export/scenario/session/{session_id}": {
    path: { session_id: string };
    query: { t_start_ns: number; t_end_ns: number; near_m?: number; road_network_file?: string; fmt?: string; name?: string };
    response: unknown;
  };
  "GET /api/export/{commit_id}/certificate": {
    path: { commit_id: string };
    query: { eval_id: string; gold_id: string; model_version: string; fmt?: string };
    response: unknown;
  };
  "GET /api/exports/resumable": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/exports/{job_id}/progress": {
    path: { job_id: string };
    response: unknown;
  };
  "POST /api/exports/{job_id}/resume": {
    path: { job_id: string };
    response: unknown;
  };
  "GET /api/find": {
    query: { q: string };
    response: FindOut;
  };
  "POST /api/flywheel/adaptive/auto": {
    query: { total_label_budget?: number; safety_floor?: number; min_share?: number };
    response: unknown;
  };
  "POST /api/flywheel/adaptive/collection-orders": {
    query: { cycle_id?: string | null };
    response: unknown;
  };
  "POST /api/flywheel/adaptive/cycle": {
    body: AdaptiveIn;
    response: unknown;
  };
  "GET /api/flywheel/adaptive/cycles": {
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/flywheel/adaptive/dispatch": {
    body: services__api__routers__flywheel__DispatchIn;
    response: unknown;
  };
  "GET /api/flywheel/gate-directed": {
    query: { budget?: number };
    response: unknown;
  };
  "GET /api/flywheel/gate-directed/{run_id}": {
    path: { run_id: string };
    query: { budget?: number };
    response: unknown;
  };
  "POST /api/flywheel/gate-directed/{run_id}/materialize": {
    body: GateBatchIn;
    path: { run_id: string };
    response: unknown;
  };
  "GET /api/flywheel/lineage/{deployment_id}": {
    path: { deployment_id: string };
    response: unknown;
  };
  "POST /api/flywheel/session/{session_id}/run": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/flywheel/stages": {
    response: unknown;
  };
  "POST /api/forgyx/benchmark": {
    body: BenchIn;
    response: unknown;
  };
  "GET /api/forgyx/benchmarks": {
    query: { model_version?: string | null };
    response: unknown;
  };
  "GET /api/forgyx/benchmarks/audit": {
    response: unknown;
  };
  "GET /api/forgyx/capabilities": {
    response: unknown;
  };
  "POST /api/forgyx/cooptimize": {
    body: CooptIn;
    response: unknown;
  };
  "GET /api/forgyx/deployments": {
    query: { model_version?: string | null };
    response: unknown;
  };
  "POST /api/forgyx/distill": {
    query: { teacher_version: string; target: string; student_arch?: string };
    response: unknown;
  };
  "GET /api/forgyx/distill/{run_id}": {
    path: { run_id: string };
    response: unknown;
  };
  "POST /api/forgyx/export/{model_version}": {
    body: services__api__routers__forgyx__ExportIn;
    path: { model_version: string };
    response: unknown;
  };
  "POST /api/forgyx/gate": {
    body: GateIn;
    response: unknown;
  };
  "POST /api/forgyx/rollout": {
    body: RolloutIn;
    response: unknown;
  };
  "POST /api/forgyx/thermal": {
    body: ThermalIn;
    response: unknown;
  };
  "GET /api/frames/{frame_id}": {
    path: { frame_id: string };
    query: { job_id?: string | null };
    response: FrameOut;
  };
  "GET /api/frames/{frame_id}/adverse": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/adverse": {
    body: AdverseIn;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/bev": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/checkpoints": {
    path: { frame_id: string };
    query: { include_auto?: boolean; limit?: number };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/checkpoints": {
    body: CheckpointIn;
    path: { frame_id: string };
    response: unknown;
  };
  "PATCH /api/frames/{frame_id}/context": {
    body: FrameContextIn;
    path: { frame_id: string };
    response: FrameContextOut;
  };
  "GET /api/frames/{frame_id}/cuboids": {
    path: { frame_id: string };
    response: CuboidProjectionOut[];
  };
  "GET /api/frames/{frame_id}/depth-order": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/describe-label": {
    body: DescribeIn;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/drivable": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/drivable": {
    path: { frame_id: string };
    response: unknown;
  };
  "PUT /api/frames/{frame_id}/drivable": {
    body: DrivableIn;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/filmstrip": {
    path: { frame_id: string };
    query: { span?: number };
    response: FilmstripOut;
  };
  "GET /api/frames/{frame_id}/image": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/lanes": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/lanes": {
    body: LaneIn;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/lanes/bev": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/lanes/bev": {
    body: BevProjectIn;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/lanes/propagate": {
    path: { frame_id: string };
    query: { frames?: number };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/lanes/propose": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/lift_ground": {
    path: { frame_id: string };
    query: { u: number; v: number };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/lint": {
    body: LintIn | null;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/next-object": {
    path: { frame_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/objects": {
    path: { frame_id: string };
    query: { job_id?: string | null; limit?: number };
    response: FrameObjectOut[];
  };
  "POST /api/frames/{frame_id}/objects": {
    body: CreateObjectIn;
    path: { frame_id: string };
    response: ObjectDetail;
  };
  "POST /api/frames/{frame_id}/occlusion": {
    path: { frame_id: string };
    query: { commit?: boolean };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/relations": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/relations/propose": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/relationships": {
    path: { frame_id: string };
    response: RelationshipOut[];
  };
  "GET /api/frames/{frame_id}/risk": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/segment": {
    path: { frame_id: string };
    query: { kind?: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/segment": {
    path: { frame_id: string };
    query: { kind?: string };
    response: unknown;
  };
  "PUT /api/frames/{frame_id}/segment": {
    body: SegmentEditIn;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/segment/labelids.png": {
    path: { frame_id: string };
    query: { kind?: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/segment/overlay": {
    path: { frame_id: string };
    query: { kind?: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/segment/panoptic.png": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/frames/{frame_id}/vlm-target/generate": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/frames/{frame_id}/vlm-targets": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/govern/audit": {
    query: { actor?: string | null; limit?: number };
    response: unknown;
  };
  "GET /api/govern/autonomy/ladder": {
    response: unknown;
  };
  "POST /api/govern/autonomy/{class_name}/level": {
    path: { class_name: string };
    query: { level: number; pinned?: boolean };
    response: unknown;
  };
  "POST /api/govern/consent": {
    body: ConsentIn;
    response: unknown;
  };
  "GET /api/govern/consent/{consent_status}/gate": {
    path: { consent_status: string };
    response: unknown;
  };
  "GET /api/govern/control/pending": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/govern/control/precision": {
    response: unknown;
  };
  "POST /api/govern/control/seed": {
    query: { limit?: number; rate?: number | null };
    response: unknown;
  };
  "POST /api/govern/control/{sample_id}/verdict": {
    body: VerdictIn;
    path: { sample_id: string };
    response: unknown;
  };
  "POST /api/govern/controller/tick": {
    query: { schedule_bursts?: boolean };
    response: unknown;
  };
  "POST /api/govern/cost/gate": {
    body: CostGateIn;
    response: unknown;
  };
  "POST /api/govern/drift/scan": {
    body: services__api__routers__govern__DriftIn;
    response: unknown;
  };
  "POST /api/govern/erase": {
    body: EraseIn;
    response: unknown;
  };
  "POST /api/govern/killswitch/engage": {
    body: EngageIn;
    response: unknown;
  };
  "POST /api/govern/killswitch/release": {
    response: unknown;
  };
  "GET /api/govern/lineage/{subject}": {
    path: { subject: string };
    response: unknown;
  };
  "GET /api/govern/pii-access": {
    query: { user_id?: string | null; subject_id?: string | null; session_id?: string | null; action?: string | null; since_hours?: number | null; limit?: number; offset?: number };
    response: unknown;
  };
  "GET /api/govern/pii-access/summary": {
    query: { hours?: number };
    response: unknown;
  };
  "POST /api/govern/promote": {
    query: { model_version: string; task?: string };
    response: unknown;
  };
  "POST /api/govern/redaction/proof": {
    body: RedactionProofIn;
    response: unknown;
  };
  "GET /api/govern/registry": {
    query: { task?: string | null };
    response: unknown;
  };
  "POST /api/govern/registry/register": {
    body: RegisterIn;
    response: unknown;
  };
  "POST /api/govern/registry/register_run": {
    query: { run_id: string; task?: string | null };
    response: unknown;
  };
  "GET /api/govern/retention/due": {
    response: unknown;
  };
  "POST /api/govern/retention/sweep": {
    body: RetentionSweepIn;
    response: unknown;
  };
  "GET /api/govern/settlement/lots": {
    query: { status?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/govern/settlement/plan": {
    query: { class_name: string; epoch?: string | null };
    response: unknown;
  };
  "POST /api/govern/settlement/{lot_id}/ack": {
    path: { lot_id: string };
    response: unknown;
  };
  "POST /api/govern/settlement/{lot_id}/revert": {
    path: { lot_id: string };
    query: { reason?: string };
    response: unknown;
  };
  "GET /api/govern/state": {
    response: unknown;
  };
  "GET /api/gpu/slot": {
    response: unknown;
  };
  "POST /api/hardening/efficiency": {
    body: EfficiencyIn;
    response: unknown;
  };
  "POST /api/hardening/reproducible": {
    body: ReproIn;
    response: unknown;
  };
  "POST /api/hardening/slo": {
    body: SloIn;
    response: unknown;
  };
  "GET /api/hardening/slo/board": {
    response: unknown;
  };
  "GET /api/hdmap/commits": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/hdmap/elements": {
    query: { commit_id?: string | null; session_id?: string | null };
    response: unknown;
  };
  "POST /api/hdmap/elements/metric": {
    body: MetricElementIn;
    query: { session_id: string };
    response: unknown;
  };
  "POST /api/hdmap/fuse": {
    query: { session_ids: string; region?: string | null; compute_target?: string };
    response: unknown;
  };
  "POST /api/hdmap/georef": {
    query: { session_id: string; height_m?: number | null };
    response: unknown;
  };
  "GET /api/hdmap/provenance": {
    query: { element_id: string };
    response: unknown;
  };
  "GET /api/health": {
    response: unknown;
  };
  "GET /api/home": {
    response: HomeOut;
  };
  "GET /api/imports": {
    query: { limit?: number };
    response: ImportJobOut[];
  };
  "POST /api/imports/dry-run": {
    body: ImportStartIn;
    response: unknown;
  };
  "GET /api/imports/formats": {
    response: ImportFormatsOut;
  };
  "POST /api/imports/start": {
    body: ImportStartIn;
    response: unknown;
  };
  "GET /api/imports/{job_id}": {
    path: { job_id: string };
    response: ImportJobOut;
  };
  "POST /api/inference-runs": {
    body: ExternalRunIn;
    response: unknown;
  };
  "GET /api/ingest/progress": {
    response: unknown;
  };
  "POST /api/inspector/health/sweep": {
    query: { limit?: number };
    response: Record<string, unknown>;
  };
  "POST /api/inspector/index/backfill": {
    query: { limit?: number };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/layouts": {
    response: Record<string, unknown>;
  };
  "POST /api/inspector/layouts": {
    body: LayoutIn;
    response: Record<string, unknown>;
  };
  "DELETE /api/inspector/layouts/{layout_id}": {
    path: { layout_id: string };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/sessions": {
    query: { limit?: number };
    response: Record<string, unknown>[];
  };
  "GET /api/inspector/sessions/{session_id}/annotations-at": {
    path: { session_id: string };
    query: { ts_ns: number };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/sessions/{session_id}/cloud": {
    path: { session_id: string };
    query: { ts_ns: number; max_points?: number };
    response: unknown;
  };
  "GET /api/inspector/sessions/{session_id}/events": {
    path: { session_id: string };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/sessions/{session_id}/frame-at": {
    path: { session_id: string };
    query: { ts_ns: number };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/sessions/{session_id}/health": {
    path: { session_id: string };
    response: Record<string, unknown>;
  };
  "POST /api/inspector/sessions/{session_id}/health": {
    path: { session_id: string };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/sessions/{session_id}/index": {
    path: { session_id: string };
    response: Record<string, unknown>;
  };
  "POST /api/inspector/sessions/{session_id}/index": {
    path: { session_id: string };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/sessions/{session_id}/lichtblick": {
    path: { session_id: string };
    response: Record<string, unknown>;
  };
  "GET /api/inspector/sessions/{session_id}/mcap-url": {
    path: { session_id: string };
    response: Record<string, unknown>;
  };
  "GET /api/integrations/events": {
    response: unknown;
  };
  "GET /api/integrations/sources": {
    query: { project_id?: string | null };
    response: unknown;
  };
  "POST /api/integrations/sources": {
    body: SourceIn;
    response: unknown;
  };
  "DELETE /api/integrations/sources/{source_id}": {
    path: { source_id: string };
    response: unknown;
  };
  "GET /api/integrations/sources/{source_id}/preview": {
    path: { source_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/integrations/webhooks": {
    query: { project_id?: string | null };
    response: unknown;
  };
  "POST /api/integrations/webhooks": {
    body: WebhookIn;
    response: unknown;
  };
  "DELETE /api/integrations/webhooks/{webhook_id}": {
    path: { webhook_id: string };
    response: unknown;
  };
  "POST /api/intent/propose-session": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/intent/vocab": {
    response: unknown;
  };
  "GET /api/jobs": {
    query: { limit?: number };
    response: JobOut[];
  };
  "POST /api/jobs/{kind}/{job_id}/cancel": {
    path: { kind: string; job_id: string };
    response: JobCancelOut;
  };
  "GET /api/labelops/issues": {
    query: { frame_id?: string | null; job_id?: string | null; object_id?: string | null; status?: string | null; mine?: boolean };
    response: unknown;
  };
  "POST /api/labelops/issues": {
    body: IssueIn;
    response: unknown;
  };
  "GET /api/labelops/issues/{issue_id}": {
    path: { issue_id: string };
    response: unknown;
  };
  "POST /api/labelops/issues/{issue_id}/comment": {
    body: CommentIn;
    path: { issue_id: string };
    response: unknown;
  };
  "POST /api/labelops/issues/{issue_id}/resolve": {
    path: { issue_id: string };
    query: { reopen?: boolean };
    response: unknown;
  };
  "GET /api/labelops/jobs": {
    query: { project_id?: string | null; task_id?: string | null; assignee_id?: string | null; stage?: string | null; state?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/labelops/jobs/{job_id}/assign": {
    body: services__api__routers__labelops__AssignIn;
    path: { job_id: string };
    response: unknown;
  };
  "POST /api/labelops/jobs/{job_id}/state": {
    body: StateIn;
    path: { job_id: string };
    response: unknown;
  };
  "POST /api/labelops/jobs/{job_id}/submit": {
    body: SubmitIn;
    path: { job_id: string };
    response: unknown;
  };
  "GET /api/labelops/judge-agreement": {
    query: { model_version?: string | null; batch_id?: string | null };
    response: unknown;
  };
  "GET /api/labelops/my-jobs": {
    response: unknown;
  };
  "POST /api/labelops/precision-batch": {
    query: { target?: number };
    response: unknown;
  };
  "GET /api/labelops/precision/{batch_id}": {
    path: { batch_id: string };
    response: unknown;
  };
  "GET /api/labelops/prereview/{batch_id}": {
    path: { batch_id: string };
    query: { model_version?: string | null; agreement_batch_id?: string | null };
    response: unknown;
  };
  "POST /api/labelops/prereview/{batch_id}": {
    path: { batch_id: string };
    query: { limit?: number | null };
    response: unknown;
  };
  "GET /api/labelops/projects": {
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/labelops/projects": {
    body: services__api__routers__labelops__ProjectIn;
    response: unknown;
  };
  "GET /api/labelops/projects/{project_id}/board": {
    path: { project_id: string };
    response: unknown;
  };
  "POST /api/labelops/replica-groups/{replica_group}/agreement": {
    path: { replica_group: string };
    query: { iou_thresh?: number };
    response: unknown;
  };
  "GET /api/labelops/scorecards": {
    query: { project_id?: string | null };
    response: unknown;
  };
  "GET /api/labelops/scorecards/full": {
    query: { confidence?: number };
    response: unknown;
  };
  "GET /api/labelops/scorecards/me": {
    response: unknown;
  };
  "POST /api/labelops/tasks": {
    body: TaskIn;
    response: unknown;
  };
  "GET /api/labelops/tasks/{task_id}/agreement": {
    path: { task_id: string };
    response: unknown;
  };
  "POST /api/labelox/propagate4d": {
    body: Propagate4dIn;
    response: unknown;
  };
  "POST /api/labelox/quality/agreement": {
    body: AgreementIn;
    response: unknown;
  };
  "POST /api/labelox/quality/audit": {
    body: services__api__routers__labelox__AuditIn;
    response: unknown;
  };
  "GET /api/labelox/queue": {
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/labelox/reconcile/parity": {
    body: ParityIn;
    response: unknown;
  };
  "GET /api/lanes/type-coverage": {
    response: unknown;
  };
  "PUT /api/lanes/{lane_id}": {
    body: LaneIn;
    path: { lane_id: string };
    response: unknown;
  };
  "DELETE /api/lanes/{lane_id}": {
    path: { lane_id: string };
    response: unknown;
  };
  "POST /api/lidar/aggregate": {
    body: AggregateIn;
    response: unknown;
  };
  "POST /api/lidar/aggregate/{agg_id}/label": {
    body: AggLabelIn;
    path: { agg_id: string };
    response: unknown;
  };
  "GET /api/lidar/analytics3d": {
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}": {
    path: { cloud_id: string };
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}/bev-labels": {
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/extract": {
    path: { cloud_id: string };
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}/ground-qa": {
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/lift": {
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/link": {
    path: { cloud_id: string };
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}/objects3d": {
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/objects3d": {
    body: Cuboid3DIn;
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/occupancy": {
    path: { cloud_id: string };
    query: { voxel_size?: number };
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}/points": {
    path: { cloud_id: string };
    query: { variant?: string | null; max?: number; full?: boolean };
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}/quality3d": {
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/quality3d": {
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/segment": {
    path: { cloud_id: string };
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}/segmentation": {
    path: { cloud_id: string };
    response: unknown;
  };
  "GET /api/lidar/clouds/{cloud_id}/segmentation/points": {
    path: { cloud_id: string };
    query: { max?: number };
    response: unknown;
  };
  "POST /api/lidar/clouds/{cloud_id}/traverse": {
    path: { cloud_id: string };
    response: unknown;
  };
  "POST /api/lidar/cuboids/{frame_id}": {
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/lidar/export3d": {
    body: Export3DIn;
    response: unknown;
  };
  "POST /api/lidar/frames/{frame_id}/lift": {
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/lidar/objects2d/{object_id}/linked3d": {
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/lidar/objects3d/batch_correct": {
    body: BatchCorrectIn;
    response: unknown;
  };
  "PATCH /api/lidar/objects3d/{object_3d_id}": {
    body: Cuboid3DEdit;
    path: { object_3d_id: string };
    response: unknown;
  };
  "DELETE /api/lidar/objects3d/{object_3d_id}": {
    path: { object_3d_id: string };
    response: unknown;
  };
  "POST /api/lidar/objects3d/{object_3d_id}/consistency": {
    path: { object_3d_id: string };
    response: unknown;
  };
  "GET /api/lidar/objects3d/{object_3d_id}/linked": {
    path: { object_3d_id: string };
    response: unknown;
  };
  "GET /api/lidar/objects3d/{object_3d_id}/projection": {
    path: { object_3d_id: string };
    query: { cam_id?: string; w?: number; h?: number };
    response: unknown;
  };
  "POST /api/lidar/objects3d/{object_3d_id}/properties": {
    path: { object_3d_id: string };
    response: unknown;
  };
  "GET /api/lidar/objects3d/{object_3d_id}/similar": {
    path: { object_3d_id: string };
    query: { k?: number };
    response: unknown;
  };
  "POST /api/lidar/quality3d/bridge": {
    query: { cloud_id?: string | null; session_id?: string | null; commit?: boolean };
    response: unknown;
  };
  "POST /api/lidar/quality3d/{flag_id}/confirm": {
    path: { flag_id: string };
    response: unknown;
  };
  "GET /api/lidar/search3d": {
    query: { q: string };
    response: unknown;
  };
  "POST /api/lidar/sessions/{session_id}/build": {
    body: services__api__routers__lidar__BuildIn | null;
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/lidar/sessions/{session_id}/clouds": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/lidar/sessions/{session_id}/rare3d": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/lidar/sessions/{session_id}/scene3d": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/lidar/sessions/{session_id}/static_elements": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/lidar/sessions/{session_id}/track3d": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/lidar/sessions/{session_id}/tracks3d": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/lidar/sessions/{session_id}/trajectory": {
    path: { session_id: string };
    query: { ref_ts_ns?: number | null };
    response: unknown;
  };
  "POST /api/lidar/sessions/{session_id}/validate": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/lidar/tracks3d/{track_3d_id}/interpolate": {
    path: { track_3d_id: string };
    response: unknown;
  };
  "GET /api/lineage/dataset/{commit_id}": {
    path: { commit_id: string };
    response: unknown;
  };
  "GET /api/lineage/model/{model_version}": {
    path: { model_version: string };
    query: { max_sessions?: number };
    response: unknown;
  };
  "GET /api/lineage/session/{session_id}": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/mapassist/match": {
    query: { session_id: string; max_dist_m?: number };
    response: unknown;
  };
  "GET /api/mapassist/priors": {
    query: { frame_id: string };
    response: unknown;
  };
  "POST /api/mask/compose": {
    body: ComposeIn;
    response: unknown;
  };
  "GET /api/metrics": {
    response: unknown;
  };
  "POST /api/mine": {
    body: Record<string, unknown>;
    response: unknown;
  };
  "GET /api/models": {
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/models/external": {
    body: ExternalModelIn;
    response: unknown;
  };
  "POST /api/multicam/associate": {
    query: { session_id: string };
    response: unknown;
  };
  "POST /api/multicam/consistency-check": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/multicam/group/at": {
    query: { session_id: string; ts_ns: number };
    response: unknown;
  };
  "POST /api/multicam/group/confirm": {
    query: { group_id: string; confirmed?: boolean };
    response: unknown;
  };
  "GET /api/multicam/group/nav": {
    query: { session_id: string; group_id: string; direction?: string };
    response: unknown;
  };
  "GET /api/multicam/groups": {
    query: { session_id: string; tol_ms?: number };
    response: unknown;
  };
  "POST /api/multicam/groups/build": {
    query: { session_id: string; tol_ms?: number };
    response: unknown;
  };
  "GET /api/multicam/groups/persisted": {
    query: { session_id: string };
    response: unknown;
  };
  "POST /api/multicam/link": {
    body: LinkBody;
    response: unknown;
  };
  "POST /api/multicam/propagate": {
    query: { object_id: string; use_sam?: boolean };
    response: unknown;
  };
  "GET /api/multicam/readiness": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/multicam/rig-objects": {
    query: { session_id: string; group_id: string };
    response: unknown;
  };
  "GET /api/multicam/rig-track/timeline": {
    query: { session_id: string; rig_track_id: string };
    response: unknown;
  };
  "GET /api/multicam/rig-tracks": {
    query: { session_id: string };
    response: unknown;
  };
  "POST /api/multicam/rig-tracks/build": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/multicam/suggest-links": {
    query: { session_id: string; group_id: string; appearance_cos?: number };
    response: unknown;
  };
  "POST /api/multicam/unlink": {
    query: { object_id: string };
    response: unknown;
  };
  "GET /api/notifications": {
    query: { unread_only?: boolean; limit?: number; offset?: number };
    response: NotificationsOut;
  };
  "GET /api/notifications/count": {
    response: UnreadCountOut;
  };
  "POST /api/notifications/read-all": {
    response: MarkAllReadOut;
  };
  "POST /api/notifications/{notification_id}/read": {
    path: { notification_id: string };
    response: MarkReadOut;
  };
  "POST /api/objects/bulk-review": {
    body: BulkReviewIn;
    response: BulkReviewOut;
  };
  "POST /api/objects/classify": {
    body: ClassifyIn;
    response: unknown;
  };
  "POST /api/objects/crops": {
    body: CropSheetIn;
    response: unknown;
  };
  "POST /api/objects/quality/backfill": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/objects/{object_id}": {
    path: { object_id: string };
    response: ObjectDetail;
  };
  "DELETE /api/objects/{object_id}": {
    path: { object_id: string };
    response: ObjectDeletedOut;
  };
  "GET /api/objects/{object_id}/crop": {
    path: { object_id: string };
    query: { pad?: number };
    response: unknown;
  };
  "GET /api/objects/{object_id}/explain": {
    path: { object_id: string };
    response: unknown;
  };
  "GET /api/objects/{object_id}/history": {
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/objects/{object_id}/keyframe": {
    path: { object_id: string };
    query: { value?: boolean };
    response: unknown;
  };
  "PUT /api/objects/{object_id}/mask": {
    body: MaskIn;
    path: { object_id: string };
    response: MaskSavedOut;
  };
  "POST /api/objects/{object_id}/propagate": {
    path: { object_id: string };
    query: { frames?: number };
    response: unknown;
  };
  "GET /api/objects/{object_id}/quality": {
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/objects/{object_id}/reinterpolate": {
    path: { object_id: string };
    query: { method?: string };
    response: unknown;
  };
  "POST /api/objects/{object_id}/relate": {
    body: RelateIn;
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/objects/{object_id}/review": {
    body: ReviewIn;
    path: { object_id: string };
    response: ReviewOut;
  };
  "POST /api/objects/{object_id}/sam_propagate": {
    path: { object_id: string };
    query: { frames?: number; direction?: string; refine?: boolean };
    response: unknown;
  };
  "GET /api/objects/{object_id}/similar": {
    path: { object_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/occupancy/sessions/{session_id}": {
    path: { session_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/occupancy/sessions/{session_id}/build": {
    body: OccupancyBuildIn | null;
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/occupancy/{grid_id}/voxels": {
    path: { grid_id: string };
    query: { max_voxels?: number };
    response: unknown;
  };
  "POST /api/ocr/region": {
    body: RegionOcrIn;
    response: Record<string, unknown>;
  };
  "POST /api/ocr/run": {
    query: { session_id: string; limit?: number | null };
    response: unknown;
  };
  "GET /api/ontology": {
    query: { frame_id?: string | null; session_id?: string | null; project_id?: string | null };
    response: OntologyOut;
  };
  "POST /api/ontology/classes": {
    body: NewClassIn;
    response: ClassCreatedOut;
  };
  "POST /api/ontology/classes/merge": {
    body: MergeIn;
    response: unknown;
  };
  "POST /api/ontology/classes/rename": {
    body: RenameIn;
    response: unknown;
  };
  "POST /api/ontology/classes/retire": {
    body: number[];
    response: unknown;
  };
  "POST /api/ontology/merges/{run_id}/revert": {
    path: { run_id: string };
    response: unknown;
  };
  "GET /api/ontology/templates": {
    response: ClassTemplatesOut;
  };
  "GET /api/oraclyx/board": {
    response: unknown;
  };
  "POST /api/oraclyx/disagreements/rank": {
    body: RankIn;
    response: unknown;
  };
  "GET /api/oraclyx/distillation": {
    query: { min_score?: number };
    response: unknown;
  };
  "POST /api/oraclyx/mono-depth": {
    body: MonoDepthIn;
    response: unknown;
  };
  "POST /api/oraclyx/object/{object_id}/consensus": {
    body: ConsensusIn;
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/oraclyx/radar/fuse": {
    body: RadarIn;
    response: unknown;
  };
  "GET /api/oraclyx/thresholds": {
    query: { model_version: string };
    response: unknown;
  };
  "POST /api/oraclyx/thresholds/fit": {
    body: FitThresholdsIn;
    response: unknown;
  };
  "GET /api/oraclyx/thresholds/{fit_id}": {
    path: { fit_id: string };
    response: unknown;
  };
  "POST /api/oraclyx/thresholds/{fit_id}/activate": {
    path: { fit_id: string };
    response: unknown;
  };
  "POST /api/oraclyx/tracks4d/stitch": {
    body: StitchIn;
    response: unknown;
  };
  "POST /api/oraclyx/uncertainty": {
    body: UncertaintyIn;
    response: unknown;
  };
  "GET /api/platforms": {
    response: PlatformsOut;
  };
  "GET /api/predictions/{prediction_id}/crop": {
    path: { prediction_id: string };
    query: { pad?: number };
    response: unknown;
  };
  "POST /api/presence/frames/{frame_id}/heartbeat": {
    body: HeartbeatIn;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/presence/frames/{frame_id}/lock/acquire": {
    body: TabIn;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/presence/frames/{frame_id}/lock/release": {
    body: ReleaseIn;
    path: { frame_id: string };
    response: unknown;
  };
  "POST /api/presence/frames/{frame_id}/lock/takeover": {
    body: TabIn;
    path: { frame_id: string };
    response: unknown;
  };
  "GET /api/projects": {
    response: ProjectsOut;
  };
  "POST /api/projects": {
    body: services__api__routers__project_settings__ProjectIn;
    response: ProjectCreatedOut;
  };
  "PATCH /api/projects/{project_id}": {
    body: ProjectPatch;
    path: { project_id: string };
    response: ProjectChangedOut;
  };
  "GET /api/projects/{project_id}/assets": {
    path: { project_id: string };
    query: { state?: string | null; media_type?: string | null; limit?: number; offset?: number };
    response: unknown;
  };
  "PUT /api/projects/{project_id}/classes": {
    body: ClassesIn;
    path: { project_id: string };
    response: ProjectClassesOut;
  };
  "POST /api/projects/{project_id}/label-config": {
    body: Record<string, unknown>;
    path: { project_id: string };
    response: unknown;
  };
  "GET /api/projects/{project_id}/members": {
    path: { project_id: string };
    response: ProjectMembersOut;
  };
  "PUT /api/projects/{project_id}/members/{user_id}": {
    path: { project_id: string; user_id: string };
    response: MembershipOut;
  };
  "DELETE /api/projects/{project_id}/members/{user_id}": {
    path: { project_id: string; user_id: string };
    response: MembershipOut;
  };
  "POST /api/projects/{project_id}/sessions": {
    body: MoveIn;
    path: { project_id: string };
    response: SessionsMovedOut;
  };
  "GET /api/projects/{project_id}/stats": {
    path: { project_id: string };
    response: unknown;
  };
  "GET /api/propagation/eval": {
    query: { steps?: number; tracks?: number; refine?: boolean };
    response: unknown;
  };
  "POST /api/qa/vlm": {
    query: { session_id: string; limit?: number };
    response: unknown;
  };
  "GET /api/quality/attr-audit/{session_id}": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/quality/calibrate/fit": {
    body: CalibrateFitIn;
    response: unknown;
  };
  "GET /api/quality/gold-sets": {
    response: unknown;
  };
  "POST /api/quality/gold/seal": {
    body: GoldSealIn;
    response: unknown;
  };
  "POST /api/quality/gold/{gold_id}/measure": {
    path: { gold_id: string };
    response: unknown;
  };
  "POST /api/quality/iaa": {
    body: IaaIn;
    response: unknown;
  };
  "GET /api/quality/sheet": {
    query: { gold_id: string };
    response: unknown;
  };
  "POST /api/quality/track-gap-fill": {
    query: { max_tracks?: number };
    response: unknown;
  };
  "GET /api/quality/track-gap-fill/plan": {
    response: unknown;
  };
  "POST /api/quality/track-relabel-backfill": {
    query: { max_tracks?: number };
    response: unknown;
  };
  "GET /api/quality/track-relabel-backfill/plan": {
    response: unknown;
  };
  "GET /api/readyz": {
    response: unknown;
  };
  "GET /api/reasoner/attribution": {
    query: { since_hours?: number | null };
    response: unknown;
  };
  "GET /api/reasoner/coverage": {
    response: unknown;
  };
  "POST /api/reasoner/explain": {
    body: ExplainIn;
    response: unknown;
  };
  "GET /api/reasoner/outcomes": {
    query: { since_hours?: number | null };
    response: unknown;
  };
  "GET /api/reasoner/priors": {
    response: unknown;
  };
  "POST /api/reasoner/rerun/{session_id}": {
    path: { session_id: string };
    query: { limit?: number; apply?: boolean };
    response: unknown;
  };
  "GET /api/reasoner/trace/{object_id}": {
    path: { object_id: string };
    response: unknown;
  };
  "GET /api/reasoner/weights": {
    query: { since_hours?: number | null };
    response: unknown;
  };
  "GET /api/recall/candidates": {
    query: { status?: string; session_id?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/recall/candidates/{candidate_id}/{verdict}": {
    path: { candidate_id: string; verdict: string };
    response: unknown;
  };
  "GET /api/recall/progress": {
    response: unknown;
  };
  "POST /api/recall/run/{session_id}": {
    path: { session_id: string };
    query: { shortlist_only?: boolean };
    response: unknown;
  };
  "POST /api/relabel/ingest": {
    body: IngestIn;
    response: unknown;
  };
  "GET /api/relabel/runs": {
    response: unknown;
  };
  "POST /api/relabel/runs/{run_id}/revert": {
    path: { run_id: string };
    response: unknown;
  };
  "POST /api/relabel/start": {
    body: RelabelStartIn;
    response: unknown;
  };
  "POST /api/relations/{relationship_id}/status": {
    path: { relationship_id: string };
    query: { status: string };
    response: unknown;
  };
  "DELETE /api/relationships/{relationship_id}": {
    path: { relationship_id: string };
    response: unknown;
  };
  "GET /api/release/gold/{gold_id}/lineage": {
    path: { gold_id: string };
    response: unknown;
  };
  "GET /api/release/registry": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/release/{commit_id}/verify": {
    path: { commit_id: string };
    response: unknown;
  };
  "GET /api/samples": {
    response: unknown;
  };
  "POST /api/samples/{sample_id}/load": {
    path: { sample_id: string };
    response: unknown;
  };
  "GET /api/sanyx/board": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/sanyx/compat-matrix": {
    query: { top?: number };
    response: unknown;
  };
  "GET /api/sanyx/gate/{session_id}": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sanyx/rig/{vehicle_id}/trends": {
    path: { vehicle_id: string };
    response: unknown;
  };
  "GET /api/sanyx/session/{session_id}": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sanyx/session/{session_id}/override": {
    body: OverrideIn;
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sanyx/session/{session_id}/run": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/scenarios": {
    query: { session_id?: string | null; type?: string | null; city?: string | null; limit?: number };
    response: unknown;
  };
  "GET /api/scenarios/search": {
    query: { q: string; city?: string | null; session_id?: string | null; limit?: number; semantic?: boolean };
    response: unknown;
  };
  "GET /api/scenarios/{scenario_id}": {
    path: { scenario_id: string };
    response: unknown;
  };
  "GET /api/scene-graph/vocab": {
    response: unknown;
  };
  "POST /api/scene/classify": {
    query: { session_id?: string | null; limit?: number | null };
    response: unknown;
  };
  "GET /api/search/objects": {
    query: { q: string; session_id?: string | null; limit?: number };
    response: unknown;
  };
  "GET /api/search/semantic": {
    query: { q: string; k?: number; rarity_weight?: number | null };
    response: unknown;
  };
  "POST /api/search/similar": {
    body: SimilarIn;
    response: unknown;
  };
  "GET /api/sec/identities": {
    query: { min_cameras?: number; limit?: number };
    response: unknown;
  };
  "GET /api/sec/identities/{identity_id}": {
    path: { identity_id: string };
    response: unknown;
  };
  "DELETE /api/sec/identities/{identity_id}": {
    path: { identity_id: string };
    response: unknown;
  };
  "GET /api/sec/incidents": {
    query: { camera_id?: string | null; kind?: string | null; status?: string | null; severity?: string | null; since_hours?: number | null; limit?: number; offset?: number };
    response: unknown;
  };
  "POST /api/sec/incidents/{incident_id}/acknowledge": {
    path: { incident_id: string };
    query: { close?: boolean };
    response: unknown;
  };
  "POST /api/sec/rtsp/ingest": {
    body: RtspIn;
    response: unknown;
  };
  "POST /api/sec/sessions/{session_id}/evaluate": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sec/sessions/{session_id}/link-identities": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sec/sessions/{session_id}/stitch-plates": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sec/zones": {
    query: { camera_id?: string | null; active_only?: boolean };
    response: unknown;
  };
  "POST /api/sec/zones": {
    body: ZoneIn;
    response: unknown;
  };
  "DELETE /api/sec/zones/{zone_id}": {
    path: { zone_id: string };
    response: unknown;
  };
  "GET /api/security/pack": {
    query: { pack_id?: string };
    response: unknown;
  };
  "GET /api/security/reads": {
    query: { session_id?: string | null; camera_id?: string | null; plate?: string | null; state_code?: string | null; hits_only?: boolean; limit?: number; offset?: number };
    response: unknown;
  };
  "POST /api/security/recognize": {
    body: RecognizeIn;
    response: unknown;
  };
  "GET /api/security/sessions": {
    query: { limit?: number; offset?: number };
    response: unknown;
  };
  "GET /api/security/stats": {
    response: unknown;
  };
  "GET /api/security/watchlist": {
    query: { active_only?: boolean; limit?: number };
    response: unknown;
  };
  "POST /api/security/watchlist": {
    body: WatchlistIn;
    response: unknown;
  };
  "DELETE /api/security/watchlist/{entry_id}": {
    path: { entry_id: string };
    response: unknown;
  };
  "POST /api/segment": {
    body: SegmentIn;
    response: unknown;
  };
  "GET /api/service-accounts": {
    query: { include_revoked?: boolean };
    response: unknown;
  };
  "POST /api/service-accounts": {
    body: MintIn;
    response: unknown;
  };
  "POST /api/service-accounts/{account_id}/revoke": {
    path: { account_id: string };
    response: unknown;
  };
  "POST /api/service-accounts/{account_id}/rotate": {
    path: { account_id: string };
    response: unknown;
  };
  "GET /api/sessions": {
    query: { limit?: number };
    response: SessionOut[];
  };
  "GET /api/sessions/page": {
    query: { limit?: number; offset?: number; vehicle_id?: string | null; project_id?: string | null };
    response: SessionPageOut;
  };
  "GET /api/sessions/states": {
    response: SessionStateOut[];
  };
  "GET /api/sessions/{session_id}/driving-events": {
    path: { session_id: string };
    query: { kind?: string | null; state?: string | null; severity?: string | null; track_id?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/driving-events": {
    body: DrivingEventIn;
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/driving-events/derive": {
    path: { session_id: string };
    query: { prune_stale?: boolean };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/driving-events/preview": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sessions/{session_id}/driving-events/summary": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sessions/{session_id}/egostate": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sessions/{session_id}/events": {
    path: { session_id: string };
    query: { modality?: string | null };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/events": {
    body: EventIn;
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/events/auto": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/events/correlate": {
    path: { session_id: string };
    query: { ts_ns: number; window_ns?: number };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/events/scene": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sessions/{session_id}/first-frame": {
    path: { session_id: string };
    response: FirstFrameOut;
  };
  "GET /api/sessions/{session_id}/inertial_events": {
    path: { session_id: string };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/lanes/classify": {
    path: { session_id: string };
    query: { apply?: boolean; reclassify?: boolean };
    response: unknown;
  };
  "POST /api/sessions/{session_id}/lanes/link": {
    path: { session_id: string };
    query: { apply?: boolean };
    response: unknown;
  };
  "GET /api/sessions/{session_id}/qa/consistency": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sessions/{session_id}/static-dynamic": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/sessions/{session_id}/stats": {
    path: { session_id: string };
    response: SessionStatsOut;
  };
  "GET /api/sessions/{session_id}/timeline": {
    path: { session_id: string };
    response: unknown;
  };
  "GET /api/shadow/disagreements": {
    query: { sweep_run_id?: string | null; state?: string | null; kind?: string | null; limit?: number };
    response: unknown;
  };
  "GET /api/shadow/runs": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/shadow/summary": {
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/shadow/sweep": {
    body: services__api__routers__shadow__SweepIn | null;
    response: unknown;
  };
  "GET /api/sievyx/cliques": {
    response: unknown;
  };
  "POST /api/sievyx/cliques/reward": {
    body: CliqueRewardIn;
    response: unknown;
  };
  "POST /api/sievyx/cliques/select": {
    query: { run_id: string; budget?: number; seed?: number | null };
    response: unknown;
  };
  "GET /api/sievyx/composition": {
    query: { session_id?: string | null; top_n?: number };
    response: unknown;
  };
  "GET /api/sievyx/discover": {
    query: { limit?: number; min_size?: number };
    response: unknown;
  };
  "POST /api/sievyx/maneuver": {
    body: ManeuverIn;
    response: unknown;
  };
  "POST /api/sievyx/odd/gaps": {
    body: OddIn;
    response: unknown;
  };
  "GET /api/sievyx/priority": {
    query: { session_id?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/signs/recognize": {
    query: { session_id: string; limit?: number | null };
    response: unknown;
  };
  "GET /api/signs/taxonomy": {
    response: unknown;
  };
  "POST /api/superpixels/{frame_id}": {
    path: { frame_id: string };
    query: { n?: number };
    response: unknown;
  };
  "POST /api/synth/build": {
    body: services__api__routers__synth__BuildIn;
    response: unknown;
  };
  "POST /api/synth/plan": {
    body: PlanIn;
    response: unknown;
  };
  "GET /api/synth/runs": {
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/system/about": {
    response: AboutOut;
  };
  "GET /api/system/checks": {
    response: ChecksOut;
  };
  "GET /api/system/features": {
    response: FeaturesOut;
  };
  "GET /api/system/llm-usage": {
    query: { days?: number };
    response: unknown;
  };
  "GET /api/system/models": {
    response: unknown;
  };
  "POST /api/system/models/release": {
    query: { key?: string | null };
    response: unknown;
  };
  "GET /api/system/resources": {
    response: unknown;
  };
  "PATCH /api/track-events/{event_id}": {
    body: TrackEventPatch;
    path: { event_id: string };
    response: unknown;
  };
  "DELETE /api/track-events/{event_id}": {
    path: { event_id: string };
    response: unknown;
  };
  "POST /api/tracklets/objects/{object_id}/keyframe": {
    body: KeyframeIn;
    path: { object_id: string };
    response: unknown;
  };
  "POST /api/tracklets/objects/{object_id}/propagate": {
    path: { object_id: string };
    query: { direction?: string; frames?: number; refine?: boolean };
    response: unknown;
  };
  "GET /api/tracklets/stats/summary": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "GET /api/tracklets/{track_id}": {
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracklets/{track_id}/attributes": {
    body: TrackAttrsIn;
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracklets/{track_id}/derive": {
    path: { track_id: string };
    query: { method?: string; overwrite_human?: boolean };
    response: unknown;
  };
  "GET /api/tracklets/{track_id}/drift": {
    path: { track_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/tracklets/{track_id}/suggest-keyframes": {
    path: { track_id: string };
    query: { budget?: number };
    response: unknown;
  };
  "POST /api/tracks/judge-tubes": {
    body: JudgeTubesIn;
    response: unknown;
  };
  "POST /api/tracks/retrack": {
    query: { session_id: string };
    response: unknown;
  };
  "GET /api/tracks/{track_id}": {
    path: { track_id: string };
    response: unknown;
  };
  "DELETE /api/tracks/{track_id}": {
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/accept": {
    body: AcceptTrackIn;
    path: { track_id: string };
    response: unknown;
  };
  "GET /api/tracks/{track_id}/attribute-timeline": {
    path: { track_id: string };
    query: { key: string };
    response: unknown;
  };
  "GET /api/tracks/{track_id}/changepoints": {
    path: { track_id: string };
    query: { source?: string };
    response: unknown;
  };
  "GET /api/tracks/{track_id}/driving-events": {
    path: { track_id: string };
    response: unknown;
  };
  "GET /api/tracks/{track_id}/events": {
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/events": {
    body: TrackEventIn;
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/intent/propose": {
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/intent/set": {
    body: IntentSetIn;
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/intent/vlm": {
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/interpolate": {
    path: { track_id: string };
    query: { method?: string; anchor_policy?: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/interpolate-keyframed": {
    path: { track_id: string };
    query: { method?: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/merge": {
    body: MergeTrackIn;
    path: { track_id: string };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/relabel": {
    body: RelabelTrackIn;
    path: { track_id: string };
    response: unknown;
  };
  "GET /api/tracks/{track_id}/seg4d-consistency": {
    path: { track_id: string };
    query: { window?: number };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/smooth": {
    path: { track_id: string };
    query: { window?: number };
    response: unknown;
  };
  "POST /api/tracks/{track_id}/split": {
    body: SplitTrackIn;
    path: { track_id: string };
    response: unknown;
  };
  "GET /api/tracks/{track_id}/tube-verdict": {
    path: { track_id: string };
    response: unknown;
  };
  "GET /api/training": {
    query: { limit?: number };
    response: TrainingJobOut[];
  };
  "GET /api/training/registry": {
    response: ModelLineOut[];
  };
  "GET /api/training/runs": {
    query: { limit?: number };
    response: ModelRunOut[];
  };
  "GET /api/training/runs/{run_id}/curve": {
    path: { run_id: string };
    response: unknown;
  };
  "POST /api/training/start": {
    body: TrainingStartIn;
    response: unknown;
  };
  "POST /api/training/sweep": {
    body: services__api__routers__training__SweepIn;
    response: unknown;
  };
  "GET /api/training/sweep/{name}": {
    path: { name: string };
    query: { metric?: string };
    response: unknown;
  };
  "GET /api/training/tasks": {
    response: TrainingTaskOut[];
  };
  "GET /api/training/{job_id}": {
    path: { job_id: string };
    response: TrainingJobOut;
  };
  "POST /api/training/{job_id}/cancel": {
    path: { job_id: string };
    response: unknown;
  };
  "GET /api/triage": {
    query: { states?: string; session_id?: string | null; klass?: string | null; city?: string | null; flywheel?: string | null; control?: boolean; limit?: number };
    response: TriageRow[];
  };
  "GET /api/triage/ordering": {
    query: { gold_id?: string | null };
    response: unknown;
  };
  "POST /api/upload/abort": {
    body: MultipartAbortIn;
    response: unknown;
  };
  "POST /api/upload/complete": {
    body: MultipartCompleteIn;
    response: unknown;
  };
  "POST /api/upload/init": {
    body: MultipartInitIn;
    response: unknown;
  };
  "POST /api/upload/presign-put": {
    body: MultipartInitIn;
    response: unknown;
  };
  "POST /api/upload/sign": {
    body: MultipartSignIn;
    response: unknown;
  };
  "GET /api/users": {
    query: { limit?: number; offset?: number };
    response: UsersOut;
  };
  "POST /api/users": {
    body: UserCreateIn;
    response: unknown;
  };
  "POST /api/users/invite": {
    body: InviteIn;
    response: unknown;
  };
  "GET /api/users/me": {
    response: UserOut;
  };
  "POST /api/users/{user_id}/revoke-tokens": {
    path: { user_id: string };
    response: Record<string, unknown>;
  };
  "POST /api/users/{user_id}/sign-in-link": {
    path: { user_id: string };
    response: unknown;
  };
  "POST /api/users/{user_id}/token": {
    path: { user_id: string };
    response: unknown;
  };
  "POST /api/verdyx/blind-audit/seed": {
    body: SeedAuditIn;
    response: unknown;
  };
  "GET /api/verdyx/blind-audit/{audit_id}": {
    path: { audit_id: string };
    response: unknown;
  };
  "POST /api/verdyx/blind-audit/{audit_id}/close": {
    path: { audit_id: string };
    query: { reason?: string };
    response: unknown;
  };
  "GET /api/verdyx/blind-audit/{audit_id}/estimate": {
    path: { audit_id: string };
    response: unknown;
  };
  "POST /api/verdyx/blind-audit/{audit_id}/mark-labeled": {
    path: { audit_id: string };
    response: unknown;
  };
  "POST /api/verdyx/blind-audit/{audit_id}/score": {
    path: { audit_id: string };
    query: { min_frames?: number };
    response: unknown;
  };
  "GET /api/verdyx/blind-audits": {
    query: { run_id?: string | null; limit?: number };
    response: unknown;
  };
  "POST /api/verdyx/counterfactual/{run_id}": {
    path: { run_id: string };
    query: { perturbations?: string | null; max_frames?: number; score_thr?: number };
    response: unknown;
  };
  "POST /api/verdyx/evaluate": {
    body: services__api__routers__verdyx__EvalIn;
    response: unknown;
  };
  "GET /api/verdyx/hierarchical": {
    query: { run_id: string; gold_id?: string | null; score_thr?: number };
    response: unknown;
  };
  "GET /api/verdyx/matrix": {
    query: { champion: string; challenger: string; slice_metric?: string };
    response: unknown;
  };
  "GET /api/verdyx/model/{model_version}/evals": {
    path: { model_version: string };
    query: { limit?: number };
    response: unknown;
  };
  "GET /api/verdyx/pairs": {
    response: unknown;
  };
  "POST /api/verdyx/safety/recall": {
    body: SafetyIn;
    response: unknown;
  };
  "POST /api/verdyx/shadow/triage": {
    body: TriageIn;
    response: unknown;
  };
  "GET /api/verdyx/sim/capability": {
    response: unknown;
  };
  "POST /api/verdyx/stats/bootstrap": {
    body: BootstrapIn;
    response: unknown;
  };
  "POST /api/verdyx/stats/significance": {
    body: SignificanceIn;
    response: unknown;
  };
  "GET /api/vlm-dataset/export": {
    query: { session_id?: string | null };
    response: unknown;
  };
  "POST /api/vlm-targets/{target_id}/status": {
    path: { target_id: string };
    query: { status: string };
    response: unknown;
  };
  "POST /api/workforce": {
    body: WorkforceIn;
    response: unknown;
  };
  "POST /api/workforce/dispatch": {
    body: services__api__routers__workforce__DispatchIn;
    response: unknown;
  };
  "GET /api/workforce/rating": {
    response: unknown;
  };
  "GET /api/workforce/{workforce_id}/pending": {
    path: { workforce_id: string };
    query: { limit?: number };
    response: unknown;
  };
  "POST /api/workforce/{workforce_id}/return": {
    path: { workforce_id: string };
    response: unknown;
  };
  "POST /api/workforce/{workforce_id}/return/upload": {
    path: { workforce_id: string };
    response: unknown;
  };
  "GET /metrics": {
    response: unknown;
  };
};

/** The body a request to this operation takes. */
export type Body<K extends keyof Operations> = Operations[K] extends { body: infer B } ? B : never;
/** What this operation returns (`unknown` until the server declares it). */
export type Result<K extends keyof Operations> = Operations[K]["response"];

export const API_SCHEMA_STATS = { operations: 787, typedResponses: 90, schemas: 291 } as const;
