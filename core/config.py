"""Typed, env-overridable configuration.

Loads configs/default.yaml then overlays environment variables (LBX_ prefix, __ nesting).
Access through get_settings(), which is cached for the process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class PostgresSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "labelox"
    password: str = "labelox"
    db: str = "labeloxav"

    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_dsn(self) -> str:
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class MinioSettings(BaseModel):
    endpoint: str = "http://localhost:9000"
    access_key: str = "labelox"
    secret_key: str = "labelox123"
    secure: bool = False
    bucket: str = "labeloxav"
    # Public-facing endpoint for presigned URLs handed to browsers/buyers. Defaults to the internal
    # endpoint (works only on-host); set to the externally reachable S3/MinIO URL in any real deployment
    # so download/upload links resolve off the host.
    public_endpoint: str = ""


class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class RedpandaSettings(BaseModel):
    brokers: str = "localhost:19092"


class IngestSettings(BaseModel):
    target_fps: float = 3.0
    blur_threshold: float = 60.0
    exposure_low: float = 12.0
    exposure_high: float = 243.0
    clip_fraction_max: float = 0.45
    max_width: int = 1920  # downscale frames wider than this on ingest (0 disables); 4K wastes
                           # label compute. Detectors run at imgsz anyway; masks/review want 1080p.


class GpuSettings(BaseModel):
    device: str = "cuda:0"
    mode: str = "sequential"  # sequential | concurrent
    vram_total_mb: int = 16000
    vram_headroom_mb: int = 1500


class YoloSettings(BaseModel):
    # Path A, deterministic head detector. Production target: yolo26l.pt (NMS-free e2e).
    # Realized today with yolo11l.pt (see the model-substitution note).
    weights: str = "yolo11l.pt"
    imgsz: int = 1280
    half: bool = True
    conf: float = 0.20


class OpenVocabSettings(BaseModel):
    # Path B, open-vocab detect + segment. Production target: SAM 3.1 PCS (sam3.pt).
    # Realized today as YOLO-World (text concept -> boxes) + SAM (box -> mask).
    detector_weights: str = "yolov8s-worldv2.pt"
    seg_weights: str = "sam_b.pt"
    half: bool = True
    conf: float = 0.12          # open-vocab floor: low enough for the long tail, high enough that
                                # most boxes are real (0.05 buried the gate in noise)
    max_boxes: int = 40         # cap SAM box prompts per frame (VRAM + latency)


class VlmSettings(BaseModel):
    enabled: bool = True
    # ollama is the working default on this box (bitsandbytes 4-bit is unusable on Blackwell here,
    # and transformers 4.47 lacks Qwen3-VL). transformers backend kept for boxes where it works.
    backend: str = "ollama"  # ollama | transformers | vllm
    model: str = "Qwen/Qwen2-VL-2B-Instruct"  # transformers backend; target: Qwen3-VL-4B
    ollama_url: str = "http://localhost:11434"
    ollama_tag: str = "qwen2.5vl:7b"  # locally available; target: qwen3-vl:4b
    quant: str = "nf4"
    max_context: int = 8192
    crop_margin: float = 0.15
    max_calls_per_session: int = 400   # hard budget; VLM is the costliest token (Principle 08)
    max_calls_per_frame: int = 2       # per-frame cap so a busy frame cannot stall the pipeline
    shortlist_size: int = 20           # classes offered to the VLM (siblings + cross-superclass anchors)
    vote_count: int = 1                # N-vote agreement (>1 enables adversarial verify across crops/temps)
    cross_vote_min: float = 0.6        # required agreement fraction to accept a cross-superclass override
    timeout_s: float = 60.0


class ClipSettings(BaseModel):
    model: str = "ViT-B/32"   # CLIP/SigLIP backbone for embeddings (frame/crop + text)
    crop_margin: float = 0.1


# ---- Phase 2 Perception Depth ----
class LaneSettings(BaseModel):
    model: str = "clrernet"        # clrernet | ufldv2 (pod) | classical (local fallback)
    backend: str = "local"         # local (classical proposal) | pod (CLRerNet via cloud seam)
    max_lanes: int = 6
    control_points: int = 5        # control points per fitted spline


class DrivableSettings(BaseModel):
    backend: str = "local"         # local (sam_b click + grounding) | pod (SAM 3.1 PCS concept prompts)
    concepts: list[str] = ["drivable road", "non-drivable area", "unpaved or unmarked road"]


class SignSettings(BaseModel):
    taxonomy_path: str = "ontology/signs_in_v0.yaml"
    siglip_scale: float = 100.0    # softmax temperature on the zero-shot logits
    vlm_for_unusual: bool = True   # read unusual / text-bearing signs with Qwen-VL (duty-cycled)


class OcrSettings(BaseModel):
    backend: str = "local"         # local (Qwen/Ollama) | pod (PaddleOCR Indic)
    langs: list[str] = ["en", "devanagari", "ta", "te", "kn"]
    conf: float = 0.4
    plate_iou_exclude: float = 0.2  # drop any text region overlapping a plate bbox by this IoU


class ModelsSettings(BaseModel):
    yolo: YoloSettings = YoloSettings()
    openvocab: OpenVocabSettings = OpenVocabSettings()
    vlm: VlmSettings = VlmSettings()
    clip: ClipSettings = ClipSettings()
    lane: LaneSettings = LaneSettings()
    drivable: DrivableSettings = DrivableSettings()
    sign: SignSettings = SignSettings()
    ocr: OcrSettings = OcrSettings()


class FusionSettings(BaseModel):
    iou_match: float = 0.55
    iom_match: float = 0.65
    centroid_px: float = 48.0
    mask_box_disagree_iou: float = 0.70
    class_priors: dict = Field(default_factory=dict)


class CalibrateSettings(BaseModel):
    method: str = "temperature"   # temperature | isotonic
    temperature: float = 1.5
    agreement_bonus: float = 0.10
    disagreement_penalty: float = 0.25
    isotonic_uri: str | None = None  # s3 uri of a fitted isotonic curve (JSON knots); used when
                                     # method=isotonic so a calibrated 0.95 means ~95% precision (M9)


class GateSettings(BaseModel):
    auto_accept: float = 0.95
    review_low: float = 0.60
    force_review_on_rare: bool = True
    force_review_on_mask_box_disagree: bool = True


class TrackerSettings(BaseModel):
    iou_match: float = 0.3
    max_age_frames: int = 5
    min_track_len: int = 2
    backend: str = "bot_sort"        # bot_sort (Kalman + DINOv3 appearance) | greedy (legacy IoU)
    appearance_weight: float = 0.45  # weight of DINOv3 cosine vs IoU in the association cost
    reid_cos: float = 0.55           # appearance cosine that alone makes a match feasible (re-entry)


class EventSettings(BaseModel):
    hard_brake_decel: float = -3.0       # m/s^2 on ego longitudinal
    static_disp_frac: float = 0.04       # max centroid displacement (frac of width) for "static"
    static_min_frames: int = 4
    congestion_min_objects: int = 8
    congestion_max_speed_frac: float = 0.01
    cut_in_area_growth: float = 1.35     # box-area ratio signalling a closing cut-in
    cut_in_center_frac: float = 0.33     # how close to ego column (image center) counts as in-path
    near_miss_ttc_s: float = 2.0
    wrong_side_frames: int = 4
    shoulder_margin_frac: float = 0.15


class IntelligenceSettings(BaseModel):
    tracker: TrackerSettings = TrackerSettings()
    events: EventSettings = EventSettings()


# ---- Data Intelligence Layer (Phase 1): embedding backbone + the capabilities riding on it ----
class IntelEmbedSettings(BaseModel):
    siglip2_model: str = "google/siglip2-so400m-patch14-384"  # 1152-d image + text (semantic, scene)
    siglip2_dim: int = 1152
    dinov3_model: str = "vit_base_patch16_dinov3.lvd1689m"     # timm non-gated mirror, 768-d (visual)
    dinov3_dim: int = 768
    crop_margin: float = 0.15      # context margin around an object box for crop embeddings
    batch_size: int = 16
    device: str = "cuda:0"         # falls back to cpu when CUDA is unavailable (overnight backfill)


class DedupSettings(BaseModel):
    phash_hamming: int = 6         # stage 1: perceptual-hash candidate threshold (within a session)
    dino_cos: float = 0.95         # stage 2: DINOv3 cosine confirm threshold


class SceneSettings(BaseModel):
    low_conf: float = 0.45         # per-axis confidence below which the VLM may confirm
    vlm_confirm: bool = False      # duty-cycled Qwen-VL confirmation of low-confidence axes
    max_vlm_per_session: int = 50


class DiscoverySettings(BaseModel):
    min_cluster_size: int = 15     # HDBSCAN minimum cluster size
    outlier_quantile: float = 0.95  # distance-to-centroid quantile flagged as embedding_outlier


class ExtractSettings(BaseModel):
    target_budget_frac: float = 0.5  # keep at most this fraction of the fixed-rate frames
    scene_change_cos: float = 0.85   # DINOv3 cosine vs previous kept below this = scene change (keep)
    diversity_cos: float = 0.92      # too similar to an already-selected frame = drop
    min_gap_frames: int = 2


class IntelSettings(BaseModel):
    embed: IntelEmbedSettings = IntelEmbedSettings()
    dedup: DedupSettings = DedupSettings()
    scene: SceneSettings = SceneSettings()
    discovery: DiscoverySettings = DiscoverySettings()
    extract: ExtractSettings = ExtractSettings()


class PiiSettings(BaseModel):
    # Gate A (DPDPA): face + license-plate blur before any frame reaches the object store. Mandatory
    # by default: a frame with identifiable faces/plates cannot be legally resold. CPU keeps PII off
    # the 16 GB autolabel GPU budget. Weights are swappable (verified at build, like YOLO26/SAM3).
    enabled: bool = True
    blur_method: str = "gaussian"     # gaussian | pixelate
    kernel: int = 51                  # odd; gaussian kernel / pixelate block size
    face_conf: float = 0.50
    plate_conf: float = 0.35
    face_weights: str = ".scratch/models/pii/face_yunet.onnx"
    plate_weights: str = ".scratch/models/pii/plate_yolov8.pt"
    device: str = "cpu"
    # DPDPA: when the gate is on, BOTH face and plate detectors must be available, otherwise ingestion
    # fails loud rather than silently passing un-blurred plates into the object store. Set false only for
    # face-only corpora where plates are provably absent.
    plate_mandatory: bool = True
    # Source for `make pii-models` to fetch a license-plate detector to plate_weights. Override with
    # LBX_PII__PLATE_URL if this mirror moves; an Ultralytics-loadable .pt is expected. Hugging Face now
    # requires a token even for public files, so run `HF_TOKEN=... make pii-models`.
    plate_url: str = (
        "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/"
        "license-plate-finetune-v1n.pt"
    )


class M9Settings(BaseModel):
    # Gate B (measurement): gold-set eval, Safe-mIoU weighting, calibration reliability.
    safety_weight: float = 2.0        # scales the Safe-mIoU penalty for unsafe class confusions
    mask_iou_thresh: float = 0.5
    ece_bins: int = 10


class TrainingSettings(BaseModel):
    # In-app training platform. One worker drains the training_job queue serially on the single GPU.
    worker_poll_s: float = 5.0
    default_epochs: int = 20
    default_imgsz: int = 960
    default_batch: int = 12
    advisory_lock_key: int = 815  # Postgres advisory-lock id used as the GPU mutex across processes
    vram_required_mb: int = 4000  # preflight floor before a train starts (small jobs need ~3-4 GB)


class CloudSettings(BaseModel):
    # Hybrid GPU: the RunPod A100 environment for heavy real-model work. Shared by the cloud/ runbook
    # scripts and the app so they agree on names/paths. The pod itself is provisioned on demand.
    volume_name: str = "labeloxav-vol"
    workspace: str = "/workspace"
    ckpt_dir: str = "/workspace/ckpts"
    gpu_pref: list[str] = Field(default_factory=lambda: ["A100 80GB", "H100 PCIe"])
    budget_cap_usd: float = 50.0
    image: str = ""  # pinned once the runbook discovers a CUDA 12.8 devel image


class OntologySettings(BaseModel):
    path: str = "ontology/labelox_in_v0.yaml"


class PathsSettings(BaseModel):
    scratch: str = ".scratch"


# ---- Phase 3 Multi-Sensor and Spatial ----
class LensIntrinsics(BaseModel):
    model: str = "pinhole"          # pinhole | fisheye (wide STURDeCAM31 lenses are fisheye)
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = []          # distortion coefficients (k1,k2,p1,p2[,k3] / fisheye k1..k4)
    fov_deg: float                  # configured horizontal FOV, for the FOV check


class RigSettings(BaseModel):
    # Nominal intrinsics per lens type (e-con STURDeCAM31), used until real per-camera calibration is
    # ingested. Image plane assumed 1920x1080; cx/cy at center.
    # Defined at a reference image width of 1920. fx is consistent with fov_deg: narrow pinhole 37deg
    # -> fx = 1920 / (2 tan(18.5deg)) ~= 2870; wide fisheye 120deg (equidistant) -> fx = 1920 / 2.094 ~= 917.
    lenses: dict[str, LensIntrinsics] = {
        "narrow": LensIntrinsics(model="pinhole", fx=2870.0, fy=2870.0, cx=960.0, cy=540.0,
                                 dist=[-0.12, 0.04, 0.0, 0.0, 0.0], fov_deg=37.0),
        "wide": LensIntrinsics(model="fisheye", fx=917.0, fy=917.0, cx=960.0, cy=540.0,
                               dist=[-0.05, 0.01, -0.004, 0.0007], fov_deg=120.0),
    }
    ref_width: int = 1920
    # per-camera lens type (the narrow front + wide surround typical of the rig)
    camera_lens: dict[str, str] = {"cam_f": "narrow", "cam_l": "wide", "cam_r": "wide", "cam_b": "wide"}
    # nominal per-camera mounting yaw (deg) relative to vehicle forward, composed into world georef so
    # side/rear cameras are not rotated up to 180deg. Real extrinsics override these once calibrated.
    camera_yaw_deg: dict[str, float] = {"cam_f": 0.0, "cam_l": -90.0, "cam_r": 90.0, "cam_b": 180.0}


class SpatialSettings(BaseModel):
    reproj_px_warn: float = 1.5
    reproj_px_fail: float = 3.0
    fov_tolerance_deg: float = 8.0          # implied FOV vs configured beyond this -> fail (lens mix)
    time_offset_ns_warn: int = 2_000_000    # 2 ms camera/IMU/GNSS skew
    time_offset_ns_fail: int = 10_000_000   # 10 ms
    imu_hz_target: float = 200.0
    imu_hz_tolerance: float = 5.0           # 247 Hz vs 200 Hz target -> fail
    osm_extract_path: str = ".scratch/maps/bangalore-roads.osm"
    map_formats: list[str] = ["lanelet2", "opendrive"]
    ipm_horizon_frac: float = 0.55          # road plane assumed below this image row fraction
    camera_height_m: float = 1.5            # forward-camera height above the road plane (IPM)
    camera_pitch_deg: float = 0.0           # downward pitch of the optical axis
    fuse_cluster_m: float = 4.0             # multi-drive fusion: merge same-kind elements within this radius
    map_region: str = "bangalore"


class ActiveLearnSettings(BaseModel):
    # M4.0 value-ranked selection. Weights sum to 1; the band is the most-informative confidence window.
    w_uncertainty: float = 0.40
    w_diversity: float = 0.25
    w_rarity: float = 0.20
    w_error_prone: float = 0.15
    uncertainty_lo: float = 0.55         # below this is hopeless, above uncertainty_hi is easy
    uncertainty_hi: float = 0.92
    diversity_knn: int = 5               # near-duplicate suppression neighbourhood
    sec_per_item: float = 30.0           # human-hour budget conversion (seconds per item to label)
    retrain_min_new: int = 50            # new corrections/controls before the loop fires a fine-tune


class LakeFSSettings(BaseModel):
    # M4.3 git-like dataset versioning over the object store. Defaults match docker-compose.
    endpoint: str = "http://localhost:8001"
    access_key: str = "AKIALABELOXAVKEY01"
    secret_key: str = "labeloxavlakefssecretkey0123456789abcd"
    repo: str = "labeloxav"
    storage_namespace: str = "s3://labeloxav/lakefs/labeloxav"
    default_branch: str = "main"


class RelabelSettings(BaseModel):
    # M4.2 selective apply. Never overwrite these sources; auto-apply only confident improvements.
    never_touch_sources: list[str] = ["human"]
    auto_apply_min_conf: float = 0.92
    auto_apply_min_uplift: float = 0.08  # new conf must beat old by this to auto-apply
    regression_margin: float = 0.10      # new conf below old by this on a changed class -> flag regression


class GovernSettings(BaseModel):
    # M4.4 governance. Safety is never automated to zero.
    safety_affinity_min: float = 0.8     # affinity_cost >= this is a safety-critical confusion (VRU vs non-VRU = 1.0)
    safe_miou_max_drop: float = 0.0      # a challenger may not regress Safe-mIoU at all
    min_map_uplift: float = 0.005        # challenger must strictly beat champion mAP by at least this (no ties)
    control_sample_rate: float = 0.02    # fraction of auto-accepts mirrored to human control review
    control_precision_floor: float = 0.97  # measured auto-accept precision below this pauses auto-promotion
    drift_psi_breach: float = 0.2        # population-stability-index breach on embeddings / label dist
    ood_zscore: float = 3.0              # input this far from the corpus centroid defers regardless of confidence
    offhours_utc: list[int] = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3]  # when the controller may burst the A100
    controller_poll_s: float = 60.0      # the autonomy daemon ticks the controller this often
    controller_lock_key: int = 816       # Postgres advisory-lock id so only one controller daemon runs


class Phase4Settings(BaseModel):
    activelearn: ActiveLearnSettings = ActiveLearnSettings()
    lakefs: LakeFSSettings = LakeFSSettings()
    relabel: RelabelSettings = RelabelSettings()
    govern: GovernSettings = GovernSettings()


class AuthSettings(BaseModel):
    # Deny-by-default API auth. When enabled, every mutating /api request needs a known X-Lbx-User-Id,
    # and role floors gate destructive/governance routes. Reads stay open. Disable only for unit tests.
    enabled: bool = True


class LidarSettings(BaseModel):
    # 3D LiDAR module. The BluRabbit fleet is camera only, so pseudo-LiDAR (camera depth lift) is the
    # default source; real LiDAR and public datasets normalize to the same internal cloud.
    source_default: str = "pseudo"          # lidar | pseudo | dataset
    # Pinned metric depth checkpoint for pseudo-LiDAR. The Outdoor variant is the driving model (metric
    # depth in metres, trained on VKITTI); Small runs interactively on the local 5080.
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    ground_method: str = "ransac"           # patchwork | ransac
    ground_dist_thresh_m: float = 0.2       # RANSAC plane inlier distance
    ground_max_iter: int = 300
    denoise_nb_neighbors: int = 20          # statistical outlier removal neighbourhood
    denoise_std_ratio: float = 2.0          # statistical outlier removal std multiplier
    denoise_radius_m: float = 0.5           # radius outlier removal radius
    denoise_min_points: int = 8             # radius outlier removal minimum neighbours
    voxel_size_m: float = 0.05              # voxel downsample for bulk operations
    viewer_max_points: int = 400000         # decimation ceiling for interactive rendering
    cloud_prefix: str = "clouds"            # object-store key prefix for stored clouds
    calib_reproj_warn_px: float = 2.0       # LiDAR-camera reprojection residual: warn above
    calib_reproj_fail_px: float = 5.0       # LiDAR-camera reprojection residual: fail and exclude above
    calib_drift_ratio: float = 1.5          # residual grown past this multiple of baseline flags drift
    quality_min_points: int = 500           # below this a cloud is sparse / a missing scan
    quality_max_empty_wedge_deg: float = 90.0  # a 360 scan with a wider empty wedge is a partial scan
    # Phase 2 (3D annotation). The box source: lifted (2D-to-3D, robust on pseudo-LiDAR) is the default for
    # camera clouds; native (CenterPoint/PV-RCNN++) is for real LiDAR. 'auto' picks by the cloud source.
    box_source: str = "auto"                # auto | lifted | native
    lift_min_frustum_points: int = 12       # below this a frustum is too sparse to fit a cuboid
    lift_depth_gate_m: float = 8.0          # keep frustum points within this depth band of the nearest surface
    native_detector: str = "centerpoint"    # centerpoint | pv_rcnn_pp | bevfusion (OpenPCDet, via burst)
    native_ckpt: str = "centerpoint_nuscenes"   # pinned OpenPCDet checkpoint id
    segmenter: str = "ptv3"                  # ptv3 (Pointcept, via burst) | projected_2d (pseudo-LiDAR fallback)
    segmenter_ckpt: str = "ptv3-nuscenes-semseg"  # pinned Pointcept checkpoint id
    seg_low_conf: float = 0.5               # per-point segmentation confidence below this is flagged for review
    track3d_max_age: int = 5                 # frames a 3D track survives unmatched before termination
    track3d_min_hits: int = 2                # matches before a tentative 3D track is confirmed
    track3d_iou_thresh: float = 0.1          # 3D IoU association gate
    track3d_appearance_w: float = 0.3        # DINOv3 appearance weight in the association cost
    # Phase 3 (scene intelligence + export).
    extract_cluster_eps: float = 0.5         # DBSCAN radius for clustering non-ground points (metres)
    extract_cluster_min_points: int = 10
    pole_min_height_m: float = 2.0           # a pole is taller than this
    pole_max_footprint_m: float = 0.9        # and thinner than this (a thin vertical structure)
    building_min_facade_points: int = 150    # a facade plane needs at least this many points
    marking_intensity_pct: float = 80.0      # road points above this intensity percentile are markings
    traverse_grid_res_m: float = 0.5         # occupancy / free-space grid resolution
    traverse_obstacle_h_m: float = 0.3       # a ground-cell obstacle rises at least this high
    register_method: str = "gicp"            # icp | ndt | gicp scan alignment
    register_voxel_m: float = 0.3            # downsample voxel for registration
    register_min_fitness: float = 0.3        # below this the registration is flagged low-confidence
    loop_closure_radius_m: float = 8.0       # revisit detection radius for loop closure
    quality_max_dim_m: float = 25.0          # a box dimension above this is impossible
    quality_float_gap_m: float = 0.5         # a box bottom this far above the ground is floating
    quality_below_ground_m: float = 0.5      # a box bottom this far below the ground is below-ground
    quality_duplicate_iou: float = 0.6       # two boxes overlapping above this 3D IoU are duplicates
    quality_misalign_fill: float = 0.05      # a box with fewer enclosed points than this is misaligned


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LBX_",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=str(DEFAULT_CONFIG),
    )

    env: str = "local"
    log_level: str = "INFO"

    postgres: PostgresSettings = PostgresSettings()
    minio: MinioSettings = MinioSettings()
    redis: RedisSettings = RedisSettings()
    redpanda: RedpandaSettings = RedpandaSettings()
    ingest: IngestSettings = IngestSettings()
    gpu: GpuSettings = GpuSettings()
    models: ModelsSettings = ModelsSettings()
    fusion: FusionSettings = FusionSettings()
    calibrate: CalibrateSettings = CalibrateSettings()
    gate: GateSettings = GateSettings()
    intelligence: IntelligenceSettings = IntelligenceSettings()
    intel: IntelSettings = IntelSettings()  # Data Intelligence Layer (Phase 1)
    rig: RigSettings = RigSettings()        # Phase 3 camera rig layout + nominal intrinsics
    spatial: SpatialSettings = SpatialSettings()  # Phase 3 calibration + map thresholds
    pii: PiiSettings = PiiSettings()
    m9: M9Settings = M9Settings()
    training: TrainingSettings = TrainingSettings()
    cloud: CloudSettings = CloudSettings()
    ontology: OntologySettings = OntologySettings()
    paths: PathsSettings = PathsSettings()
    phase4: Phase4Settings = Phase4Settings()  # Phase 4 closed loop + governance
    auth: AuthSettings = AuthSettings()        # deny-by-default API auth
    lidar: LidarSettings = LidarSettings()     # 3D LiDAR module (ingestion, clean, viewer)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit init args, then env, then .env, then the YAML defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    _DEV_ENVS = {"local", "test", "ci", "dev"}

    @model_validator(mode="after")
    def _require_prod_secrets(self):
        """Outside a dev env, the known dev-default credentials are refused so a deployment cannot ship
        with `labelox`/`labelox123`. Set the real secrets via env (LBX_POSTGRES__PASSWORD, etc.)."""
        if self.env.lower() in self._DEV_ENVS:
            return self
        weak = []
        if self.postgres.password == "labelox":
            weak.append("LBX_POSTGRES__PASSWORD")
        if self.minio.secret_key == "labelox123":
            weak.append("LBX_MINIO__SECRET_KEY")
        if self.phase4.lakefs.secret_key.startswith("labeloxavlakefssecret"):
            weak.append("LBX_PHASE4__LAKEFS__SECRET_KEY")
        if weak:
            raise ValueError(
                f"env '{self.env}' is not a dev env but still uses default credentials; set: {', '.join(weak)}"
            )
        return self

    def scratch_path(self) -> Path:
        p = REPO_ROOT / self.paths.scratch
        p.mkdir(parents=True, exist_ok=True)
        return p

    def ontology_abspath(self) -> Path:
        return REPO_ROOT / self.ontology.path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
