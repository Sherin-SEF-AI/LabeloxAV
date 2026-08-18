"""Typed, env-overridable configuration.

Loads configs/default.yaml then overlays environment variables (LBX_ prefix, __ nesting).
Access through get_settings(), which is cached for the process.
"""

from __future__ import annotations

import os
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
    # Upper bound on variance-of-Laplacian: a real image is sharp at hundreds to low thousands; random noise
    # (a corrupted/garbage frame) is tens of thousands. Above this the frame is rejected as noise, not kept
    # as "maximally sharp". Validated against the corpus: real frames < 2000, noise frames 27000+.
    noise_blur_threshold: float = 8000.0
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
    backend: str = "ollama"  # ollama | llamacpp | transformers | vllm
    # llama-server (llama.cpp) as an alternative local backend. It takes a JSON schema in `response_format`
    # and compiles it to GBNF itself, so the shortlist is enforced during sampling rather than requested in
    # the prompt, and the prompt is then identical across objects and worth prefix-caching. Unset by
    # default: this changes which process serves Path C, and that is a deployment decision, not a default.
    llamacpp_url: str = "http://localhost:8080"
    llamacpp_model: str = ""   # empty means whatever the server has loaded
    model: str = "Qwen/Qwen2-VL-2B-Instruct"  # transformers backend; target: Qwen3-VL-4B
    ollama_url: str = "http://localhost:11434"
    ollama_tag: str = "qwen2.5vl:7b"  # locally available; target: qwen3-vl:4b
    # The model that judges existing labels, when it should differ from the one that proposes them.
    #
    # Separate because the two jobs are different and were measured separately. `ollama_tag` is read by
    # eight call sites: the Path C verifier, road-text and ANPR OCR, captioning, VLM QA and the ops agent.
    # The judge was measured against 118 human decisions and qwen3-vl:8b-instruct-q8_0 won on the axes that
    # matter (specificity 0.89 against 0.80, equal superclass sensitivity, and it actually abstains where
    # qwen2.5vl:7b returned `unsure` zero times in 547 crops). None of that says anything about naming an
    # object, which is what the other seven callers ask for, so changing one tag for all of them would be
    # generalising a measurement past what it covers.
    #
    # Empty means "use ollama_tag", so this costs nothing until somebody sets it.
    judge_tag: str = ""
    quant: str = "nf4"
    max_context: int = 8192
    crop_margin: float = 0.15
    max_calls_per_session: int = 400   # hard budget; VLM is the costliest token (Principle 08)
    max_calls_per_frame: int = 2       # per-frame cap so a busy frame cannot stall the pipeline
    shortlist_size: int = 20           # classes offered to the VLM (siblings + cross-superclass anchors)
    vote_count: int = 1                # N-vote agreement (>1 enables adversarial verify across crops/temps)
    cross_vote_min: float = 0.6        # required agreement fraction to accept a cross-superclass override
    timeout_s: float = 60.0
    # Provider routing (services/llm/router.py). ollama is the always-available local floor; groq is the fast
    # cloud path. Text (nl/intent) and vision (Path C) are chosen independently so text can go cloud while
    # vision stays local. A missing GROQ_API_KEY, a cloud failure, or an open circuit all fall back to ollama,
    # so the cloud is never a hard dependency.
    text_provider: str = "ollama"      # ollama | groq | anthropic  (nl.py / intent, text-only, no media leaves the box)
    vision_provider: str = "ollama"    # ollama | groq | anthropic  (Path C VLM verifier)
    escalate_provider: str | None = None  # optional stronger provider re-asked only on a not-confident verdict
    allow_cloud_media: bool = True     # False keeps crops local even when the provider is a cloud one (data residency)
    breaker_threshold: int = 3         # consecutive cloud failures before the circuit opens
    breaker_cooldown_s: float = 60.0   # how long the circuit stays open before a half-open retry


class GroqSettings(BaseModel):
    """Groq cloud inference (OpenAI-compatible). Optional: an empty api_key means 'not configured' and the
    router uses ollama, so local dev needs zero Groq setup. The key comes from the environment
    (LBX_GROQ__API_KEY or GROQ_API_KEY), never stored in the app; it is an outbound credential, not a secret
    that protects LabeloxAV, so it is not part of _require_prod_secrets."""
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    text_model: str = "llama-3.3-70b-versatile"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    timeout_s: float = 30.0


class AnthropicSettings(BaseModel):
    """Anthropic cloud inference (Messages API). Optional in exactly the way Groq is: an empty api_key means
    "not configured" and the router falls back to the local provider, so nothing here is a hard dependency.

    Present alongside Groq rather than instead of it because they answer different questions. Groq is the
    fast cloud path for the autolabel stream; this is the judge, re-read on a crop the corpus could not
    otherwise get a verdict on, where being right matters more than being quick.

    The key comes from the environment (LBX_ANTHROPIC__API_KEY or ANTHROPIC_API_KEY) and is an outbound
    credential rather than a secret protecting LabeloxAV, so it stays out of _require_prod_secrets.
    """
    api_key: str = ""
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"
    text_model: str = "claude-sonnet-5"
    vision_model: str = "claude-sonnet-5"
    # The judge re-reads one crop and answers with a short JSON object, so the ceiling only has to cover a
    # verdict plus a one-line reason. Kept tight because a runaway generation on a 300-crop batch is a real
    # cost, not a theoretical one.
    max_tokens: int = 1024
    timeout_s: float = 60.0


class BillingSettings(BaseModel):
    """Prices for metered deliveries, in INR per unit.

    Zero by default, and deliberately so: a system that starts charging because somebody deployed a new
    version is worse than one that meters quantity and leaves the rate to be set explicitly. A price of
    zero still records the delivery, the quantity and the certification status, which is the part that
    cannot be reconstructed after the fact.
    """
    unit_price_inr: dict[str, float] = {"export": 0.0, "inference": 0.0, "judge": 0.0}
    default_account: str = "default"


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
    # How far the best sign type must beat the best "not a sign" prompt before the type is written. Cosine
    # similarities from SigLIP 2 are already L2-normalised, so this is in similarity units, not probability.
    # Set from the corpus: real sign crops clear the negatives comfortably, while vehicles and blank patches,
    # which the previous version typed confidently, do not. Raising it types fewer signs and gets fewer
    # wrong; a sign left untyped costs a lookup, a sign typed wrongly is exported and believed.
    min_margin: float = 0.02
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
    # Post-fusion de-duplication: after clustering, suppress a fused box that overlaps a higher-confidence
    # same-object box. Two boxes are the same object when their classes are compatible (same class, same l1
    # superclass, or one is a fallback) AND they either overlap heavily (IoU) or one nests in the other
    # (intersection-over-min, for the many-scales-of-one-car case). Different-l1 pairs (rider on a bike)
    # are never merged. This turns "one object wrapped in many boxes" into one box.
    dedupe_iou: float = 0.70
    dedupe_iom: float = 0.85


class CalibrateSettings(BaseModel):
    method: str = "temperature"   # temperature | isotonic
    temperature: float = 1.5
    agreement_bonus: float = 0.10
    disagreement_penalty: float = 0.25
    isotonic_uri: str | None = None  # s3 uri of a fitted isotonic curve (JSON knots); used when
                                     # method=isotonic so a calibrated 0.95 means ~95% precision (M9)


class ReasonerSettings(BaseModel):
    """The reasoning layer between detection and label.

    On by default. The Tier 1 checks are deterministic and cost microseconds, and the failure they catch
    (a confident wrong label auto-accepted into the training set) is the one that has actually hurt this
    corpus. Turning it off is for measuring what it is worth, not for running without it.
    """

    enabled: bool = True
    # Which collectors to run. Empty means all of them; naming a subset is how a check under suspicion is
    # isolated without editing code.
    checks: list[str] = []
    # Whether a conflicted detection may escalate to the VLM. Separate from the VLM's own enable flag,
    # because a deployment may want the free Tier 1 checks and not the GPU cost of Tier 2.
    adjudicate: bool = True
    # Ceiling on Tier 2 calls per session, on top of the VLM budget. The reasoner escalates only conflicted
    # objects, which should be a small fraction; a ceiling here means a bad prior that suddenly conflicts
    # everywhere cannot silently consume the whole GPU budget.
    max_adjudications_per_session: int = 200
    # Write the full trace onto provenance. Off would make the layer unauditable and unmeasurable, so it
    # exists only for a deployment with a hard storage constraint.
    record_trace: bool = True


class GateSettings(BaseModel):
    auto_accept: float = 0.95          # benign default; calibrated, so this is a precision floor
    safety_auto_accept: float = 0.99   # M-Q.4: safety-critical classes (VRU, animal) must be near-certain
    review_low: float = 0.60
    # Strict escape hatch: rare/fallback classes never auto-accept. Off by default in favour of the smarter
    # agreement+VLM rule below; flip on to fully freeze the long tail (e.g. a fresh ontology).
    force_review_on_rare: bool = False
    force_review_on_mask_box_disagree: bool = True
    # M-Q.4: a rare/fallback class earns auto-accept only with cross-path agreement AND VLM confirmation,
    # never on one model's output. Off -> a rare class auto-accepts on agreement alone like a common class.
    rare_needs_agreement_and_vlm: bool = True


class QualitySettings(BaseModel):
    # M-Q.4 quality reviewer: geometric/contextual checks that demote nonsense before the auto-accept queue.
    # Each threshold backs a rule that catches a real live error (sky boxes, impossible sizes, tyre-as-
    # vehicle, duplicates, pedestrian-in-car). Deterministic and model-free, so it runs on every object.
    enabled: bool = True
    horizon_frac: float = 0.45     # a ground object whose box sits entirely above this frame fraction is sky
    dup_iou: float = 0.85          # IoU above which two boxes are the same detection (keep the higher conf)
    part_contain: float = 0.80     # a small vehicle box this contained in a much larger one is a part (wheel)
    vru_contain: float = 0.70      # a VRU box this contained in a four-wheeler/heavy box is implausible
    # a single instance spanning more than this fraction of the frame is the "everything" box: a fusion
    # artifact that wraps most of the scene. An absolute ceiling above every per-class band, so even a
    # heavy-vehicle box cannot slip a full-frame span through. Demote (route to review), never delete.
    max_area_frac: float = 0.85
    # per-superclass (l1) plausible area as a fraction of the frame; outside is an impossible size
    size_bounds: dict[str, list[float]] = Field(default_factory=lambda: {
        "two_wheeler": [0.0004, 0.45], "three_wheeler": [0.0008, 0.55], "four_wheeler": [0.0008, 0.75],
        "heavy": [0.0025, 0.92], "vru": [0.0002, 0.55], "animal": [0.0004, 0.55],
    })


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
    # Continuous embedder (services/intelligence/embed/daemon.py): the controller tick embeds a bounded slice
    # of whatever is still unembedded, so find-similar coverage tracks new data instead of drifting to zero
    # until someone remembers a backfill. Bounded per tick and VRAM-gated so it always yields the GPU to
    # autolabel and training (the concurrency that starved the sweep is exactly what min_free_mb prevents).
    daemon_enabled: bool = True
    daemon_max_frames_per_tick: int = 400
    daemon_max_objects_per_tick: int = 4000
    daemon_min_free_mb: int = 4000   # skip this tick if free VRAM is below this; embed peak is ~3 GB


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
    # Inference resolution for the plate detector, capped. Ultralytics defaults to 640, which on a 1920x1080
    # dashcam frame is a 3x downsample: an Indian registration plate around 150x40 pixels arrives as 50x13
    # and is invisible. Measured on a frame with three legible plates: 0 detections at 640, 960 and 1280,
    # and 2 at 1920. Every model size in the family behaved identically, so this was a preprocessing
    # setting, not a capacity problem. Lower it only if a GPU cannot hold the activations.
    plate_imgsz_cap: int = 1920
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
    # DPDPA gate v2: whether the export gate checks that a blur COVERED an annotated person or vehicle, as
    # opposed to only checking that the frame carries an anonymization audit row at all.
    #
    #   off       - the pre-v2 verdict, audit-row existence only.
    #   advisory  - the coverage gaps are computed and returned as warnings; `pass` is unchanged.
    #   enforcing - they are a blocker.
    #
    # Ships as "advisory" deliberately. 82.4% of frames holding an annotated person have zero faces
    # redacted, so flipping straight to enforcing would refuse most of the corpus, including exports that
    # are legitimate today. The route to enforcing is to run the re-check over the corpus
    # (POST /api/agent/reanalyze/all), watch the advisory count fall to zero, and then turn it on - which
    # makes this a measurement first and a gate second, in that order.
    coverage_gate: str = "advisory"      # off | advisory | enforcing


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
    # Task-specific base checkpoints. A segmentation or pose run must start from a checkpoint that already
    # has the right head: fine-tuning a detection checkpoint would train a mask or keypoint head from
    # scratch, which needs far more data than a corpus of human-corrected labels typically holds.
    segmentation_weights: str = "yolo11n-seg.pt"
    pose_weights: str = "yolo11n-pose.pt"
    # A crop classifier starts from a classification checkpoint: the head shape has to match the task, and
    # starting a classifier from a detection checkpoint discards the pretrained head entirely.
    classification_weights: str = "yolo11n-cls.pt"
    # Named so the 3D task can report what it would load, and refuse by name rather than by silence. There
    # is no 3D detector in this environment; see services/training/tasks/detect3d.py.
    detect3d_weights: str = "openpcdet-pointpillars.pth"
    # A crashed run resumes from its last checkpoint instead of restarting at epoch 0. The worker resets an
    # orphaned job to pending; without this that discarded every epoch already paid for.
    resume_orphaned_runs: bool = True


class CloudSettings(BaseModel):
    # Hybrid GPU: the RunPod A100 environment for heavy real-model work. Shared by the cloud/ runbook
    # scripts and the app so they agree on names/paths. The pod itself is provisioned on demand.
    volume_name: str = "labeloxav-vol"
    workspace: str = "/workspace"
    ckpt_dir: str = "/workspace/ckpts"
    gpu_pref: list[str] = Field(default_factory=lambda: ["A100 80GB", "H100 PCIe"])
    budget_cap_usd: float = 50.0
    per_job_cap_usd: float = 10.0        # M18: hard per-job ceiling; a single job estimated above this is refused
    image: str = ""  # pinned once the runbook discovers a CUDA 12.8 devel image

    # Warm-session mode (the user-facing connect/disconnect control): one pod held across a work session,
    # torn down on disconnect. Cost safety here is a hard requirement, never an optional setting. The
    # RUNPOD_API_KEY and LBX_RUNPOD_VOLUME_ID come from the environment (as in cloud/runpod_api.py).
    warm_gpu_type_id: str = "NVIDIA A100 80GB PCIe"  # RunPod gpuTypeId provisioned for the warm session
    warm_hourly_usd: float = 1.89        # A100 80GB on-demand rate; drives the cost meter and the ack gate
    warm_idle_minutes: int = 15          # auto-terminate after this long with no job running
    warm_max_session_hours: float = 4.0  # hard cap; auto-terminate after this regardless of activity


class OntologySettings(BaseModel):
    path: str = "ontology/labelox_in_v0.yaml"

    # M-Q.0 ontology pullback. The open-vocab models (Path B concept prompts, Path C VLM shortlist) are
    # prompted ONLY with the supported core plus the fallback classes, never with the ungrounded long tail,
    # so the model literally cannot invent bus_shelter / water_bottles / vintage_car when those are not in
    # its prompt set. The supported core is the IDD-anchored road actors and safety primitives we have
    # verified instances for; a class beyond it re-enters only by earning promotion_min_instances verified
    # (gate-accepted) instances. Everything else folds into object_fallback / vehicle_fallback honestly.
    supported_core: list[str] = Field(default_factory=lambda: [
        # vehicles (IDD-anchored + the India primitives we actually see)
        "sedan", "hatchback", "suv", "small_suv", "minivan", "pickup", "lcv", "truck", "tempo",
        "bus", "school_bus", "mini_bus", "tractor", "water_tanker",
        "motorcycle", "scooter", "moped", "cycle",
        "autorickshaw", "e_auto", "e_rickshaw",
        # vulnerable road users (safety, always prompted even while the corpus is small)
        "pedestrian", "child", "rider", "cyclist", "person",
        # animals
        "cattle", "dog", "buffalo", "goat",
        # the real, common road infrastructure (IDD has sign/light; pole and barrier are everywhere)
        "traffic_sign", "traffic_signal", "pole", "barrier", "road_divider", "speed_bump",
        # Advertising is grounded alongside signs rather than left out. Without a class of its own to land
        # in, every hoarding the detector saw was proposed as a traffic_sign.
        "hoarding",
    ])
    promotion_min_instances: int = 50  # verified (gate-accepted) instances a class must earn to re-enter


class PacksSettings(BaseModel):
    # The active domain pack for engine paths that are not yet per-session pack-routed (governance, curation).
    # Defaults to 'av' so every legacy path resolves to the AV pack exactly as before. See packs/ and
    # docs/PACK_INTERFACE.md.
    default_pack: str = "av"


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
    # M4.0 value-ranked selection. The weights are relative ranking coefficients (the value is a weighted sum
    # of per-signal [0,1] scores used only to sort, so they need not sum to 1); the base four total 1.0 and the
    # flicker/fn signals are added on top. The band is the most-informative confidence window.
    w_uncertainty: float = 0.40
    w_diversity: float = 0.25
    w_rarity: float = 0.20
    w_error_prone: float = 0.15
    w_flicker: float = 0.15              # temporal box jitter: a high-flicker high-confidence track is a
                                         # classic auto-label failure the scalar confidence misses, so it earns
                                         # review priority (core.accel.uncertainty.flicker_scores)
    # The reasoning layer's own conflict: evidence that disagrees with itself, which is where a human adds
    # the most and where no other term here can see. Weighted alongside uncertainty rather than below it,
    # because the checks behind it carry measured lift over the review base rate while error_prone currently
    # fires on 40% of the corpus at a near-constant score.
    w_reasoner_conflict: float = 0.35
    w_fn: float = 0.6                    # recall-recovery (false-negative) value, so a recovered miss ranks
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
    harness_reconcile_epsilon: float = 0.05  # val-pass mAP50 and prediction-plane AP50 may differ by at most this
                                             # on the same gold set + model, else the metrics are flagged divergent
    min_map_uplift: float = 0.005        # challenger must strictly beat champion mAP by at least this (no ties)
    control_sample_rate: float = 0.02    # fraction of auto-accepts mirrored to human control review
    control_precision_floor: float = 0.97  # measured auto-accept precision below this pauses auto-promotion
    drift_psi_breach: float = 0.2        # population-stability-index breach on embeddings / label dist
    ood_zscore: float = 3.0              # input this far from the corpus centroid defers regardless of confidence
    offhours_utc: list[int] = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3]  # when the controller may burst the A100
    controller_poll_s: float = 60.0      # the autonomy daemon ticks the controller this often
    controller_lock_key: int = 816       # Postgres advisory-lock id so only one controller daemon runs
    # M18 governance: the key that signs a release redaction-proof attestation so a buyer can verify PII
    # redaction ran over every frame of a release. Dev default is refused in prod by _require_prod_secrets.
    attestation_key: str = "labeloxavgovernattestationkey0123456789"
    redaction_coverage_floor: float = 1.0  # a release proof passes only when every frame passed the PII gate
    # Eval slices that must always be reported even if empty (VERDYX). Empty here means "use the active pack's
    # eval_strata.protected_slices"; set it to override per deployment. (Adds the field verdyx/run.py already
    # reads via getattr, so the pack default now flows through instead of a hardcoded fallback.)
    protected_slices: list[str] = Field(default_factory=list)


class RecallSettings(BaseModel):
    # Recall recovery: source candidate objects from channels that do not depend on the primary
    # detector's recall, so a missed object can be manufactured, gated, and labeled. Region channel is
    # the noisy one (SAM everything mode): the area band and the VLM-confidence floor are the only
    # filters between it and a flooded queue, so keep it behind enable_region_channel.
    enable_model_channels: bool = True
    enable_region_channel: bool = True
    iou_match: float = 0.45              # a proposal at or above this IoU with an existing object is not a miss
    fuse_iou: float = 0.55               # NMS IoU for collapsing cross-channel proposals onto one candidate
    max_gap_frames: int = 5              # a track gap longer than this is a likely exit/occlusion, not misses
    conf_decay: float = 0.9              # per-frame confidence decay across an interpolated gap
    region_min_area_frac: float = 0.0008  # drop specks below this fraction of frame area
    region_max_area_frac: float = 0.25    # drop whole-scene planes above this fraction of frame area
    region_min_vlm_conf: float = 0.45    # drop a region the VLM is not confident about
    vlm_crop_margin: float = 0.15        # context margin when cropping a region for the VLM
    sam_everything_weights: str = "sam2_b.pt"   # SAM2 base; this ultralytics build has no sam2.1 key
    device: str = "cuda:0"
    # Recall promotion gate (fail-closed on safety recall, mirrors the Safe-mIoU check):
    require_safety_recall: bool = True   # a challenger that cannot report safety-class recall is refused
    safety_recall_floor: float = 0.50    # a safety class below this recall blocks promotion
    safety_recall_max_drop: float = 0.05  # a safety class may not lose more than this recall vs the incumbent
    # Per-class safety-AP drop tolerance: a dict held tighter for VRU/animal than the global default, or a
    # single float for the old global behavior.
    safety_class_drop: dict = Field(default_factory=lambda: {"vru": 0.10, "animal": 0.10, "_default": 0.15})


class Phase4Settings(BaseModel):
    activelearn: ActiveLearnSettings = ActiveLearnSettings()
    lakefs: LakeFSSettings = LakeFSSettings()
    relabel: RelabelSettings = RelabelSettings()
    govern: GovernSettings = GovernSettings()
    recall: RecallSettings = RecallSettings()


class RateLimitSettings(BaseModel):
    """A budget per caller. On by default, because the alternative is what shipped: no bound at all on how
    fast anyone can pull frames of Indian roads out of the media routes.

    Off only for a test that means to hammer an endpoint, and the suite turns it off for exactly that
    reason: a rate limiter that throttles the test suite hides real failures behind 429s.
    """

    enabled: bool = True


class CorsSettings(BaseModel):
    """Which browser origins may call this API.

    Was a hardcoded ["http://localhost:3000", "http://127.0.0.1:3000"] in services/api/main.py, so a real
    deployment either had every browser call blocked or had the list patched in place on the host - which
    is a source edit that no longer matches the repo and is invisible to anyone reading it.

    The default keeps the two dev origins, so a laptop is unchanged. Set LBX_CORS__ORIGINS to a
    comma-separated list for a deployment. `*` is accepted and deliberately not the default: this API is
    credentialed, and an allow-all origin on a credentialed API is how a browser gets talked into making
    authenticated requests on someone else's behalf.
    """

    # A comma-separated string rather than a list[str]: pydantic-settings JSON-decodes complex fields
    # coming from the environment, so a list here would require LBX_CORS__ORIGINS to be valid JSON
    # (["https://a"]) - a format nobody types correctly under a shell, and one that fails at boot with a
    # JSONDecodeError rather than saying what it wanted.
    origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.origins.split(",") if o.strip()]


class AuthSettings(BaseModel):
    # Deny-by-default API auth. When enabled, every mutating /api request needs a valid signed Bearer token
    # (services/api/auth_token.py) and role floors gate destructive/governance routes. Reads under
    # data-bearing prefixes are gated too. Disable only for unit tests.
    enabled: bool = True
    # HMAC key that signs/verifies API tokens. The weak default below is refused on any non-local deployment
    # by _require_prod_secrets: a known signing key lets anyone mint an admin token.
    signing_key: str = "labeloxavauthsigningkey0123456789abcdef"
    # Dev convenience: POST /api/auth/dev-login hands the web app an admin token so a fresh browser is not
    # locked out by deny-by-default auth. Hard-gated to env == "local" in the route, so it cannot mint a
    # token on any real deployment no matter how this flag is set. Turn off to exercise the real login path.
    dev_login: bool = True
    # Token time-to-live. A leaked token is valid at most this long; the web app refreshes silently before it
    # lapses. Default 12 hours.
    token_ttl_seconds: int = 43200
    # Accept the legacy lbx1 tokens (no expiry, no revocation) during a migration window. Off by default and
    # refused on any non-local deployment by _require_prod_secrets: a legacy token cannot be expired or revoked.
    accept_legacy_tokens: bool = False

    # ---- how a person actually signs in ----
    # Local passwords. On by default: without them the only credential is an admin-minted token, which is
    # what made this undeployable. A directory-only deployment turns this off so no local password exists
    # to be phished or reused.
    password_login: bool = True
    # Whether an unauthenticated visitor may create their own account. Off by default: on a corpus holding
    # personal data, open registration is a decision an operator makes deliberately, not a default they
    # discover. When on, the account is created at self_signup_role and an admin promotes from there.
    self_signup: bool = False
    self_signup_role: str = "annotator"
    # Restrict self-signup to a set of email domains, e.g. ["example.com"]. Empty means no restriction,
    # which is only sensible on a closed network.
    self_signup_domains: list[str] = []
    # Require a second factor for these roles once they have enrolled one. Reviewers and admins can approve
    # labels and promote models, so their accounts are worth more than an annotator's.
    mfa_required_roles: list[str] = ["admin"]

    # ---- OIDC / SSO ----
    # Set issuer and client_id to enable. The client secret is optional because a public client with PKCE
    # is a valid and increasingly common configuration.
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_scopes: str = "openid email profile"
    # Where the provider sends the browser back. Must match the registration at the provider exactly.
    oidc_redirect_uri: str = "http://localhost:3000/api/auth/oidc/callback"
    # Role given to an account created on its first federated sign-in.
    oidc_default_role: str = "annotator"

    @property
    def oidc_enabled(self) -> bool:
        return bool(self.oidc_issuer and self.oidc_client_id)


class IntegrationsSettings(BaseModel):
    # A webhook URL is attacker-controlled input the server then fetches with its own network position, so an
    # unrestricted one is an SSRF primitive: it can reach cloud instance metadata or an internal object store.
    # Private, loopback, and link-local targets are refused by default. A self-hosted deployment whose
    # receiver genuinely lives on an internal network turns this on deliberately, and the test suite needs it
    # because its receivers are on localhost.
    allow_private_webhook_targets: bool = False


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
    lift_max_tilt_rad: float = 0.35         # clamp estimated 9-DOF pitch/roll (~20 deg); noisy ground -> wild normals
    lift_ground_band_m: float = 0.2         # thickness of the contact-ground band used to estimate slope tilt
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
    quality_2d3d_min_iou: float = 0.3        # a cuboid's projection must reach this 2D IoU with its box in
                                             # at least one camera; below it the 3D and 2D boxes disagree
    quality_misalign_fill: float = 0.05      # a box with fewer enclosed points than this is misaligned


class InspectorSettings(BaseModel):
    """Session Inspector: the Lichtblick escape hatch, the MCAP indexer, and session-health thresholds."""

    # M-I.0 Lichtblick escape hatch. The self-hosted Lichtblick (open-source Foxglove fork, MPL-2.0) reads
    # the session MCAP straight from the object store via a presigned remote-file deep link. The data-source
    # id and arg keys follow Lichtblick's URL scheme (ds=<id>, ds.url=<file>); kept configurable so a fork
    # change is a config edit, not a code edit.
    lichtblick_base_url: str = "http://localhost:8090"
    lichtblick_ds_id: str = "remote-file"
    presign_expiry_s: int = 3600

    # M-I.1 indexer. Version stamps every index + health row for reproducibility.
    indexer_version: str = "inspector-idx-v1"
    gap_min_factor: float = 5.0        # a silence longer than this * the topic's nominal period is a gap

    # M-I.2 health thresholds.
    rate_tolerance: float = 0.10       # measured rate within +/- this fraction of expected is a pass
    rate_warn_tolerance: float = 0.20  # beyond rate_tolerance but within this is a warn; beyond it is a fail
    dropout_factor: float = 10.0       # a gap longer than this * nominal period is a dropout (fail-worthy)
    max_cross_sensor_offset_ms: float = 500.0  # first/last-ts skew across sensors beyond this is suspect
    min_gnss_fix: int = 1              # minimum acceptable GNSS fix quality (0=no fix)

    # M-I.2 rig manifest: the topics a session should carry and their expected rates. A required topic that
    # is absent fails the session; a measured rate outside tolerance warns or fails. match is a substring of
    # the topic name (so /camera/cam_f and /imu both resolve).
    expected_topics: list = Field(default_factory=lambda: [
        {"match": "camera", "rate": 10.0, "required": True},
        {"match": "imu", "rate": 200.0, "required": True},
        {"match": "gnss", "rate": 10.0, "required": True},
        {"match": "can", "rate": 100.0, "required": False},
    ])

    # M-I.3 default panel layout (image, IMU plot, CAN plot, map, raw messages).
    default_layout: list = Field(default_factory=lambda: ["image", "imu_plot", "can_plot", "map", "raw"])

    # M-I.4 backend decode-on-seek fallback for video the browser cannot decode.
    decode_enabled: bool = True


class SanyxSettings(BaseModel):
    """SANYX ingest-QA thresholds. All config-driven so a rig with different tolerances retunes without code.
    A check returns a sub-score in [0,1]; the weighted overall is 0..100. A check may also hard-fail (force
    quarantine regardless of score), which is how a lost time-sync pulse quarantines a session outright."""

    # cross-channel time sync (the invariant SANYX exists to protect)
    pps_required: bool = False            # when True, a missing time-sync reference pulse hard-fails
    clock_slope_ppm_max: float = 200.0    # clock slope error tolerance (parts per million)
    # dropped frames
    drop_rate_warn: float = 0.005         # fraction of missing frames that degrades
    drop_rate_fail: float = 0.05          # fraction that quarantines
    max_gap_frames_fail: int = 30         # a single gap this long quarantines
    # exposure / rolling shutter (per-frame stats come from ingest.score_frame)
    exposure_ok_frac_warn: float = 0.90   # fraction of frames that must be well exposed
    exposure_ok_frac_fail: float = 0.60
    rolling_shutter_skew_max: float = 0.35  # normalized skew on high-motion frames
    # GPS integrity (u-blox NEO-F9P)
    hdop_max: float = 5.0
    min_fix_frac: float = 0.80            # fraction of samples with a usable fix
    min_rtk_frac: float = 0.0             # RTK continuity floor (0 = RTK not required)
    max_dropout_s: float = 5.0
    # IMU integrity (Xsens MTi-3)
    imu_sat_frac_max: float = 0.02        # fraction of accel/gyro samples at saturation
    imu_bias_jump_max: float = 2.0        # sudden bias step (m/s^2 or rad/s) that flags
    imu_gap_ms_max: float = 100.0
    # lens contamination
    fft_highfreq_min_ratio: float = 0.001  # min high-frequency energy fraction (below = fouled/defocused);
    #                                        calibrated so compressed web imagery clears and only a severely
    #                                        defocused or occluded lens (near-total high-freq loss) fails
    occlusion_area_frac_max: float = 0.15  # persistent dark/blurred area fraction that flags
    occlusion_persist_frac: float = 0.80   # fraction of frames a region must persist to count as contamination
    # scoring
    weights: dict = {
        "time_sync": 0.30, "dropped_frames": 0.15, "exposure": 0.10,
        "gps": 0.15, "imu": 0.15, "lens": 0.15,
    }
    pass_min: float = 85.0                # overall >= pass_min -> pass
    quarantine_below: float = 55.0        # overall < quarantine_below -> quarantine, otherwise degraded


class CalyxSettings(BaseModel):
    """CALYX calibration-drift thresholds. Drift is estimated as an SE(3) delta between what the cameras see
    (visual odometry / epipolar) and what the IMU and GNSS report; its magnitude sets the severity that gates
    or flags the session."""

    rot_deg_flag: float = 0.5      # SE(3) drift rotation that raises drift_detected
    rot_deg_block: float = 2.0     # rotation that blocks the session
    trans_m_flag: float = 0.05     # translation drift that flags
    trans_m_block: float = 0.20    # translation drift that blocks
    min_correspondences: int = 8   # below this the estimate is untrustworthy and returns ok with low confidence
    slerp_discontinuity_deg: float = 5.0  # rotation step between adjacent frames inconsistent with dynamics


class LabeloxSettings(BaseModel):
    """LabeloxAV core enhancement flags (M13). The learned three-path reconciler stays off until it passes the
    parity gate against the heuristic on a held-out set; the heuristic remains the default and live."""

    learned_reconcile: bool = False       # switch to the learned fusion head (only after parity)
    reconcile_parity_margin: float = 0.0  # the learned head must beat the heuristic by at least this to promote


class ForgyxSettings(BaseModel):
    """FORGYX edge-deployment settings (M16). The deploy signing key signs the deployment package manifest so a
    device can verify integrity and provenance; the default is a dev key and MUST be overridden via
    LBX_FORGYX__DEPLOY_SIGNING_KEY in production (enforced by _require_prod_secrets)."""

    deploy_signing_key: str = "labeloxforgyxdeploysigningkey0123456789"


class AnprSettings(BaseModel):
    """ANPR-India (SEC-M7): a security-domain plate-reading capability, off by default and additionally gated
    on the active pack declaring 'anpr' (so it can never run in the AV / DPDPA context). Reuses the plate
    detector; the OCR is a wire seam (Qwen via Ollama, or the PaddleOCR pod)."""

    enabled: bool = False
    plate_weights: str = ".scratch/models/pii/plate_yolov8.pt"
    plate_conf: float = 0.35
    device: str = "cpu"
    min_plate_area_frac: float = 0.0002   # ignore specks below this fraction of frame area
    ocr_min_conf: float = 0.50            # drop a read the OCR is not confident about
    # The local generative-VLM wire exposes no calibrated per-read score, so it reports None rather than a
    # constant that would make ocr_min_conf a no-op. Unscored reads are kept by default (the text is still
    # evidence, flagged unscored downstream); a deployment that must gate on a real score sets this true.
    require_measured_ocr_conf: bool = False
    ocr_backend: str = "qwen"             # qwen (Ollama) | pod (PaddleOCR Indic)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LBX_",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=str(DEFAULT_CONFIG),
        # Read the repo .env directly (absolute, so cwd does not matter). settings_customise_sources already
        # lists dotenv_settings, but that source is inert without env_file; wiring it here makes .env actually
        # "consumed by the app" as the file claims, so LBX_-prefixed entries (e.g. LBX_GROQ__API_KEY) work
        # without the process having to source .env first. Unprefixed compose vars are ignored (extra=ignore).
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
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
    reasoner: ReasonerSettings = ReasonerSettings()
    quality: QualitySettings = QualitySettings()
    intelligence: IntelligenceSettings = IntelligenceSettings()
    intel: IntelSettings = IntelSettings()  # Data Intelligence Layer (Phase 1)
    rig: RigSettings = RigSettings()        # Phase 3 camera rig layout + nominal intrinsics
    spatial: SpatialSettings = SpatialSettings()  # Phase 3 calibration + map thresholds
    pii: PiiSettings = PiiSettings()
    m9: M9Settings = M9Settings()
    training: TrainingSettings = TrainingSettings()
    cloud: CloudSettings = CloudSettings()
    ontology: OntologySettings = OntologySettings()
    packs: PacksSettings = PacksSettings()      # active domain pack for non-session-routed engine paths
    paths: PathsSettings = PathsSettings()
    phase4: Phase4Settings = Phase4Settings()  # Phase 4 closed loop + governance
    auth: AuthSettings = AuthSettings()        # deny-by-default API auth
    cors: CorsSettings = CorsSettings()        # browser origins allowed to call this API
    ratelimit: RateLimitSettings = RateLimitSettings()   # per-caller budgets on the API
    integrations: IntegrationsSettings = IntegrationsSettings()   # outbound webhook safety
    lidar: LidarSettings = LidarSettings()     # 3D LiDAR module (ingestion, clean, viewer)
    inspector: InspectorSettings = InspectorSettings()  # Session Inspector (MCAP viewer + health)
    labelox: LabeloxSettings = LabeloxSettings()
    sanyx: SanyxSettings = SanyxSettings()     # SANYX ingest QA thresholds (data engine plane)
    calyx: CalyxSettings = CalyxSettings()     # CALYX calibration-drift thresholds (data engine plane)
    forgyx: ForgyxSettings = ForgyxSettings()  # FORGYX edge-deployment signing (data engine plane)
    anpr: AnprSettings = AnprSettings()        # ANPR-India (security domain; pack-gated on 'anpr')
    groq: GroqSettings = GroqSettings()        # optional Groq cloud inference (text + vision), ollama fallback
    anthropic: AnthropicSettings = AnthropicSettings()  # optional frontier judge (text + vision), same fallback
    billing: BillingSettings = BillingSettings()  # metered delivery pricing (0 by default; metering still records)

    @model_validator(mode="after")
    def _groq_key_from_env(self):
        """Accept the conventional GROQ_API_KEY as well as LBX_GROQ__API_KEY, so setting the standard
        environment variable just works. The explicit LBX form still wins if both are set."""
        if not self.groq.api_key:
            self.groq.api_key = os.environ.get("GROQ_API_KEY", "")
        return self

    @model_validator(mode="after")
    def _anthropic_key_from_env(self):
        """Same courtesy for ANTHROPIC_API_KEY. Separate validator rather than folded into the Groq one so
        that adding a third provider does not mean editing a function named after the first."""
        if not self.anthropic.api_key:
            self.anthropic.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return self

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

    _LOCAL_HOSTS = {"localhost", "127.0.0.1", "postgres", "minio", "redis", "::1"}

    @model_validator(mode="after")
    def _require_prod_secrets(self):
        """Fail closed on default credentials. The known dev-default secrets (postgres/minio creds, lakeFS key,
        and the FORGYX/GOVERN HMAC signing keys) are refused whenever the deployment is NOT an explicit local
        dev box: either env is outside the dev set, OR a non-local service host is configured (the 'forgot
        LBX_ENV in prod' case). A purely-local dev box with dev creds is still allowed. This closes the
        fail-open default where forgetting LBX_ENV shipped forgeable signing keys and public passwords."""
        is_dev_env = self.env.lower() in self._DEV_ENVS
        non_local = not (is_dev_env and self.postgres.host in self._LOCAL_HOSTS)

        # Auth policy on a real deployment: authentication must be ON, and legacy tokens (which cannot expire or
        # be revoked) must be OFF. A DPDPA-scoped corpus with auth off is an open door; leaning on the redaction
        # plane for access control is the wrong control.
        if non_local and not self.auth.enabled:
            raise ValueError(
                f"auth is disabled on a non-local deployment (env={self.env!r}); set LBX_AUTH__ENABLED=true")
        if non_local and self.auth.accept_legacy_tokens:
            raise ValueError(
                f"legacy tokens accepted on a non-local deployment (env={self.env!r}); "
                "set LBX_AUTH__ACCEPT_LEGACY_TOKENS=false (legacy tokens cannot expire or be revoked)")

        weak = []
        if self.postgres.password == "labelox":
            weak.append("LBX_POSTGRES__PASSWORD")
        if self.minio.secret_key == "labelox123":
            weak.append("LBX_MINIO__SECRET_KEY")
        if getattr(self.minio, "access_key", None) == "labelox":
            weak.append("LBX_MINIO__ACCESS_KEY")
        if self.phase4.lakefs.secret_key.startswith("labeloxavlakefssecret"):
            weak.append("LBX_PHASE4__LAKEFS__SECRET_KEY")
        if self.forgyx.deploy_signing_key.startswith("labeloxforgyxdeploysigningkey"):
            weak.append("LBX_FORGYX__DEPLOY_SIGNING_KEY")
        if self.phase4.govern.attestation_key.startswith("labeloxavgovernattestationkey"):
            weak.append("LBX_PHASE4__GOVERN__ATTESTATION_KEY")
        if self.auth.enabled and self.auth.signing_key.startswith("labeloxavauthsigningkey"):
            weak.append("LBX_AUTH__SIGNING_KEY")
        if not weak:
            return self
        is_dev_env = self.env.lower() in self._DEV_ENVS
        local_only = self.postgres.host in self._LOCAL_HOSTS
        if not is_dev_env or not local_only:
            raise ValueError(
                f"default/weak credentials detected (env={self.env!r}, postgres host={self.postgres.host!r}); "
                f"set real values for: {', '.join(weak)}"
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
