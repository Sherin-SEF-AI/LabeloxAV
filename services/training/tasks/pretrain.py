"""Continued self-supervised pretraining of the visual backbone on Indian road frames.

Every model this engine trains starts from weights that have never seen an Indian road. The detector
starts from COCO, and the embedding backbone that drives duplicate detection, similarity, clustering and
novelty is `vit_base_patch16_dinov3.lvd1689m`, pretrained on LVD-1689M. Those are good weights and they
are not weights for this domain: an autorickshaw, a hoarding, a metro pillar, and traffic at Indian
densities are all out of distribution for them, and everything downstream inherits that.

The fix is continued pretraining, not fine-tuning: the DINO/iBOT objective needs no labels, so it can
learn from all 41,204 real frames rather than the labelled fraction. It also needs many GPU-hours on a
large card, which is why this task builds and dispatches rather than trains. Local execution would hold
the only GPU in this deployment for days, and the plan for this program says so explicitly.

Three things this task refuses rather than guesses:

* **Synthetic frames.** A composite is real pixels rearranged, and a backbone that learns the seams of
  the copy-paste generator would then find those seams "similar" for the rest of its life. The manifest
  takes `Frame.origin == REAL` like every other deny-list reader (`core/origin.py`).
* **Duplicates.** Consecutive dashcam frames are near-identical, and a self-supervised objective that
  sees the same picture a thousand times learns that picture. The manifest keeps one frame per
  `dup_group_id`.
* **A job it cannot pay for.** The cost is estimated from the step budget and the pod's hourly rate
  before anything is published, and a job over `per_job_cap_usd` is refused with the number rather than
  dispatched and discovered later.
"""

from __future__ import annotations

import json

from sqlalchemy import String

from core.config import get_settings
from core.logging import get_logger
from core.origin import REAL
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_pretrain")

# The frame budget for one pretraining stage. Continued pretraining does not need every frame, it needs a
# representative sample, and an unbounded manifest is an unbounded upload.
PRETRAIN_MAX_FRAMES = 40000
# Steps rather than epochs, because the cost estimate has to exist before the pod has seen the dataset.
# The default is derived from the spend cap rather than fixed: a fixed 20,000 costs $11.81 at the ship
# default rate against a $10 per-job cap, so every unmodified dispatch would have been refused by the
# guard, and raising the cap to make the default fit would be tuning the safety rail to the number.
PRETRAIN_STEP_FLOOR = 2000
# Throughput used for the cost estimate, in optimisation steps per hour on the target card. Deliberately
# conservative: an estimate that is too low is how a job gets dispatched and then costs double.
STEPS_PER_HOUR = 3200.0
# Checkpoints are pulled back this often, so a pod that dies mid-run leaves something usable behind.
CKPT_EVERY_STEPS = 2000

BASE_MODEL = "vit_base_patch16_dinov3.lvd1689m"


def default_steps() -> int:
    """The largest step budget that fits inside the per-job spend cap, rounded down to a whole thousand.

    Derived so that raising the cap raises the default and lowering it lowers the default, rather than
    leaving a constant that silently stops fitting. Floored so a very small cap yields a refusal with a
    number rather than a budget of zero steps.
    """
    cfg = get_settings().cloud
    affordable = (float(cfg.per_job_cap_usd) / max(float(cfg.warm_hourly_usd), 1e-6)) * STEPS_PER_HOUR
    return max(PRETRAIN_STEP_FLOOR, int(affordable // 1000) * 1000)


def estimate_cost_usd(steps: int, hourly_usd: float | None = None) -> dict:
    """What the pod will cost for this step budget, and the rate it was computed from.

    Returned as a dict rather than a float so the refusal can name the rate: an estimate whose inputs are
    invisible is an estimate nobody can check against the invoice.
    """
    rate = float(hourly_usd if hourly_usd is not None else get_settings().cloud.warm_hourly_usd)
    hours = float(steps) / STEPS_PER_HOUR
    return {"steps": int(steps), "hours": round(hours, 2), "hourly_usd": rate,
            "usd": round(hours * rate, 2), "steps_per_hour": STEPS_PER_HOUR}


async def build_manifest(*, max_frames: int = PRETRAIN_MAX_FRAMES) -> dict:
    """Real, non-duplicate, selected frames and their image uris. No labels: the objective needs none."""
    from sqlalchemy import func, select

    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        # One frame per duplicate group. A frame with no group is its own group; `is_dup_canonical` names
        # the representative where the deduper has run.
        stmt = (select(Frame.frame_id, Frame.img_uri, Frame.session_id, Frame.cam_id, Frame.ts_ns)
                .join(DbSession, DbSession.session_id == Frame.session_id)
                .where(Frame.origin == REAL, DbSession.origin == REAL,
                       Frame.selected.is_(True), Frame.img_uri.isnot(None))
                .where((Frame.is_dup_canonical.is_(True)) | (Frame.dup_group_id.is_(None)))
                .order_by(func.md5(func.cast(Frame.frame_id, String)))
                .limit(max_frames))
        rows = (await db.execute(stmt)).all()
        total_real = (await db.execute(
            select(func.count()).select_from(Frame)
            .join(DbSession, DbSession.session_id == Frame.session_id)
            .where(Frame.origin == REAL, DbSession.origin == REAL,
                   Frame.selected.is_(True)))).scalar_one()

    frames = [{"frame_id": str(fid), "img_uri": uri, "session_id": str(sid), "cam_id": cam,
               "ts_ns": int(ts)} for fid, uri, sid, cam, ts in rows]
    return {"frames": frames, "n_frames": len(frames), "real_frames_total": int(total_real),
            "deduplicated": int(total_real) - len(frames) if total_real >= len(frames) else 0,
            "capped_at": max_frames}


class PretrainTask:
    """Continued DINO/iBOT pretraining of the embedding backbone. Cloud only, by design."""

    task_type = "pretrain"

    def default_base_weights(self) -> str:
        return get_settings().intel.embed.dinov3_model or BASE_MODEL

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        name = cfg["name"]
        spec = dict(cfg.get("dataset_spec") or {})
        hparams = dict(cfg.get("hparams") or {})
        max_frames = int(spec.get("max_frames") or PRETRAIN_MAX_FRAMES)
        steps = int(hparams.get("steps") or default_steps())

        manifest = await build_manifest(max_frames=max_frames)
        if manifest["n_frames"] < 1000:
            # Below this a self-supervised stage learns the sample rather than the domain, and the honest
            # answer is a refusal with the count rather than a checkpoint nobody should trust.
            raise ValueError(
                f"only {manifest['n_frames']} real deduplicated frames are available; continued "
                f"pretraining on fewer than 1000 learns the sample rather than the domain")

        cost = estimate_cost_usd(steps)
        cap = float(get_settings().cloud.per_job_cap_usd)
        if cost["usd"] > cap:
            raise ValueError(
                f"estimated ${cost['usd']} for {steps} steps at ${cost['hourly_usd']}/h exceeds the "
                f"per-job cap of ${cap}; lower `steps` or raise cloud.per_job_cap_usd")

        out = get_settings().scratch_path() / "pretrain" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(
            {"frames": manifest["frames"], "base_model": self.default_base_weights(),
             "steps": steps, "ckpt_every_steps": CKPT_EVERY_STEPS,
             "objective": "dino+ibot"}, indent=2))
        log.info("pretrain.manifest", name=name, frames=manifest["n_frames"],
                 deduplicated=manifest["deduplicated"], estimated_usd=cost["usd"])
        return {
            "name": name, "dir": str(out), "data_yaml": str(out / "manifest.json"),
            "classes": 0, "n_train_images": manifest["n_frames"], "n_val_images": 0,
            "n_train_objects": 0, "n_val_objects": 0, "gold_frames": 0,
            "real_frames_total": manifest["real_frames_total"],
            "deduplicated": manifest["deduplicated"], "cost_estimate": cost,
            "ontology_version": None,
        }

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        """Refuses locally, by design, and names the reason.

        This is not an unimplemented stub. Continued pretraining of a ViT-B is days of an A100 and this
        deployment has one 16 GB card that the detector, the autolabel plane and the depth model all
        share; running it here would stop every other loop in the program for the duration. The cloud
        path is `dispatch_cloud_job(..., entrypoint="pretrain_dino")`, which publishes the manifest this
        task built and waits for a pod.
        """
        raise RuntimeError(
            "pretrain does not run locally: continued DINO/iBOT pretraining of a ViT-B is days of an "
            "A100, and this host has one shared 16 GB card. Dispatch it with compute_target='cloud'; "
            f"the manifest is ready at {data_yaml}")

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict:
        """A pretrained backbone has no task metric, so this reports what it is instead of inventing one.

        Accuracy for these weights is measured downstream, by re-embedding the corpus under a new
        `model_versions` value and comparing retrieval and duplicate detection against the old vectors.
        Returning a fabricated mAP here would put a number in the registry that means nothing.
        """
        return {"task": "pretrain", "measured": False,
                "reason": "a self-supervised backbone has no task metric; it is evaluated downstream by "
                          "re-embedding the corpus and comparing retrieval against the previous vectors",
                "weights": weights}

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        """Never auto-promotes. A backbone swap re-embeds 41,000 frames and changes every similarity
        surface in the product, which is a person's decision."""
        return {"promote": False,
                "reasons": ["a pretrained backbone is never auto-promoted: adopting it re-embeds the "
                            "whole corpus and changes every similarity, duplicate and novelty surface"]}


register(PretrainTask())
