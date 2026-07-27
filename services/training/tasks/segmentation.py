"""SegmentationTask: train an instance-segmentation model on the masks the corpus already carries.

Registering this closes the largest capability asymmetry the audit found. The engine labels masks (SAM
assisted, human corrected), exports them to COCO, and evaluates them now, but the only trainable task type
was detection, so a mask could never improve a model. The executor (services/training/jobs.run_job) is task
agnostic, so the whole spine (build, train, evaluate, gate, register, promote) applies unchanged.

Gating uses mask mAP, not box mAP. A segmentation model whose boxes improve while its masks degrade has
regressed at the thing it was trained for, and gating on the box number would promote it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from core.config import get_settings
from core.logging import get_logger
from services.training.segmentation_dataset import SegBuildSpec, build_segmentation_dataset
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_segmentation")

_SPEC_FIELDS = {f.name for f in dataclasses.fields(SegBuildSpec)}


class SegmentationTask:
    task_type = "segmentation"

    def default_base_weights(self) -> str:
        """A -seg checkpoint. Fine-tuning a detection checkpoint would train a mask head from scratch, which
        needs far more data than a corpus of human-corrected masks typically holds."""
        return get_settings().training.segmentation_weights

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        spec = {k: v for k, v in (cfg.get("dataset_spec") or {}).items() if k in _SPEC_FIELDS}
        spec["name"] = cfg["name"]
        return await build_segmentation_dataset(SegBuildSpec(**spec))

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        from ultralytics import YOLO

        settings = get_settings()
        name = hparams["name"]
        epochs = int(hparams.get("epochs", settings.training.default_epochs))
        imgsz = int(hparams.get("imgsz", settings.training.default_imgsz))
        batch = int(hparams.get("batch", settings.training.default_batch))
        project = str(settings.scratch_path() / "training" / "runs")
        should_stop = hparams.get("_should_stop")
        resume = bool(hparams.get("resume"))

        model = YOLO(base_weights)

        def _on_epoch(trainer):
            try:
                ep = int(getattr(trainer, "epoch", 0)) + 1
                tot = int(getattr(trainer, "epochs", epochs))
                m = getattr(trainer, "metrics", {}) or {}
                # Report the mask metric as the headline: this is a segmentation run, and the box number
                # would let a mask regression pass unnoticed.
                progress({"stage": "train", "epoch": ep, "total_epochs": tot, "metrics": {
                    "map50_mask": round(float(m.get("metrics/mAP50(M)", 0.0)), 4),
                    "map50_box": round(float(m.get("metrics/mAP50(B)", 0.0)), 4),
                }})
                if should_stop and should_stop():
                    trainer.stop = True
            except Exception:  # noqa: BLE001  progress must never break training
                pass

        model.add_callback("on_fit_epoch_end", _on_epoch)
        model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz, device=settings.gpu.device,
            project=project, name=name, exist_ok=True, verbose=False, plots=False,
            batch=batch, patience=max(5, epochs // 3), workers=8, seed=7, resume=resume,
        )
        return str(Path(project) / name / "weights" / "best.pt")

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict:
        from ultralytics import YOLO

        res = YOLO(weights).val(data=data_yaml, imgsz=imgsz, verbose=False)
        seg = getattr(res, "seg", None)
        box = getattr(res, "box", None)
        out = {
            "map50_mask": float(getattr(seg, "map50", 0.0)) if seg else None,
            "map_mask": float(getattr(seg, "map", 0.0)) if seg else None,
            "map50_box": float(getattr(box, "map50", 0.0)) if box else None,
            "map_box": float(getattr(box, "map", 0.0)) if box else None,
        }
        # The gate compares on "map50"; point it at the mask number so a segmentation model is judged on its
        # masks. Keeping both under their precise names means nothing is hidden.
        out["map50"] = out["map50_mask"]
        out["map"] = out["map_mask"]
        if seg is not None and getattr(seg, "ap50", None) is not None:
            names = getattr(res, "names", {}) or {}
            out["per_class"] = {names.get(i, str(i)): round(float(v), 4)
                                for i, v in enumerate(seg.ap50)}
        return out

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        """Promote only on a mask improvement, with no per-class mask collapse.

        Reuses the detection regression gate's shape so governance reads one verdict format, but the metric
        it reads is the mask one (evaluate above maps map50 to the mask number).
        """
        from services.training import eval as eval_mod

        crit = {k: v for k, v in (criteria or {}).items()
                if k in ("min_map_delta", "max_class_drop")}
        return eval_mod.regression_gate(candidate, baseline, **crit)


register(SegmentationTask())
