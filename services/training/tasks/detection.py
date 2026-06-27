"""DetectionTask: wraps the existing YOLO fine-tune spine (dataset_builder + finetune + eval) so the
generalized executor can drive it. Specialized detectors and per-domain models are just dataset_spec
filters (include_classes / cities / route_prefix). Per-epoch progress comes from an Ultralytics
callback.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml as _yaml

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.ontology import get_ontology
from services.training import eval as eval_mod
from services.training.dataset_builder import BuildSpec, build_training_dataset
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_detection")

_BUILDSPEC_FIELDS = {f.name for f in dataclasses.fields(BuildSpec)}


class DetectionTask:
    task_type = "detection"

    def default_base_weights(self) -> str:
        return get_settings().models.yolo.weights

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        name = cfg["name"]
        data_yaml = (cfg.get("dataset_spec") or {}).get("data_yaml")
        if data_yaml:
            # External prepared YOLO dataset (e.g. the IDD anchor): real GT, no corpus build.
            droot = Path(data_yaml).parent
            meta = _yaml.safe_load(Path(data_yaml).read_text())
            return {
                "name": name, "dir": str(droot), "data_yaml": data_yaml,
                "classes": int(meta.get("nc", 0)),
                "n_train_images": len(list((droot / "images/train").glob("*"))),
                "n_val_images": len(list((droot / "images/val").glob("*"))),
                "n_train_objects": 0, "n_val_objects": 0, "gold_frames": 0,
                "ontology_version": get_ontology().version,
            }
        ds_spec = {k: v for k, v in (cfg.get("dataset_spec") or {}).items() if k in _BUILDSPEC_FIELDS}
        ds_spec["name"] = name  # the build dir + run name is the unique job name
        return await build_training_dataset(BuildSpec(**ds_spec))

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        from ultralytics import YOLO

        settings = get_settings()
        name = hparams["name"]
        epochs = int(hparams.get("epochs", settings.training.default_epochs))
        imgsz = int(hparams.get("imgsz", settings.training.default_imgsz))
        batch = int(hparams.get("batch", settings.training.default_batch))
        project = str(settings.scratch_path() / "training" / "runs")

        should_stop = hparams.get("_should_stop")

        model = YOLO(base_weights)

        def _on_epoch(trainer):
            try:
                ep = int(getattr(trainer, "epoch", 0)) + 1
                tot = int(getattr(trainer, "epochs", epochs))
                m = getattr(trainer, "metrics", {}) or {}
                map50 = float(m.get("metrics/mAP50(B)", 0.0))
                progress({"stage": "train", "epoch": ep, "total_epochs": tot, "metrics": {"map50": round(map50, 4)}})
                if should_stop and should_stop():  # best-effort cancel at the epoch boundary
                    trainer.stop = True
            except Exception:  # noqa: BLE001  progress must never break training
                pass

        model.add_callback("on_fit_epoch_end", _on_epoch)
        model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz, device=settings.gpu.device,
            project=project, name=name, exist_ok=True, verbose=False, plots=False,
            batch=batch, patience=max(5, epochs // 3), workers=8, seed=7,
        )
        return str(Path(project) / name / "weights" / "best.pt")

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict:
        metrics = eval_mod.evaluate(weights, data_yaml, imgsz=imgsz)
        try:
            sm = eval_mod.safe_miou_report(weights, data_yaml, imgsz=imgsz)
            metrics["safe_miou"] = sm.get("safe_miou")
            metrics["safe_miou_report"] = sm
        except Exception as exc:  # noqa: BLE001
            log.warning("detection.safe_miou_failed", error=str(exc))
        return metrics

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        crit = {k: v for k, v in (criteria or {}).items()
                if k in ("min_map_delta", "max_class_drop", "min_safe_miou")}
        return eval_mod.regression_gate(candidate, baseline, **crit)


register(DetectionTask())
