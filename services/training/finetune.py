"""Fine-tune runner + the close-the-loop orchestration: build dataset -> eval baseline -> fine-tune
-> eval candidate -> regression gate -> record a model_run (with provenance) -> optionally promote.

    python -m services.training.finetune --name loop-v1 --route-prefix 202606 --epochs 20
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import click

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.training.dataset_builder import BuildSpec, build_training_dataset
from services.training.eval import evaluate, regression_gate

log = get_logger("finetune")


def fine_tune(data_yaml: str, base_weights: str, epochs: int, imgsz: int, name: str, batch: int = 12) -> str:
    from ultralytics import YOLO

    settings = get_settings()
    project = str(settings.scratch_path() / "training" / "runs")
    model = YOLO(base_weights)
    model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, device=settings.gpu.device,
        project=project, name=name, exist_ok=True, verbose=False, plots=False,
        batch=batch, patience=max(5, epochs // 3), workers=8, seed=7,
    )
    best = Path(project) / name / "weights" / "best.pt"
    return str(best)


def _run_id(name: str, ds: dict, epochs: int) -> str:
    h = hashlib.sha256(f"{name}:{ds['n_train_images']}:{ds['classes']}:{epochs}".encode()).hexdigest()[:10]
    return f"mr-{name}-{h}"


async def run_loop(
    spec: BuildSpec, base_weights: str, epochs: int, imgsz: int, promote: bool,
    data_yaml: str | None = None, batch: int = 12
) -> dict:
    import yaml as _yaml

    from db.models import ModelRun

    onto = get_ontology()
    store = get_object_store()
    store.ensure_bucket()

    if data_yaml:
        # Train directly on a prepared YOLO dataset (e.g. the IDD anchor): real ground truth, real
        # val, no corpus build. The honest-accuracy benchmark.
        droot = Path(data_yaml).parent
        meta = _yaml.safe_load(Path(data_yaml).read_text())
        ds = {
            "name": spec.name, "dir": str(droot), "data_yaml": data_yaml,
            "classes": int(meta.get("nc", 0)),
            "n_train_images": len(list((droot / "images/train").glob("*"))),
            "n_val_images": len(list((droot / "images/val").glob("*"))),
            "n_train_objects": 0, "n_val_objects": 0, "gold_frames": 0,
            "ontology_version": onto.version,
        }
    else:
        ds = await build_training_dataset(spec)
    if ds["n_train_images"] < 4 or ds["n_val_images"] < 1:
        raise RuntimeError(f"not enough data to train: {ds['n_train_images']} train / {ds['n_val_images']} val images")

    log.info("loop.baseline_eval", weights=base_weights)
    baseline = evaluate(base_weights, ds["data_yaml"], imgsz=imgsz)

    log.info("loop.finetune", epochs=epochs, batch=batch)
    weights = fine_tune(ds["data_yaml"], base_weights, epochs, imgsz, spec.name, batch=batch)

    log.info("loop.candidate_eval", weights=weights)
    candidate = evaluate(weights, ds["data_yaml"], imgsz=imgsz)
    gate = regression_gate(candidate, baseline)

    weights_uri = store.put_file(f"models/{spec.name}/best.pt", weights, "application/octet-stream")

    run_id = _run_id(spec.name, ds, epochs)
    do_promote = promote and gate["promote"]
    async with get_sessionmaker()() as db:
        await db.merge(ModelRun(
            run_id=run_id, base_weights=base_weights, weights_uri=weights_uri, dataset_name=spec.name,
            n_train=ds["n_train_images"], n_val=ds["n_val_images"], epochs=epochs,
            metrics=candidate, baseline_metrics=baseline, gate=gate, promoted=do_promote,
            ontology_version=onto.version,
            notes=f"agreement_only={spec.agreement_only} max_per_class={spec.max_per_class} idd={bool(spec.idd_dir)}",
        ))
        await db.commit()

    summary = {
        "run_id": run_id, "dataset": ds, "baseline_map50": baseline["map50"],
        "candidate_map50": candidate["map50"], "map50_delta": gate["map50_delta"],
        "promote": gate["promote"], "promoted": do_promote, "reasons": gate["reasons"],
        "weights_uri": weights_uri,
    }
    log.info("loop.done", run_id=run_id, baseline=baseline["map50"], candidate=candidate["map50"],
             delta=gate["map50_delta"], promote=gate["promote"])
    if do_promote:
        log.info("loop.promote_hint", hint=f"export LBX_MODELS__YOLO__WEIGHTS={weights}")
    return summary


@click.command()
@click.option("--name", default="loop-v1")
@click.option("--base", "base_weights", default=None, help="base weights (default: configured Path A)")
@click.option("--epochs", type=int, default=20)
@click.option("--imgsz", type=int, default=960)
@click.option("--batch", type=int, default=12)
@click.option("--route-prefix", default=None, help="scope corpus to a capture batch, e.g. 202606")
@click.option("--agreement-only", is_flag=True, default=False)
@click.option("--max-per-class", type=int, default=400)
@click.option("--conf-floor", type=float, default=0.2)
@click.option("--idd-dir", default=None, help="external IDD YOLO dataset to merge (cold-start anchor)")
@click.option("--data-yaml", default=None, help="train directly on this YOLO data.yaml, skip corpus build")
@click.option("--drop", "drop_classes", multiple=True, help="ontology class to exclude (repeatable)")
@click.option("--promote/--no-promote", default=False)
def main(name, base_weights, epochs, imgsz, batch, route_prefix, agreement_only, max_per_class, conf_floor, idd_dir, data_yaml, drop_classes, promote) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    base = base_weights or settings.models.yolo.weights
    spec = BuildSpec(
        name=name, conf_floor=conf_floor, max_per_class=max_per_class, agreement_only=agreement_only,
        route_prefix=route_prefix, idd_dir=idd_dir, drop_classes=list(drop_classes),
    )
    summary = asyncio.run(run_loop(spec, base, epochs, imgsz, promote, data_yaml=data_yaml, batch=batch))
    click.echo(summary)


if __name__ == "__main__":
    main()
