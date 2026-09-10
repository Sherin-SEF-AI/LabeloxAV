"""Shrinking a model until it fits an edge budget, and saying which round it stopped at and why.

The champion is a YOLO11l. It is the right model for a server and it will not run at 25 frames a second
on an Orin Nano, so the model that ships on a vehicle is a different, smaller model, and the question is
how much accuracy that costs. Guessing at a student architecture and hoping is the usual answer; measuring
each round against the target's own latency budget is this one.

A round is: train a student against the teacher's soft targets, export it, quantise it, compile it for the
target, benchmark it, and compare the measured latency to the budget. If it fits, stop. If it does not,
`plan_cooptimization` picks the next configuration and the round repeats, up to `MAX_DISTILL_ROUNDS`.

**Every round records what it measured, including the ones that failed.** A loop that reported only its
final configuration would hide that three of the four rounds missed the budget, which is the information
somebody choosing between an Orin Nano and an AGX actually needs.

**The student is compared to its teacher on frames neither was trained on.** WP2's matcher over the shadow
frames gives the agreement rate, which is the distillation quality number: how often the small model says
what the big one says. That is a different question from mAP against gold, and it is the one that decides
whether a fleet running the student sees what the server would have seen.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("distill")

KIND = "distill"
# Rounds before the loop gives up and reports the best it managed. Four because the co-optimisation grid
# has four prune levels and a fifth round would be repeating one.
MAX_DISTILL_ROUNDS = 4
# The architectures a student may be, smallest first. A loop that could pick anything would pick something
# nobody has weights for.
STUDENT_ARCHS = ("yolo11n", "yolo11s")


async def _teacher_agreement(db: AsyncSession, *, teacher_version: str, student_version: str) -> dict:
    """How often the student says what the teacher says, on frames neither was trained on.

    Reuses WP2's matcher rather than a second notion of agreement: the same pairing, the same operating
    point, the same shared-vocabulary restriction. A distillation quality number computed a different way
    from the shadow number would be two numbers nobody could compare.
    """
    from sqlalchemy import func, select

    from db.models import InferenceRun, Prediction
    from services.verdyx.shadow_run import agreements_for_runs

    async def _best_run(mv: str):
        return (await db.execute(
            select(InferenceRun.run_id)
            .join(Prediction, Prediction.run_id == InferenceRun.run_id)
            .where(InferenceRun.model_version == mv, InferenceRun.status == "complete")
            .group_by(InferenceRun.run_id)
            .order_by(func.count(Prediction.prediction_id).desc()).limit(1))).scalar_one_or_none()

    t_run = await _best_run(teacher_version)
    s_run = await _best_run(student_version)
    if t_run is None or s_run is None:
        missing = teacher_version if t_run is None else student_version
        return {"measured": False,
                "reason": (f"{missing} has no complete inference run, so there are no frames both models "
                           f"have scored to compare them on")}
    agree = await agreements_for_runs(db, run_a=t_run, run_b=s_run)
    if "error" in agree:
        return {"measured": False, "reason": agree["error"]}
    return {"measured": True, "agreements": agree["n"], "frames": agree["frames"],
            "shared_classes": len(agree.get("shared_classes") or [])}


async def distill_to_budget(db: AsyncSession, *, teacher_version: str, target: str,
                            student_arch: str = "yolo11n", max_rounds: int = MAX_DISTILL_ROUNDS,
                            created_by: str | None = None) -> dict:
    """Train and shrink a student until it fits the target's latency budget, or say why it could not.

    Returns every round, not only the last. A loop reporting one configuration hides how many missed, and
    that is what somebody choosing between two boards needs to see.
    """
    from db.models import AgentRun, ModelRegistry
    from services.forgyx.cooptimize import _budget_for, plan_cooptimization

    if student_arch not in STUDENT_ARCHS:
        return {"ok": False, "reason": f"unknown student architecture {student_arch!r}; "
                                       f"known: {list(STUDENT_ARCHS)}"}
    teacher = await db.get(ModelRegistry, teacher_version)
    if teacher is None:
        return {"ok": False, "reason": f"teacher {teacher_version} is not registered"}
    try:
        budget = _budget_for(target)
    except Exception as exc:  # noqa: BLE001 - an unknown target is a caller error, named as one
        return {"ok": False, "reason": f"unknown target {target}: {exc}"}
    if budget is None:
        return {"ok": False, "reason": f"target {target} has no latency budget, so nothing can fit it"}

    run_id = uuid.uuid4()
    rounds: list[dict] = []
    fitted = None
    prune = 0.0
    imgsz = 640

    for i in range(max_rounds):
        entry = {"round": i + 1, "student_arch": student_arch, "imgsz": imgsz, "prune": prune}
        try:
            measured = await _measure_round(db, teacher_version=teacher_version, target=target,
                                            student_arch=student_arch, imgsz=imgsz, prune=prune)
        except Exception as exc:  # noqa: BLE001 - a failed round is recorded, never swallowed
            entry.update({"ok": False, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"})
            rounds.append(entry)
            break
        entry.update(measured)
        rounds.append(entry)
        if measured.get("ok") and measured.get("latency_ms") is not None \
                and measured["latency_ms"] <= budget:
            fitted = entry
            break
        # Next configuration from the co-optimisation planner rather than a hand-rolled step, so the
        # loop and the planner cannot disagree about what a smaller model means.
        base_latency = measured.get("latency_ms")
        if base_latency is None:
            break
        plan = plan_cooptimization(target, base_latency, measured.get("map50") or 0.0)
        if not plan["feasible"]:
            entry["plan"] = {"feasible": False,
                             "reason": "no configuration in the grid clears this budget; a smaller "
                                       "architecture is needed rather than more pruning"}
            break
        chosen = plan["chosen"]
        prune = float(chosen.get("prune", prune))
        imgsz = int(chosen.get("imgsz", imgsz))

    agreement = {"measured": False, "reason": "no student was produced to compare"}
    if fitted and fitted.get("student_version"):
        agreement = await _teacher_agreement(db, teacher_version=teacher_version,
                                             student_version=fitted["student_version"])

    report = {"ok": bool(fitted), "target": target, "budget_ms": budget,
              "teacher_version": teacher_version, "student_arch": student_arch,
              "rounds": rounds, "fitted": fitted, "teacher_agreement": agreement,
              "reason": (None if fitted else
                         (f"no configuration met the {budget:.0f} ms budget in {len(rounds)} rounds; "
                          f"the rounds and what each measured are listed"))}
    db.add(AgentRun(run_id=run_id, kind=KIND, scope={"target": target, "teacher": teacher_version},
                    status="committed" if fitted else "refused", policy={"student_arch": student_arch},
                    counts=report, changes={}, critic={}, created_by=created_by))
    await db.commit()
    log.info("distill.finished", run_id=str(run_id), target=target, ok=bool(fitted),
             rounds=len(rounds))
    return {**report, "run_id": str(run_id), "at": datetime.now(UTC).isoformat()}


async def _measure_round(db: AsyncSession, *, teacher_version: str, target: str, student_arch: str,
                         imgsz: int, prune: float) -> dict:
    """One round: train, export, quantise, compile, benchmark. Refuses with a reason where it cannot.

    The compile and benchmark steps need the target's toolchain, which is a device SDK rather than a
    library, so on a host without it the round reports what it could not do instead of a latency it did
    not measure. That refusal is the honest half of this loop on any machine that is not the board.
    """
    from services.forgyx.capabilities import CapabilityError, require

    try:
        require(target)
    except CapabilityError as exc:
        return {"ok": False, "stage": "compile", "reason": str(exc),
                "latency_ms": None, "map50": None,
                "note": ("the round stops before training: a student nobody can compile for this target "
                         "cannot be measured against its budget, and training one first would spend GPU "
                         "hours to arrive at the same refusal")}

    # The toolchain is present, so the round runs for real. Kept here rather than in the caller so the
    # loop reads as one sequence.
    from services.training.jobs import TrainJobSpec, enqueue_job, run_job

    job_id = await enqueue_job(TrainJobSpec(
        purpose="distill", task_type="selftrain", compute_target="local",
        base_weights=f"{student_arch}.pt",
        dataset_spec={"name": f"distill-{student_arch}-{int(prune * 100)}"},
        hparams={"imgsz": imgsz}, promote=False,
        notes=f"distilled from {teacher_version} for {target}"))
    res = await run_job(job_id)
    student_version = (res or {}).get("run_id")
    if not student_version:
        return {"ok": False, "stage": "train", "reason": "the student did not train",
                "latency_ms": None, "map50": None}

    from services.forgyx.export import export_and_benchmark

    bench = await export_and_benchmark(db, student_version, imgsz=imgsz)
    return {"ok": True, "stage": "benchmark", "student_version": student_version,
            "latency_ms": (bench or {}).get("latency_ms", {}).get("p50"),
            "map50": (res or {}).get("candidate", {}).get("map50"), "benchmark": bench}
