"""Replay a classifier stage against the answer key and say what it would have done to the labels.

A stage looks at each truth item's machine label and either keeps it (abstains) or proposes another answer: a
class, or "not an object". Scoring compares that proposal with what the person ruled, and splits the outcome
the way the decision to ship needs it split:

- **fixed**: the machine was wrong and the stage gave the person's answer.
- **broke**: the machine was right and the stage changed it. `wrong_fix_rate` is broke over right labels, and
  it is the number that decides whether a stage may act at all. It is reported with a Wilson 95% upper bound
  because the right-label populations are small (137 direct items) and a point estimate of 0% from 137
  items still allows 2.7%.
- **changed_wrong**: the machine was wrong, and the stage changed it to something else that is also wrong.
  Not harmful in accuracy terms, but it puts a second wrong answer in front of a reviewer.
- **missed**: the machine was wrong and the stage kept it.

Accuracy before and after is reported too, but on this set it is not a corpus estimate: people mostly opened
objects to fix them, so the machine is right on about 6% of direct items here, against far more in the
corpus. Read fix rate and wrong-fix rate, not accuracy.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from core.logging import get_logger
from services.labelagent.truth import NOT_OBJECT, TruthItem

log = get_logger("labelagent.benchmark")


@dataclass
class Prediction:
    label: int | None          # a class id, NOT_OBJECT, or None to keep the machine's label
    conf: float | None = None
    note: str = ""


def wilson_upper(k: int, n: int, z: float = 1.96) -> float | None:
    """Upper end of the Wilson score interval for k successes in n: an honest ceiling on a small rate."""
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + spread) / (1 + z * z / n))


def outcome(item: TruthItem, pred: Prediction) -> str:
    answer = item.pred if pred.label is None else pred.label
    changed = answer != item.pred
    if item.pred == item.truth:
        return "broke" if changed else "kept"
    if answer == item.truth:
        return "fixed"
    return "changed_wrong" if changed else "missed"


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _score_subset(items: list[TruthItem], preds: list[Prediction], name_of: Callable[[int], str],
                  family_of: Callable[[int], str]) -> dict:
    outs = [outcome(i, p) for i, p in zip(items, preds, strict=True)]
    c = Counter(outs)
    n_right = c["kept"] + c["broke"]
    n_wrong = c["fixed"] + c["missed"] + c["changed_wrong"]
    n_changed = c["broke"] + c["fixed"] + c["changed_wrong"]
    answers = [i.pred if p.label is None else p.label for i, p in zip(items, preds, strict=True)]

    not_obj = [(i, a) for i, a in zip(items, answers, strict=True) if i.truth == NOT_OBJECT]
    real = [(i, a) for i, a in zip(items, answers, strict=True) if i.truth != NOT_OBJECT and i.pred != NOT_OBJECT]
    caught = sum(1 for _i, a in not_obj if a == NOT_OBJECT)
    false_alarm = sum(1 for _i, a in real if a == NOT_OBJECT)

    per_class: dict[str, dict] = defaultdict(lambda: {"n": 0, "before": 0, "after": 0})
    per_family: dict[str, dict] = defaultdict(lambda: {"n": 0, "before": 0, "after": 0})
    for i, a in zip(items, answers, strict=True):
        for bucket in (per_class[name_of(i.truth)], per_family[family_of(i.truth)]):
            bucket["n"] += 1
            bucket["before"] += int(i.pred == i.truth)
            bucket["after"] += int(a == i.truth)

    confusions = Counter((name_of(i.truth), name_of(a)) for i, a in zip(items, answers, strict=True)
                         if a != i.truth)
    return {
        "n": len(items),
        "outcomes": dict(c),
        "accuracy_before": _rate(n_right, len(items)),
        "accuracy_after": _rate(sum(1 for i, a in zip(items, answers, strict=True) if a == i.truth), len(items)),
        "fix_rate": _rate(c["fixed"], n_wrong),
        "wrong_fix_rate": _rate(c["broke"], n_right),
        "wrong_fix_upper95": (round(u, 4) if (u := wilson_upper(c["broke"], n_right)) is not None else None),
        "n_right_labels": n_right,
        "change_precision": _rate(c["fixed"], n_changed),
        "abstain_rate": _rate(sum(1 for p in preds if p.label is None), len(preds)),
        "not_object_catch": _rate(caught, len(not_obj)),
        "not_object_false_alarm": _rate(false_alarm, len(real)),
        "per_class": {k: {**v, "acc_before": _rate(v["before"], v["n"]), "acc_after": _rate(v["after"], v["n"])}
                      for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]["n"])},
        "per_family": {k: {**v, "acc_before": _rate(v["before"], v["n"]), "acc_after": _rate(v["after"], v["n"])}
                       for k, v in sorted(per_family.items(), key=lambda kv: -kv[1]["n"])},
        "top_confusions": [{"truth": t, "answer": a, "n": n} for (t, a), n in confusions.most_common(15)],
    }


def namers(onto) -> tuple[Callable[[int], str], Callable[[int], str]]:
    def name_of(cid: int) -> str:
        if cid == NOT_OBJECT:
            return "NOT_OBJECT"
        try:
            return onto.by_id(cid).name
        except KeyError:
            return f"class_{cid}"

    def family_of(cid: int) -> str:
        if cid == NOT_OBJECT:
            return "not_object"
        try:
            return onto.by_id(cid).l1
        except KeyError:
            return "unknown"

    return name_of, family_of


def score(items: list[TruthItem], preds: list[Prediction], onto, *, split: str = "test") -> dict:
    """The full report for one stage: per tier, and the headline (direct, machine-made) on its own."""
    if len(items) != len(preds):
        raise ValueError(f"{len(preds)} predictions for {len(items)} items")
    # A stage that answers `undefined` or `background` is saying "not an object", which is how the truth
    # set records it too. Without this a correct catch would score as a confusion.
    from services.labelagent.truth import NOT_OBJECT_CLASS_NAMES

    noids = {c.id for c in onto.classes if c.name in NOT_OBJECT_CLASS_NAMES}
    preds = [Prediction(NOT_OBJECT, p.conf, p.note) if p.label in noids else p for p in preds]
    name_of, family_of = namers(onto)
    pairs = [(i, p) for i, p in zip(items, preds, strict=True) if split == "all" or i.split == split]

    def sub(pred_fn) -> dict:
        sel = [(i, p) for i, p in pairs if pred_fn(i)]
        return _score_subset([i for i, _ in sel], [p for _, p in sel], name_of, family_of)

    return {
        "split": split,
        # Every tier together. The right-label population of any single tier is too small to bound a 2%
        # wrong-fix rate (45 direct items on the test split), and pooling is what makes that bound usable.
        "pooled": sub(lambda i: i.machine_made or i.tier == "keep"),
        "headline": sub(lambda i: i.tier == "direct" and i.machine_made),
        "track": sub(lambda i: i.tier == "track" and i.machine_made),
        "keep": sub(lambda i: i.tier == "keep"),
        "imported": sub(lambda i: i.pred_source == "imported"),
    }


# ------------------------------------------------------------------------------------------------ stages
# Each stage takes the items (and whatever it needs) and returns one Prediction per item, in order.

def stage_detector(items: list[TruthItem]) -> list[Prediction]:
    """The baseline every stage is measured against: keep whatever the detector said."""
    return [Prediction(None) for _ in items]


async def stage_vlm_stored(db, items: list[TruthItem]) -> list[Prediction]:
    """The VLM judge's verdicts already in the database, on the machine's own label only.

    A verdict counts only if `given_class_id` is the item's machine prediction: a verdict given after a
    person had already fixed the label was judging the person, not the machine. `incorrect` with a proposed
    class is a change; `correct`, `unsure`, and `incorrect` with no proposal keep the label.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import MachineVerdict

    ids = [_uuid.UUID(i.object_id) for i in items]
    by_obj: dict[str, list] = defaultdict(list)
    for chunk in range(0, len(ids), 2000):
        rows = (await db.execute(
            select(MachineVerdict.object_id, MachineVerdict.verdict, MachineVerdict.proposed_class_id,
                   MachineVerdict.detail, MachineVerdict.ts_ns, MachineVerdict.model_version)
            .where(MachineVerdict.object_id.in_(ids[chunk:chunk + 2000]), MachineVerdict.judge == "vlm"))).all()
        for r in rows:
            by_obj[str(r.object_id)].append(r)

    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    def judged(detail: dict) -> int | None:
        # Older batches record the label by name (`given_class`, or `asked_class` in judge calibration);
        # only the label-check batch stores the id.
        if detail.get("given_class_id") is not None:
            return int(detail["given_class_id"])
        name = detail.get("given_class") or detail.get("asked_class")
        return onto.by_name(name).id if name and onto.has_name(name) else None

    out = []
    for i in items:
        cands = [r for r in by_obj.get(i.object_id, []) if judged(r.detail or {}) == i.pred]
        if not cands:
            out.append(Prediction(None, note="no verdict"))
            continue
        r = max(cands, key=lambda r: r.ts_ns)
        if r.verdict == "incorrect" and r.proposed_class_id is not None:
            out.append(Prediction(int(r.proposed_class_id), note=f"vlm:{r.model_version}"))
        else:
            out.append(Prediction(None, note=f"vlm:{r.verdict}"))
    return out


def with_floor(preds: list[Prediction], min_conf: float) -> list[Prediction]:
    """A confidence-floored copy of a stage's predictions: below the floor the label is kept. Lets one pass
    of an expensive stage be scored at several floors."""
    return [p if (p.label is None or (p.conf is not None and p.conf >= min_conf))
            else Prediction(None, p.conf, p.note + ":below_floor") for p in preds]


def coverage(preds: list[Prediction]) -> float:
    """Share of items a stage had an opinion on (a verdict, a crop), whatever that opinion was."""
    if not preds:
        return 0.0
    return round(sum(1 for p in preds if p.note not in ("no verdict", "no crop")) / len(preds), 4)


async def stage_reasoner(db, items: list[TruthItem]) -> list[Prediction]:
    """The Tier-1 reasoner, replayed on each item's frame as the machine left it.

    Every object of the frame is rebuilt from its current row, except the truth item itself, which is rebuilt
    from the machine's version (class, box, confidence and provenance from before any person touched it).
    Replayed one frame at a time, so the temporal check has no history and stays silent: this measures the
    single-frame checks only. `reject` with a suggested class proposes that class, `reject` without one
    proposes "not an object", and everything else keeps the label.
    """
    import uuid as _uuid
    from types import SimpleNamespace

    from sqlalchemy import select

    from db.models import Frame, Object
    from services.autolabel.ontology import get_ontology
    from services.autolabel.reasoner.pass_ import FrameContext, reason_frame
    from services.autolabel.reasoner.rerun import _to_unified
    from services.autolabel.reasoner.verdict import ADJUDICATE, REJECT

    onto = get_ontology()
    name_of, _ = namers(onto)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for k, i in enumerate(items):
        by_frame[i.frame_id].append(k)

    out: list[Prediction] = [Prediction(None, note="not reasoned") for _ in items]
    frame_ids = list(by_frame)
    for chunk in range(0, len(frame_ids), 200):
        fids = [_uuid.UUID(f) for f in frame_ids[chunk:chunk + 200]]
        frames = {str(f.frame_id): f for f in (await db.execute(
            select(Frame.frame_id, Frame.width, Frame.height, Frame.scene).where(Frame.frame_id.in_(fids)))).all()}
        objs: dict[str, list] = defaultdict(list)
        for o in (await db.execute(select(Object).where(Object.frame_id.in_(fids), Object.state != "rejected"))).scalars():
            objs[str(o.frame_id)].append(o)

        for fid in frame_ids[chunk:chunk + 200]:
            f = frames.get(fid)
            if f is None:
                continue
            truth_by_obj = {items[k].object_id: k for k in by_frame[fid]}
            rows, idx_of = [], {}
            for o in objs.get(fid, []):
                k = truth_by_obj.get(str(o.object_id))
                if k is not None and items[k].tier != "keep":
                    it = items[k]
                    if it.pred == NOT_OBJECT:
                        continue
                    o = SimpleNamespace(object_id=o.object_id, frame_id=o.frame_id, track_id=o.track_id,
                                        class_id=it.pred, bbox=it.bbox, conf=it.pred_conf or 0.0,
                                        provenance=it.provenance)
                if k is not None:
                    idx_of[len(rows)] = k
                rows.append(o)
            unified = []
            for r in rows:
                try:
                    unified.append(_to_unified(r, name_of(r.class_id)))
                except Exception:  # noqa: BLE001 - one unbuildable neighbour should not void the frame
                    unified.append(None)
            keep = [(n, u) for n, u in enumerate(unified) if u is not None]
            verdicts = reason_frame([u for _, u in keep], onto,
                                    FrameContext(width=int(f.width or 0), height=int(f.height or 0),
                                                 scene=f.scene or {}, track_history={}))
            for (n, _u), v in zip(keep, verdicts, strict=True):
                k = idx_of.get(n)
                if k is None:
                    continue
                sug = getattr(v, "suggested_class", None)
                if v.decision in (REJECT, ADJUDICATE) and sug and onto.has_name(sug):
                    out[k] = Prediction(onto.by_name(sug).id, note=f"reasoner:{v.decision}")
                elif v.decision == REJECT:
                    out[k] = Prediction(NOT_OBJECT, note="reasoner:reject")
                else:
                    out[k] = Prediction(None, note=f"reasoner:{v.decision}")
    return out


# ------------------------------------------------------------------------------------------------ crops

def crop_of(image, bbox: list[float], pad: float = 0.05, min_side: int = 8):
    """The item's box from its frame, padded slightly so the edge of the object is in view. None when the
    box is too small to say anything about."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < min_side or bh < min_side:
        return None
    x1, y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    x2, y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh))
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return image[y1:y2, x1:x2]


async def iter_crops(db, items: list[TruthItem]):
    """Yield (index, crop or None) for every item, reading each frame image once."""
    import uuid as _uuid

    import cv2
    import numpy as np
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import Frame

    store = get_object_store()
    by_frame: dict[str, list[int]] = defaultdict(list)
    for k, i in enumerate(items):
        by_frame[i.frame_id].append(k)
    frame_ids = list(by_frame)
    for chunk in range(0, len(frame_ids), 500):
        fids = [_uuid.UUID(f) for f in frame_ids[chunk:chunk + 500]]
        uris = dict((await db.execute(select(Frame.frame_id, Frame.img_uri).where(Frame.frame_id.in_(fids)))).all())
        for fid in frame_ids[chunk:chunk + 500]:
            img = None
            uri = uris.get(_uuid.UUID(fid))
            if uri:
                try:
                    img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
                except Exception as exc:  # noqa: BLE001 - a missing image is a missing crop, not a failed run
                    log.warning("labelagent.frame_unreadable", frame_id=fid, error=str(exc))
            for k in by_frame[fid]:
                yield k, (crop_of(img, items[k].bbox) if img is not None else None)


async def stage_classify_crop(db, items: list[TruthItem], *, min_conf: float = 0.0) -> list[Prediction]:
    """The current zero-shot crop classifier: SigLIP 2 over every ontology class name, top-1.

    `min_conf` is the softmax floor below which it keeps the label instead, which is how it would have to
    be used as a fixer. At 0 it is the raw classifier.
    """
    from services.autolabel.classify_crop import classify_crop

    out: list[Prediction] = [Prediction(None, note="no crop") for _ in items]
    async for k, crop in iter_crops(db, items):
        if crop is None:
            continue
        top = classify_crop(crop, topk=1)[0]
        if top["conf"] >= min_conf:
            out[k] = Prediction(int(top["class_id"]), conf=top["conf"], note="classify_crop")
        else:
            out[k] = Prediction(None, conf=top["conf"], note="classify_crop:below_floor")
    return out


# ------------------------------------------------------------------------------------------------ learned
# Both are cross-fitted, so every item is predicted by something that never saw its session: memory leaves
# the item's own session out of the neighbourhood, and the probe is fitted on one side of the split and
# predicts the other, then the sides swap. That is what lets them be scored on the whole set like the
# untrained stages, rather than on the 45 right labels of the test split alone. `conf` is the margin over
# the runner-up, so a split neighbourhood or a flat posterior reads as unsure.

async def stage_memory(db, items: list[TruthItem], *, k: int = 15) -> list[Prediction]:
    from services.labelagent.memory import Bank, load_vectors

    vectors = await load_vectors(db, [i.object_id for i in items])
    bank = Bank.build(items, vectors)
    out = []
    for i in items:
        q = vectors.siglip.get(i.object_id)
        if q is None:
            out.append(Prediction(None, note="no crop"))
            continue
        v = bank.vote(q, k=k, exclude_session=i.session_id, exclude_object=i.object_id,
                      near=(i.vehicle_id, i.ts_ns))
        out.append(Prediction(v.label, conf=v.margin, note="memory") if v.label is not None
                   else Prediction(None, note="memory:empty"))
    return out


async def stage_probe(db, items: list[TruthItem]) -> list[Prediction]:
    from services.labelagent.memory import NEAR_NS, Probe, load_vectors

    vectors = await load_vectors(db, [i.object_id for i in items])
    out: list[Prediction] = [Prediction(None, note="no crop") for _ in items]
    for fold in ("test", "calib"):
        # The split is by 30-minute block, so an object can sit seconds across a block edge from itself.
        # Training items within the buffer of any item being predicted are dropped.
        held = defaultdict(list)
        for i in items:
            if i.split == fold:
                held[i.vehicle_id].append(i.ts_ns)
        for v in held:
            held[v].sort()

        def near_fold(i, buf=NEAR_NS // 3, held=held) -> bool:
            import bisect
            ts = held.get(i.vehicle_id)
            if not ts:
                return False
            j = bisect.bisect_left(ts, i.ts_ns)
            return any(0 <= n < len(ts) and abs(ts[n] - i.ts_ns) < buf for n in (j - 1, j))

        probe = Probe.fit([i for i in items if i.split != fold and not near_fold(i)], vectors)
        for k, i in enumerate(items):
            if i.split != fold:
                continue
            label, _p, margin = probe.predict(i.object_id, vectors)
            if label is not None:
                out[k] = Prediction(label, conf=margin, note="probe")
    return out


# ------------------------------------------------------------------------------------------------ zero-shot
# Both read the stored SigLIP 2 vector, so only the text encoder runs.

async def stage_zs_all_stored(db, items: list[TruthItem]) -> list[Prediction]:
    """`classify_crop` as it is today (all classes, bare prompts, no negatives), on stored vectors."""
    from services.autolabel.ontology import get_ontology
    from services.labelagent.memory import load_vectors
    from services.labelagent.zeroshot import classify_all_classes

    onto = get_ontology()
    vectors = await load_vectors(db, [i.object_id for i in items])
    out = []
    for i in items:
        q = vectors.siglip.get(i.object_id)
        if q is None:
            out.append(Prediction(None, note="no crop"))
            continue
        cid, p = classify_all_classes(q, onto)
        out.append(Prediction(cid, conf=round(p, 4), note="zs_all"))
    return out


async def stage_family_zs(db, items: list[TruthItem]) -> list[Prediction]:
    """Zero-shot over the confusion-family shortlist with negatives; `conf` is the margin. The confusion
    graph is cross-fitted like the probe: built from the other half of the split."""
    from services.autolabel.ontology import get_ontology
    from services.labelagent.families import ConfusionGraph, shortlist
    from services.labelagent.memory import load_vectors
    from services.labelagent.zeroshot import classify

    onto = get_ontology()
    vectors = await load_vectors(db, [i.object_id for i in items])
    graphs = {fold: ConfusionGraph.build([i for i in items if i.split != fold]) for fold in ("test", "calib")}
    out = []
    for i in items:
        q = vectors.siglip.get(i.object_id)
        if q is None or i.pred == NOT_OBJECT:
            out.append(Prediction(None, note="no crop"))
            continue
        ans = classify(q, shortlist(i.pred, graphs[i.split], onto), onto)
        out.append(Prediction(ans.label, conf=ans.margin, note="family_zs") if ans.label is not None
                   else Prediction(None, note="family_zs:empty"))
    return out


async def stage_agent(db, items: list[TruthItem], policy=None) -> dict[str, list[Prediction]]:
    """The agent's own rule, replayed through the same `decide` the live agent uses, on the cross-fitted
    specialists above. Two readings: `agent` counts only what it changes on its own (fix and flag), and
    `agent+ask` also counts the questions as if the reviewer took every suggestion, which is the most a
    question can be worth and the most it can mislead."""
    from core.config import get_settings
    from services.autolabel.ontology import get_ontology
    from services.labelagent.agent import ASK, FIX, FLAG, Votes, decide
    from services.labelagent.memory import Vote
    from services.labelagent.zeroshot import ZsAnswer

    policy = policy or get_settings().labelagent
    name_of, _ = namers(get_ontology())
    mem = await stage_memory(db, items, k=policy.memory_k)
    prb = await stage_probe(db, items)
    zs = await stage_family_zs(db, items)
    acted, asked = [], []
    for i, m, p, z in zip(items, mem, prb, zs, strict=True):
        votes = Votes(memory=Vote(m.label, margin=m.conf or 0.0) if m.label is not None else None,
                      probe=(p.label, 0.0, p.conf or 0.0) if p.label is not None else None,
                      zs=ZsAnswer(z.label, z.conf or 0.0, {}) if z.label is not None else None)
        d = decide(i.pred, votes, policy, name_of)
        acted.append(Prediction(d.label if d.action in (FIX, FLAG) else None, note=f"agent:{d.action}"))
        asked.append(Prediction(d.label if d.action in (FIX, FLAG, ASK) else None, note=f"agent:{d.action}"))
    return {"agent": acted, "agent+ask": asked}


async def stage_agent_vlm(db, items: list[TruthItem], policy=None, *, votes: int = 2,
                          max_items: int | None = None) -> dict[str, list[Prediction]]:
    """The agent with the VLM consulted on its questions: where the rule says ASK, the VLM picks between the
    machine's class, the agreed class, "not an object" and two more of the family, twice. `agent_vlm` turns
    a question into a fix only when both votes name the agreed class; questions it does not settle stay
    questions (counted as no change). GPU and Ollama: runs outside office hours only."""
    from core.config import get_settings
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_c_vlm import VlmVerifier, make_vlm_client
    from services.labelagent import vlm as V
    from services.labelagent.agent import ASK, FIX, FLAG, Votes, decide
    from services.labelagent.families import ConfusionGraph, shortlist
    from services.labelagent.memory import Vote
    from services.labelagent.zeroshot import ZsAnswer

    settings = get_settings()
    policy = policy or settings.labelagent
    onto = get_ontology()
    name_of, _ = namers(onto)
    mem = await stage_memory(db, items, k=policy.memory_k)
    prb = await stage_probe(db, items)
    zs = await stage_family_zs(db, items)
    graphs = {f: ConfusionGraph.build([i for i in items if i.split != f]) for f in ("test", "calib")}

    decisions = []
    for i, m, p, z in zip(items, mem, prb, zs, strict=True):
        v = Votes(memory=Vote(m.label, margin=m.conf or 0.0) if m.label is not None else None,
                  probe=(p.label, 0.0, p.conf or 0.0) if p.label is not None else None,
                  zs=ZsAnswer(z.label, z.conf or 0.0, {}) if z.label is not None else None)
        decisions.append(decide(i.pred, v, policy, name_of))

    ask_idx = [k for k, d in enumerate(decisions) if d.action == ASK and items[k].pred != NOT_OBJECT]
    if max_items:
        ask_idx = ask_idx[:max_items]
    sub = [items[k] for k in ask_idx]
    verifier = VlmVerifier(make_vlm_client(settings), onto, settings)
    answers: dict[int, tuple[int | None, float]] = {}
    try:
        async for j, img in _iter_images(db, sub):
            k = ask_idx[j]
            it, d = items[k], decisions[k]
            fam = [c for c in shortlist(it.pred, graphs[it.split], onto) if c not in (it.pred, d.label, NOT_OBJECT)]
            cands = [it.pred, d.label, *fam[:2], NOT_OBJECT]
            answers[k] = V.vote(verifier, img, tuple(it.bbox), cands, onto, votes=votes) if img is not None \
                else (None, 0.0)
    finally:
        V.unload(settings)

    base, with_vlm, vlm_alone = [], [], []
    for k, d in enumerate(decisions):
        acted = d.label if d.action in (FIX, FLAG) else None
        base.append(Prediction(acted, note=f"agent:{d.action}"))
        a = answers.get(k)
        upgraded = a is not None and a[0] == d.label and a[1] >= 1.0
        with_vlm.append(Prediction(d.label if (acted is not None or upgraded) else None,
                                   note="agent_vlm:upgraded" if upgraded else f"agent:{d.action}"))
        vlm_alone.append(Prediction(a[0] if a and a[1] >= 1.0 else None, note="vlm_mc") if k in answers
                         else Prediction(None, note="no verdict"))
    return {"agent": base, "agent_vlm": with_vlm, "vlm_mc_on_asks": vlm_alone}


async def _iter_images(db, items: list[TruthItem]):
    """(index, full frame image or None) for each item, each frame read once."""
    import uuid as _uuid

    import cv2
    import numpy as np
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import Frame

    store = get_object_store()
    by_frame: dict[str, list[int]] = defaultdict(list)
    for k, i in enumerate(items):
        by_frame[i.frame_id].append(k)
    fids = list(by_frame)
    for c in range(0, len(fids), 500):
        chunk = [_uuid.UUID(f) for f in fids[c:c + 500]]
        uris = dict((await db.execute(select(Frame.frame_id, Frame.img_uri).where(Frame.frame_id.in_(chunk)))).all())
        for f in fids[c:c + 500]:
            img = None
            uri = uris.get(_uuid.UUID(f))
            if uri:
                try:
                    img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
                except Exception as exc:  # noqa: BLE001 - a missing image is a missing answer, not a failed run
                    log.warning("labelagent.frame_unreadable", frame_id=f, error=str(exc))
            for k in by_frame[f]:
                yield k, img
