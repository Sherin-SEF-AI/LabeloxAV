"""Pod-side continued self-supervised pretraining of the DINOv3 ViT-B/16 backbone on Indian road frames.

Runs on the RunPod A100. The local side (`services/training/tasks/pretrain.py`) builds a manifest of real,
deduplicated frames and publishes it with the images; this reads that manifest, runs a DINO/iBOT-style
self-distillation for a fixed step budget, and writes checkpoints back to the workspace where the local
side collects them.

Why continued pretraining rather than fine-tuning: the objective needs no labels, so it learns from all
41,204 real frames rather than the labelled fraction, and the thing being fixed is that the backbone has
never seen an autorickshaw, a hoarding, a metro pillar, or traffic at Indian densities. Everything
downstream of the embedding (duplicate detection, similarity, clustering, novelty) inherits that gap.

The objective, briefly. A student and a teacher see different augmented crops of the same frame. The
teacher is an exponential moving average of the student and takes the two global crops; the student takes
those plus several local crops. The loss is the cross-entropy between the student's output distribution
and the teacher's sharpened, centred one, so the student must predict what the teacher sees from a
different and often smaller view. Centring and sharpening are what stop the pair collapsing onto one
constant output, which is the failure mode of every self-distillation objective without them.

Steps rather than epochs: the local side has to price the job before the pod has seen the data.

No em-dashes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("PRETRAIN_BASE_MODEL", "vit_base_patch16_dinov3.lvd1689m")
# Crop counts from the DINO recipe: two global views the teacher also sees, and several small local views
# only the student sees. The local crops are what force the invariance to be learned rather than memorised.
N_GLOBAL, N_LOCAL = 2, 6
GLOBAL_PX, LOCAL_PX = 224, 96
OUT_DIM = 65536              # the projection head's output dimension, as in the DINO reference
TEACHER_TEMP, STUDENT_TEMP = 0.04, 0.1
CENTER_MOMENTUM = 0.9
EMA_BASE, EMA_FINAL = 0.996, 1.0


def _augment(image, size: int, strong: bool):
    """One augmented view. Colour jitter and blur, because the invariances wanted here are photometric."""
    from torchvision import transforms as T

    ops = [
        T.RandomResizedCrop(size, scale=(0.25, 1.0) if size == GLOBAL_PX else (0.05, 0.35),
                            antialias=True),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(0.2),
    ]
    if strong:
        ops.append(T.GaussianBlur(kernel_size=size // 20 * 2 + 1))
    ops += [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return T.Compose(ops)(image)


class MultiCropDataset:
    """Frames from the manifest, each returned as its global and local crops."""

    def __init__(self, manifest_path: Path, images_dir: Path):
        data = json.loads(Path(manifest_path).read_text())
        self.records = data["frames"]
        self.images_dir = Path(images_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        from PIL import Image

        rec = self.records[i]
        # The pod is given the images by relative path; the manifest's img_uri is the object-store key and
        # its basename is what the transfer wrote.
        name = str(rec.get("path") or rec["img_uri"]).split("/")[-1]
        img = Image.open(self.images_dir / name).convert("RGB")
        globals_ = [_augment(img, GLOBAL_PX, strong=(k == 0)) for k in range(N_GLOBAL)]
        locals_ = [_augment(img, LOCAL_PX, strong=False) for _ in range(N_LOCAL)]
        return globals_ + locals_


class Head:
    """The projection head, built lazily so importing this module needs no torch."""

    @staticmethod
    def build(in_dim: int, out_dim: int = OUT_DIM):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(in_dim, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(),
            nn.Linear(2048, 256),
            nn.utils.weight_norm(nn.Linear(256, out_dim, bias=False)),
        )


def _loss(student_out, teacher_out, center, n_global: int, n_crops: int):
    """DINO cross-entropy: every student view predicts every teacher view except its own."""
    import torch
    import torch.nn.functional as F

    s = [F.log_softmax(o / STUDENT_TEMP, dim=-1) for o in student_out.chunk(n_crops)]
    t = [F.softmax((o - center) / TEACHER_TEMP, dim=-1).detach()
         for o in teacher_out.chunk(n_global)]
    total, n = 0.0, 0
    for ti, tv in enumerate(t):
        for si, sv in enumerate(s):
            if si == ti:
                continue          # a view predicting itself teaches nothing
            total = total + torch.sum(-tv * sv, dim=-1).mean()
            n += 1
    return total / max(n, 1)


def main() -> None:
    import timm
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="Continued DINO/iBOT pretraining on road frames.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    student = timm.create_model(args.model, pretrained=True, num_classes=0).to(dev)
    teacher = timm.create_model(args.model, pretrained=True, num_classes=0).to(dev)
    in_dim = student.num_features
    s_head, t_head = Head.build(in_dim).to(dev), Head.build(in_dim).to(dev)
    t_head.load_state_dict(s_head.state_dict())
    for p in list(teacher.parameters()) + list(t_head.parameters()):
        p.requires_grad = False

    ds = MultiCropDataset(Path(args.manifest), Path(args.images))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True,
                    pin_memory=True)
    params = list(student.parameters()) + list(s_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.04)
    center = torch.zeros(1, OUT_DIM, device=dev)
    n_crops = N_GLOBAL + N_LOCAL

    step, t0 = 0, time.time()
    log_path = out / "pretrain_log.jsonl"
    while step < args.steps:
        for crops in dl:
            if step >= args.steps:
                break
            views = [c.to(dev, non_blocking=True) for c in crops]
            s_out = s_head(student(torch.cat(views, dim=0)))
            with torch.no_grad():
                t_out = t_head(teacher(torch.cat(views[:N_GLOBAL], dim=0)))
            loss = _loss(s_out, t_out, center, N_GLOBAL, n_crops)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()

            with torch.no_grad():
                # Centring keeps the teacher from collapsing onto one output; the EMA schedule makes the
                # teacher a slowly-moving target so the student always has something to chase.
                center.mul_(CENTER_MOMENTUM).add_(t_out.mean(dim=0, keepdim=True),
                                                  alpha=1 - CENTER_MOMENTUM)
                m = EMA_FINAL - (EMA_FINAL - EMA_BASE) * (math.cos(math.pi * step / args.steps) + 1) / 2
                for pt, ps in zip(teacher.parameters(), student.parameters(), strict=True):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                for pt, ps in zip(t_head.parameters(), s_head.parameters(), strict=True):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)

            step += 1
            if step % 50 == 0:
                rec = {"step": step, "loss": float(loss.item()), "ema": round(float(m), 5),
                       "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                with log_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if step % args.ckpt_every == 0 or step == args.steps:
                # Checkpoints go back often so a pod that dies mid-run leaves usable weights behind
                # rather than the whole spend being wasted.
                torch.save({"model": student.state_dict(), "step": step, "base": args.model},
                           out / f"backbone_step{step}.pt")

    torch.save({"model": student.state_dict(), "step": step, "base": args.model}, out / "backbone.pt")
    (out / "result.json").write_text(json.dumps(
        {"steps": step, "frames": len(ds), "base_model": args.model,
         "elapsed_s": round(time.time() - t0, 1), "objective": "dino"}, indent=2))
    print(json.dumps({"done": True, "steps": step, "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
