"""Train 3D Gaussian splats for one session window with gsplat, initialised from the LiDAR.

The recipe is the one gsplat's own simple trainer uses (Kerbl et al. 2023 as gsplat implements it): one
Gaussian per initial point with a scale from its neighbours, Adam per parameter group, L1 + D-SSIM, spherical
harmonics raised one degree every 1000 steps, and gsplat's DefaultStrategy for densifying and pruning. What is
particular to a drive:

* The initial points are the window's LiDAR scans placed by their ego poses and coloured from the frames, not a
  structure-from-motion cloud: the geometry is measured, so the splats start where the surfaces are.
* Every 8th frame is held out. The PSNR reported on those frames is a novel-view number, measured at poses the
  optimiser never saw; the training PSNR beside it says how much of that is memorisation.
* The sky has no LiDAR and never converges to geometry; it is rendered against a fixed background colour taken
  from the frames' top rows, so it does not grow floaters in front of the cameras.

Pure over arrays: no database and no object store, so it runs the same from the job and from a test.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

SH_C0 = 0.28209479177387814
HOLDOUT_EVERY = 8


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / SH_C0


def sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    return np.clip(sh0 * SH_C0 + 0.5, 0.0, 1.0)


def knn_scale(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to the k nearest neighbours, the initial Gaussian radius."""
    from scipy.spatial import cKDTree

    if len(xyz) <= k:
        return np.full(len(xyz), 0.1, dtype=np.float32)
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    return np.clip(d[:, 1:].mean(axis=1), 0.01, 2.0).astype(np.float32)


def _ssim(img1, img2, window: int = 11):
    """Structural similarity of two (B, 3, H, W) images with a Gaussian window, as in the 3DGS reference."""
    import torch
    import torch.nn.functional as F

    sigma = 1.5
    g = torch.exp(-((torch.arange(window, device=img1.device) - window // 2) ** 2).float() / (2 * sigma ** 2))
    g = (g / g.sum())
    w2 = (g[:, None] @ g[None, :])[None, None].expand(3, 1, window, window).contiguous()
    pad = window // 2
    mu1 = F.conv2d(img1, w2, padding=pad, groups=3)
    mu2 = F.conv2d(img2, w2, padding=pad, groups=3)
    s11 = F.conv2d(img1 * img1, w2, padding=pad, groups=3) - mu1 ** 2
    s22 = F.conv2d(img2 * img2, w2, padding=pad, groups=3) - mu2 ** 2
    s12 = F.conv2d(img1 * img2, w2, padding=pad, groups=3) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m.mean()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def split_holdout(n: int, every: int = HOLDOUT_EVERY) -> tuple[list[int], list[int]]:
    """(train, held-out) indices: every `every`-th frame from the 4th is held out, never the first or last."""
    held = [i for i in range(every // 2, n - 1, every)] if n >= every else []
    train = [i for i in range(n) if i not in set(held)]
    return train, held


def train_splats(images: list[np.ndarray], viewmats: np.ndarray, Ks: np.ndarray, init_xyz: np.ndarray,
                 init_rgb: np.ndarray, *, iters: int = 6000, sh_degree: int = 2, device: str = "cuda",
                 background: np.ndarray | None = None, max_gaussians: int = 1_500_000, seed: int = 0,
                 progress: Callable[[float, str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> dict:
    """Optimise splats against `images` (uint8 HxWx3 RGB) seen from `viewmats` (world -> camera, 4x4) with
    intrinsics `Ks`. Returns numpy arrays of the trained Gaussians and the train/held-out PSNR."""
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(images)
    train_idx, held_idx = split_holdout(n)
    dev = torch.device(device)
    gt = [torch.from_numpy(im).to(dev).float().div_(255.0) for im in images]
    vm = torch.from_numpy(np.asarray(viewmats, dtype=np.float32)).to(dev)
    Kt = torch.from_numpy(np.asarray(Ks, dtype=np.float32)).to(dev)
    H, W = images[0].shape[:2]
    bg = torch.tensor(background if background is not None else [0.6, 0.65, 0.7], dtype=torch.float32, device=dev)

    cams = np.linalg.inv(np.asarray(viewmats, dtype=np.float64))[:, :3, 3]
    scene_scale = float(np.linalg.norm(cams - cams.mean(axis=0), axis=1).max()) * 1.1 or 1.0
    scene_scale = max(scene_scale, 1.0)

    N = len(init_xyz)
    k_sh = (sh_degree + 1) ** 2
    scales0 = np.log(knn_scale(init_xyz))
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(init_xyz.astype(np.float32)).to(dev)),
        "scales": torch.nn.Parameter(torch.from_numpy(np.repeat(scales0[:, None], 3, axis=1)).to(dev)),
        "quats": torch.nn.Parameter(torch.from_numpy(
            np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (N, 1))).to(dev)),
        "opacities": torch.nn.Parameter(torch.full((N,), math.log(0.1 / 0.9), device=dev)),
        "sh0": torch.nn.Parameter(torch.from_numpy(rgb_to_sh0(init_rgb.astype(np.float32))[:, None, :]).to(dev)),
        "shN": torch.nn.Parameter(torch.zeros((N, k_sh - 1, 3), device=dev)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
           "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / max(1, iters)))
    strategy = DefaultStrategy(refine_stop_iter=int(iters * 0.6), reset_every=max(1000, int(iters * 0.4)),
                               refine_every=100, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    t0 = time.time()
    losses: list[float] = []
    stopped = False
    for step in range(iters):
        i = int(train_idx[rng.integers(len(train_idx))])
        deg = min(sh_degree, step // 1000)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        render, _alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=vm[i:i + 1], Ks=Kt[i:i + 1],
            width=W, height=H, sh_degree=deg, packed=False, backgrounds=bg[None], absgrad=strategy.absgrad)
        img = render[0]
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (img - gt[i]).abs().mean()
        ssim = _ssim(img.permute(2, 0, 1)[None], gt[i].permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if len(params["means"]) > max_gaussians:
            # A hard ceiling on memory: keep the most opaque splats.
            keep = torch.topk(torch.sigmoid(params["opacities"]).detach(), max_gaussians).indices
            mask = torch.zeros(len(params["means"]), dtype=torch.bool, device=dev)
            mask[keep] = True
            from gsplat.strategy.ops import remove

            remove(params, optimizers, state, ~mask)
        losses.append(float(loss.detach()))
        if step % 200 == 0 or step == iters - 1:
            if progress is not None:
                progress((step + 1) / iters, f"step {step + 1}/{iters}, {len(params['means']):,} splats")
            if should_stop is not None and should_stop():
                stopped = True
                break

    out = {k: v.detach().cpu().numpy() for k, v in params.items()}
    out["sh_degree"] = sh_degree

    def _render_idx(idx: list[int]) -> list[float]:
        vals = []
        with torch.no_grad():
            for j in idx:
                r = render_views(out, vm[j:j + 1].cpu().numpy(), Kt[j:j + 1].cpu().numpy(), W, H, device=device,
                                 background=bg.cpu().numpy())[0]
                vals.append(psnr(r, images[j].astype(np.float32) / 255.0))
        return vals

    train_psnr = _render_idx(train_idx[:: max(1, len(train_idx) // 8)])
    held_psnr = _render_idx(held_idx)
    metrics = {"iterations": step + 1, "stopped_early": stopped, "gaussians": int(len(out["means"])),
               "initial_points": int(N), "train_frames": len(train_idx), "held_out_frames": len(held_idx),
               "psnr_train": round(float(np.mean(train_psnr)), 2) if train_psnr else None,
               "psnr_held_out": round(float(np.mean(held_psnr)), 2) if held_psnr else None,
               "final_loss": round(float(np.mean(losses[-100:])), 4) if losses else None,
               "seconds": round(time.time() - t0, 1), "scene_scale_m": round(scene_scale, 2),
               "image_size": [int(W), int(H)], "held_out_indices": held_idx}
    del params, optimizers, gt
    torch.cuda.empty_cache()
    return {"gaussians": out, "metrics": metrics}


def render_views(g: dict, viewmats: np.ndarray, Ks: np.ndarray, width: int, height: int, *, device: str = "cuda",
                 background: np.ndarray | None = None, ortho: bool = False) -> list[np.ndarray]:
    """Render trained splats from the given cameras. Returns float32 HxWx3 images in 0..1."""
    import torch
    from gsplat import rasterization

    dev = torch.device(device)
    t = {k: torch.from_numpy(np.asarray(g[k], dtype=np.float32)).to(dev)
         for k in ("means", "scales", "quats", "opacities", "sh0", "shN")}
    colors = torch.cat([t["sh0"], t["shN"]], dim=1)
    bg = torch.tensor(background if background is not None else [0.6, 0.65, 0.7], dtype=torch.float32, device=dev)
    outs = []
    with torch.no_grad():
        for i in range(len(viewmats)):
            r, _a, _ = rasterization(
                means=t["means"], quats=t["quats"], scales=torch.exp(t["scales"]),
                opacities=torch.sigmoid(t["opacities"]), colors=colors,
                viewmats=torch.from_numpy(np.asarray(viewmats[i:i + 1], np.float32)).to(dev),
                Ks=torch.from_numpy(np.asarray(Ks[i:i + 1], np.float32)).to(dev), width=int(width),
                height=int(height), sh_degree=int(g.get("sh_degree", 0)), packed=False, backgrounds=bg[None],
                camera_model="ortho" if ortho else "pinhole")
            outs.append(r[0].clamp(0, 1).cpu().numpy())
    return outs


def to_ply(g: dict) -> bytes:
    """The splats in the standard 3DGS .ply layout (x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*), which
    common splat viewers open as is."""
    means = np.asarray(g["means"], np.float32)
    n = len(means)
    sh0 = np.asarray(g["sh0"], np.float32).reshape(n, 3)
    shn = np.asarray(g["shN"], np.float32)
    # 3DGS stores the rest coefficients channel-major: all of R's, then G's, then B's.
    rest = shn.transpose(0, 2, 1).reshape(n, -1)
    cols = [means, np.zeros((n, 3), np.float32), sh0, rest,
            np.asarray(g["opacities"], np.float32).reshape(n, 1), np.asarray(g["scales"], np.float32),
            np.asarray(g["quats"], np.float32)]
    data = np.concatenate(cols, axis=1).astype("<f4")
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                                "rot_0", "rot_1", "rot_2", "rot_3"])
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
        "".join(f"property float {p}\n" for p in names) + "end_header\n"
    return header.encode("ascii") + data.tobytes()
