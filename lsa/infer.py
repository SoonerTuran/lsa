"""Test-time latent optimisation (decoder frozen), shape evaluation and latent interpolation.

    python -m lsa.infer --ckpt <out>/checkpoints/last.pt --out <dir> [--n-shapes 50] [--interp]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from lsa.data.sampling import SAMPLING_DEFAULTS, sample_points
from lsa.data.shapes import ShapeSpec, random_specs, sdf_of
from lsa.eval.metrics import chamfer_l2, grid_metrics
from lsa.models.decoder import Decoder, build_decoder
from lsa.utils import amp_dtype, autocast, device_name, resolve_device
from lsa.viz.mesh import save_ply, slice_png, strip_png

INFER_DEFAULTS = dict(
    n_shapes=50,
    steps=800,
    lr=1.0e-2,
    lr_decay_at=0.5,
    lr_decay_factor=0.1,
    points_per_step=8192,
    points_per_shape=32768,
    seed_offset=100000,
    n_viz=4,
)


def clamped_l1(pred: Tensor, gt: Tensor, delta: float) -> Tensor:
    """Per-shape DeepSDF loss: mean over points of |clamp(pred) - clamp(gt)|, (B,N) -> (B,)."""
    return (pred.clamp(-delta, delta) - gt.clamp(-delta, delta)).abs().mean(dim=-1)


def gt_samples(specs: Sequence[ShapeSpec], M: int, seed: int, sampling: dict | None = None):
    """Exact GT samples (n,M,3), (n,M) for shapes, shape i seeded by (seed, i)."""
    kw = {**SAMPLING_DEFAULTS, **(sampling or {})}
    xs, ds = zip(*(sample_points(s, M, np.random.default_rng([seed, i]), **kw) for i, s in enumerate(specs)))
    return torch.stack(xs), torch.stack(ds)


def fit_latents(
    decoder: Decoder,
    specs: Sequence[ShapeSpec],
    steps: int = 800,
    lr: float = 1e-2,
    lr_decay_at: float = 0.5,
    lr_decay_factor: float = 0.1,
    points_per_step: int = 8192,
    points_per_shape: int = 32768,
    delta: float = 0.1,
    latent_reg: float = 1e-4,
    init_std: float = 0.01,
    seed: int = 0,
    device: torch.device | str = "cpu",
    amp: bool = False,
    sampling: dict | None = None,
    log_every: int = 0,
) -> Tensor:
    """Optimise one latent per new shape with the decoder frozen; returns Z (n,d) on device.

    All shapes are optimised jointly with the SUM of per-shape losses: Adam is elementwise,
    so this equals running each shape separately.
    """
    device = torch.device(device)
    x_all, d_all = gt_samples(specs, points_per_shape, seed, sampling)
    x_all, d_all = x_all.to(device), d_all.to(device)
    n, M = d_all.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    z = (torch.randn(n, decoder.latent_dim, generator=gen, device=device) * init_std).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    decay_step = int(lr_decay_at * steps) if lr_decay_at else steps + 1
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(decay_step, 1), gamma=lr_decay_factor)
    adt = amp_dtype(device, amp)
    scaler = torch.amp.GradScaler(device.type, enabled=adt == torch.float16)

    was_training = decoder.training
    flags = [p.requires_grad for p in decoder.parameters()]
    decoder.eval().requires_grad_(False)
    try:
        for step in range(steps):
            idx = torch.randint(0, M, (n, points_per_step), generator=gen, device=device)
            x = torch.gather(x_all, 1, idx[..., None].expand(-1, -1, 3))
            gt = torch.gather(d_all, 1, idx)
            with autocast(device, adt):
                pred = decoder(z, x)
            per_shape = clamped_l1(pred.float(), gt, delta) + latent_reg * z.pow(2).sum(-1)
            loss = per_shape.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if log_every and (step + 1) % log_every == 0:
                print(f"  fit step {step + 1}/{steps} mean loss {per_shape.mean().item():.5f}", flush=True)
    finally:
        for p, f in zip(decoder.parameters(), flags):
            p.requires_grad_(f)
        decoder.train(was_training)
    return z.detach()


@torch.no_grad()
def evaluate_shapes(
    decoder: Decoder,
    Z: Tensor,
    specs: Sequence[ShapeSpec],
    res: int = 64,
    bound: float = 1.0,
    delta: float = 0.1,
    device=None,
    n_chamfer: int = 0,
    chamfer_points: int = 30000,
    chamfer_res: int = 128,
) -> list[dict]:
    """Per-shape IoU / band-MAE on a res^3 grid (and chamfer for the first n_chamfer)."""
    was_training = decoder.training
    decoder.eval()
    rows = []
    for i, (z, spec) in enumerate(zip(Z, specs)):
        pred, gt = decoder.sdf_fn(z), sdf_of(spec)
        row = {"kind": spec.kind, **grid_metrics(pred, gt, res, bound, delta, device)}
        if i < n_chamfer:
            row["chamfer"] = chamfer_l2(pred, gt, chamfer_points, chamfer_res, bound, device, seed=i)
        rows.append(row)
    decoder.train(was_training)
    return rows


def summarize(rows: list[dict]) -> dict:
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["kind"]].append(r["iou"])
    out = {
        "n": len(rows),
        "iou_mean": float(np.mean([r["iou"] for r in rows])),
        "iou_min": float(np.min([r["iou"] for r in rows])),
        "iou_by_family": {k: float(np.mean(v)) for k, v in sorted(by_family.items())},
        "mae_mean": float(np.nanmean([r["mae"] for r in rows])),
    }
    cds = [r["chamfer"] for r in rows if "chamfer" in r]
    if cds:
        out["chamfer_mean"] = float(np.mean(cds))
        out["chamfer_median"] = float(np.median(cds))
    return out


@torch.no_grad()
def save_shape_viz(
    decoder: Decoder, z: Tensor, spec: ShapeSpec, prefix: Path, mesh_res: int = 64, slice_res: int = 128,
    device=None, gt_mesh: bool = False, title: str | None = None,
) -> None:
    """prefix.png: GT vs prediction slices; prefix.ply: predicted mesh (+ prefix_gt.ply)."""
    was_training = decoder.training
    decoder.eval()
    pred, gt = decoder.sdf_fn(z), sdf_of(spec)
    slice_png([gt, pred], f"{prefix}.png", labels=["gt", "pred"], res=slice_res, title=title, device=device)
    save_ply(pred, f"{prefix}.ply", res=mesh_res, device=device)
    if gt_mesh:
        save_ply(gt, f"{prefix}_gt.ply", res=mesh_res, device=device)
    decoder.train(was_training)


@torch.no_grad()
def interpolate(
    decoder: Decoder, z_a: Tensor, z_b: Tensor, out_dir: Path, n: int = 8, mesh_res: int = 64,
    slice_res: int = 128, device=None, names: tuple[str, str] = ("a", "b"),
) -> list[str]:
    """Meshes (and one xz-slice strip PNG) along the straight line z_a -> z_b."""
    was_training = decoder.training
    decoder.eval()
    out_dir = Path(out_dir)
    alphas = np.linspace(0.0, 1.0, n)
    fns, paths = [], []
    for k, a in enumerate(alphas):
        fn = decoder.sdf_fn((1 - a) * z_a + a * z_b)
        fns.append(fn)
        p = out_dir / f"interp_{k:02d}_alpha{a:.2f}.ply"
        if save_ply(fn, p, res=mesh_res, device=device) is not None:
            paths.append(str(p))
    labels = [f"{names[0]}" if a == 0 else (names[1] if a == 1 else f"a={a:.2f}") for a in alphas]
    strip_png(fns, out_dir / "interp_xz.png", labels=labels, normal_axis=1, res=slice_res, device=device)
    strip_png(fns, out_dir / "interp_xy.png", labels=labels, normal_axis=2, res=slice_res, device=device)
    decoder.train(was_training)
    return paths


def evaluate_unseen(decoder: Decoder, cfg: dict, out_dir: Path, device) -> dict:
    """Fit latents for fresh random shapes and evaluate them (phase-1 acceptance metric)."""
    icfg = {**INFER_DEFAULTS, **cfg.get("infer", {})}
    tcfg, dcfg, ecfg = cfg["train"], cfg["data"], cfg.get("final", {})
    specs = random_specs(icfg["n_shapes"], seed=cfg["seed"] + icfg["seed_offset"], kinds=dcfg["families"])
    t0 = time.time()
    Z = fit_latents(
        decoder, specs, steps=icfg["steps"], lr=icfg["lr"], lr_decay_at=icfg["lr_decay_at"],
        lr_decay_factor=icfg["lr_decay_factor"], points_per_step=icfg["points_per_step"],
        points_per_shape=icfg["points_per_shape"], delta=tcfg["delta"], latent_reg=tcfg["latent_reg"],
        init_std=tcfg["latent_init_std"], seed=cfg["seed"] + icfg["seed_offset"], device=device,
        amp=tcfg.get("amp", False), sampling=_sampling_cfg(dcfg), log_every=max(1, icfg["steps"] // 4),
    )
    fit_time = time.time() - t0
    res, bound = cfg["log"]["grid_res"], cfg.get("bound", 1.0)
    rows = evaluate_shapes(
        decoder, Z, specs, res, bound, tcfg["delta"], device, n_chamfer=ecfg.get("n_chamfer", 0),
        chamfer_points=ecfg.get("chamfer_points", 30000), chamfer_res=ecfg.get("chamfer_res", 128),
    )
    out_dir = Path(out_dir)
    for j in range(min(icfg["n_viz"], len(specs))):
        save_shape_viz(
            decoder, Z[j], specs[j], out_dir / f"unseen{j:02d}_{specs[j].kind}",
            ecfg.get("mesh_res", 64), cfg["log"].get("slice_res", 128), device, gt_mesh=True,
            title=f"unseen {j} {specs[j].kind} IoU={rows[j]['iou']:.3f}",
        )
    torch.save({"Z": Z.cpu(), "specs": [(s.kind, s.params) for s in specs]}, out_dir / "unseen_latents.pt")
    summary = summarize(rows)
    summary.update(fit_steps=icfg["steps"], fit_time_s=round(fit_time, 1))
    return {"summary": summary, "rows": rows}


def _sampling_cfg(dcfg: dict) -> dict:
    return {k: dcfg[k] for k in ("uniform_frac", "sigmas", "n_iters") if k in dcfg}


def load_checkpoint_decoder(ckpt_path: str | Path, device) -> tuple[Decoder, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    decoder = build_decoder(cfg["model"])
    decoder.load_state_dict(ckpt["decoder"])
    return decoder.to(device), cfg, ckpt


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Unseen-shape evaluation with a frozen decoder")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-shapes", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--seed-offset", type=int)
    ap.add_argument("--interp", action="store_true", help="also interpolate two trained latents")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    decoder, cfg, ckpt = load_checkpoint_decoder(args.ckpt, device)
    for key, val in (("n_shapes", args.n_shapes), ("steps", args.steps), ("seed_offset", args.seed_offset)):
        if val is not None:
            cfg.setdefault("infer", {})[key] = val
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = evaluate_unseen(decoder, cfg, out, device)
    result["summary"].update(ckpt=str(args.ckpt), step=ckpt["step"], device=device_name(device))
    if args.interp:
        from lsa.train.phase1 import pick_interp_pair

        specs = random_specs(cfg["data"]["n_shapes"], seed=cfg["seed"], kinds=cfg["data"]["families"])
        ia, ib = pick_interp_pair(specs)
        W = ckpt["latents"]["emb.weight"].to(device)
        interpolate(decoder, W[ia], W[ib], out / "interp", n=cfg.get("final", {}).get("interp_steps", 8),
                    device=device, names=(specs[ia].kind, specs[ib].kind))
    (out / "infer_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
