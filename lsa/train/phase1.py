"""Phase 1: DeepSDF auto-decoder on analytic primitives.

Writes into <out>: config.yaml, data/train.npz (GT cache), train_log.csv, eval_log.csv,
viz/gt/*, viz/step_XXXXXX/*, checkpoints/last.pt, final/{train,unseen}/*, interp/*,
metrics.json. Re-running with the same <out> resumes from checkpoints/last.pt.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lsa.data.sampling import SDFSamples, load_or_precompute
from lsa.data.shapes import FAMILIES, ShapeSpec, random_specs
from lsa.infer import (
    _sampling_cfg,
    clamped_l1,
    evaluate_shapes,
    evaluate_unseen,
    interpolate,
    save_shape_viz,
    summarize,
)
from lsa.models.decoder import build_decoder
from lsa.models.latent import LatentTable
from lsa.utils import (
    amp_dtype,
    atomic_save,
    autocast,
    device_name,
    resolve_device,
    rng_state,
    save_yaml,
    set_rng_state,
    set_seed,
)

ACCEPT_TRAIN_IOU = 0.95
ACCEPT_UNSEEN_IOU = 0.90


def pick_viz_indices(specs: list[ShapeSpec], n: int) -> list[int]:
    """First shape of each family (non-sphere families first), padded with early indices."""
    order = ["box", "cylinder", "capsule", "torus", "sphere"]
    picked = []
    for kind in order:
        idx = next((i for i, s in enumerate(specs) if s.kind == kind), None)
        if idx is not None and len(picked) < n:
            picked.append(idx)
    picked += [i for i in range(len(specs)) if i not in picked][: n - len(picked)]
    return picked


def pick_interp_pair(specs: list[ShapeSpec]) -> tuple[int, int]:
    """First box and first torus (fallback: the first two shapes of different families)."""
    first = {}
    for i, s in enumerate(specs):
        first.setdefault(s.kind, i)
    if "box" in first and "torus" in first:
        return first["box"], first["torus"]
    kinds = list(first)
    return (first[kinds[0]], first[kinds[1]]) if len(kinds) > 1 else (0, min(1, len(specs) - 1))


class CsvLog:
    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = path, fields
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fields)

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])


def run(cfg: dict, out: str | Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dcfg, mcfg, tcfg, lcfg, fcfg = cfg["data"], cfg["model"], cfg["train"], cfg["log"], cfg.get("final", {})
    bound = cfg.get("bound", 1.0)
    save_yaml(cfg, out / "config.yaml")
    print(f"[phase1] out={out} device={device_name(device)}", flush=True)

    # ---- data -------------------------------------------------------------------------
    K = dcfg["n_shapes"]
    specs = random_specs(K, seed=cfg["seed"], kinds=dcfg.get("families", FAMILIES))
    t0 = time.time()
    data = load_or_precompute(
        specs, dcfg["points_per_shape"], out / "data" / "train.npz", seed=cfg["seed"], **_sampling_cfg(dcfg)
    )
    print(f"[phase1] data: {K} shapes x {dcfg['points_per_shape']} pts ({time.time() - t0:.1f}s)", flush=True)
    dataset = SDFSamples(data, dcfg["points_per_step"])
    B = min(tcfg["shapes_per_step"], K)
    nw = tcfg.get("num_workers", 0)
    loader = DataLoader(
        dataset, batch_size=B, shuffle=True, drop_last=True, num_workers=nw,
        pin_memory=device.type == "cuda", persistent_workers=nw > 0,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    # ---- model / optimiser ------------------------------------------------------------
    decoder = build_decoder(mcfg).to(device)
    latents = LatentTable(K, mcfg["latent_dim"], tcfg["latent_init_std"]).to(device)
    opt = torch.optim.Adam(
        [{"params": decoder.parameters(), "lr": tcfg["lr_decoder"]}, {"params": latents.parameters(), "lr": tcfg["lr_latent"]}]
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=tcfg["lr_decay_every"], gamma=tcfg["lr_decay_gamma"])
    adt = amp_dtype(device, tcfg.get("amp", False))
    scaler = torch.amp.GradScaler(device.type, enabled=adt == torch.float16)
    delta, steps = tcfg["delta"], tcfg["steps"]

    ckpt_path = out / "checkpoints" / "last.pt"
    step, train_time = 0, 0.0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        decoder.load_state_dict(ckpt["decoder"])
        latents.load_state_dict(ckpt["latents"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        set_rng_state(ckpt["rng"])
        step, train_time = ckpt["step"], ckpt["train_time"]
        print(f"[phase1] resumed from {ckpt_path} at step {step}", flush=True)

    def save_ckpt():
        atomic_save(
            {
                "step": step, "train_time": train_time, "config": cfg,
                "decoder": decoder.state_dict(), "latents": latents.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state(),
            },
            ckpt_path,
        )

    eval_idx = list(range(min(lcfg["n_eval_shapes"], K)))
    viz_idx = pick_viz_indices(specs, lcfg["n_viz_shapes"])
    res, slice_res, mesh_res = lcfg["grid_res"], lcfg.get("slice_res", 128), fcfg.get("mesh_res", 64)

    def evaluate(tag: str) -> dict:
        Z = latents.weight.detach()
        rows = evaluate_shapes(decoder, Z[eval_idx], [specs[i] for i in eval_idx], res, bound, delta, device)
        summ = summarize(rows)
        vdir = out / "viz" / tag
        for i in viz_idx:
            save_shape_viz(decoder, Z[i], specs[i], vdir / f"shape{i:04d}_{specs[i].kind}", mesh_res, slice_res,
                           device, title=f"{tag} shape {i} ({specs[i].kind})")
        return summ

    if step == 0:
        from lsa.data.shapes import sdf_of
        from lsa.viz.mesh import save_ply, slice_png

        for i in viz_idx:
            prefix = out / "viz" / "gt" / f"shape{i:04d}_{specs[i].kind}"
            slice_png(sdf_of(specs[i]), f"{prefix}.png", labels=["gt"], res=slice_res)
            save_ply(sdf_of(specs[i]), f"{prefix}.ply", res=mesh_res)

    train_log = CsvLog(out / "train_log.csv", ["step", "loss", "loss_sdf", "loss_reg", "lr_decoder", "lr_latent", "steps_per_s", "train_time_s"])
    eval_log = CsvLog(out / "eval_log.csv", ["step", "iou_mean", "iou_min", "mae_mean"] + [f"iou_{k}" for k in FAMILIES])

    # ---- training loop ----------------------------------------------------------------
    decoder.train()
    it = iter(loader)
    acc = {"loss": 0.0, "loss_sdf": 0.0, "loss_reg": 0.0, "n": 0}
    t_seg, step_seg = time.time(), step
    while step < steps:
        try:
            idx, pts, gt = next(it)
        except StopIteration:
            it = iter(loader)
            idx, pts, gt = next(it)
        idx, pts, gt = idx.to(device), pts.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        z = latents(idx)
        with autocast(device, adt):
            pred = decoder(z, pts)
        loss_sdf = clamped_l1(pred.float(), gt, delta).mean()
        loss_reg = tcfg["latent_reg"] * z.pow(2).sum(-1).mean()
        loss = loss_sdf + loss_reg
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        acc["loss"] += loss.item()
        acc["loss_sdf"] += loss_sdf.item()
        acc["loss_reg"] += loss_reg.item()
        acc["n"] += 1

        last = step == steps
        if step % lcfg["every"] == 0 or last:
            now = time.time()
            train_time += now - t_seg
            sps = (step - step_seg) / max(now - t_seg, 1e-9)
            n = max(acc["n"], 1)
            row = {
                "step": step, "loss": acc["loss"] / n, "loss_sdf": acc["loss_sdf"] / n, "loss_reg": acc["loss_reg"] / n,
                "lr_decoder": opt.param_groups[0]["lr"], "lr_latent": opt.param_groups[1]["lr"],
                "steps_per_s": round(sps, 3), "train_time_s": round(train_time, 1),
            }
            train_log.write(row)
            print(f"[phase1] step {step}/{steps} loss {row['loss']:.5f} (sdf {row['loss_sdf']:.5f}) "
                  f"{sps:.2f} it/s", flush=True)
            acc = {"loss": 0.0, "loss_sdf": 0.0, "loss_reg": 0.0, "n": 0}
            t_seg, step_seg = time.time(), step
        if step % lcfg["eval_every"] == 0 or last:
            summ = evaluate(f"step_{step:06d}")
            eval_log.write({"step": step, "iou_mean": summ["iou_mean"], "iou_min": summ["iou_min"], "mae_mean": summ["mae_mean"],
                            **{f"iou_{k}": v for k, v in summ["iou_by_family"].items()}})
            print(f"[phase1] eval step {step}: IoU {summ['iou_mean']:.4f} (min {summ['iou_min']:.3f}) "
                  f"MAE {summ['mae_mean']:.5f}", flush=True)
            t_seg = time.time()  # exclude eval from throughput
        if step % lcfg["ckpt_every"] == 0 or last:
            save_ckpt()

    # ---- final evaluation ---------------------------------------------------------------
    metrics_path = out / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text()).get("step") == step:
        print(f"[phase1] training complete; metrics already in {metrics_path}", flush=True)
        return json.loads(metrics_path.read_text())

    t_final = time.time()
    Z = latents.weight.detach()
    n_train = min(fcfg.get("n_train_eval", 200), K)
    rows = evaluate_shapes(
        decoder, Z[:n_train], specs[:n_train], res, bound, delta, device, n_chamfer=fcfg.get("n_chamfer", 0),
        chamfer_points=fcfg.get("chamfer_points", 30000), chamfer_res=fcfg.get("chamfer_res", 128),
    )
    train_summary = summarize(rows)
    for i in viz_idx:
        save_shape_viz(decoder, Z[i], specs[i], out / "final" / "train" / f"shape{i:04d}_{specs[i].kind}",
                       mesh_res, slice_res, device, gt_mesh=True, title=f"train shape {i} ({specs[i].kind})")
    print(f"[phase1] final train IoU {train_summary['iou_mean']:.4f} over {n_train} shapes", flush=True)

    unseen = evaluate_unseen(decoder, cfg, out / "final" / "unseen", device)
    print(f"[phase1] unseen IoU {unseen['summary']['iou_mean']:.4f} over {unseen['summary']['n']} shapes", flush=True)

    ia, ib = pick_interp_pair(specs)
    interpolate(decoder, Z[ia], Z[ib], out / "interp", n=fcfg.get("interp_steps", 8), mesh_res=mesh_res,
                slice_res=slice_res, device=device, names=(specs[ia].kind, specs[ib].kind))

    metrics = {
        "config": cfg.get("name"),
        "step": step,
        "device": device_name(device),
        "train_time_s": round(train_time, 1),
        "final_eval_time_s": round(time.time() - t_final, 1),
        "session_wall_time_s": round(time.time() - t_start, 1),
        "grid_res": res,
        "train": train_summary,
        "unseen": unseen["summary"],
        "interp_pair": {"a": [ia, specs[ia].kind], "b": [ib, specs[ib].kind]},
        "acceptance": {
            f"train_iou>{ACCEPT_TRAIN_IOU}": train_summary["iou_mean"] > ACCEPT_TRAIN_IOU,
            f"unseen_iou>{ACCEPT_UNSEEN_IOU}": unseen["summary"]["iou_mean"] > ACCEPT_UNSEEN_IOU,
        },
    }
    (out / "final").mkdir(parents=True, exist_ok=True)
    (out / "final" / "per_shape.json").write_text(json.dumps({"train": rows, "unseen": unseen["rows"]}, indent=1))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
