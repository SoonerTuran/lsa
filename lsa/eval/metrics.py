"""Shape metrics between SDF callables (analytic or learned): grid IoU, SDF MAE in the
|gt| < delta band, symmetric chamfer-L2 between surface samples."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor

from lsa.viz.mesh import eval_grid, extract_mesh

SDFFn = Callable[[Tensor], Tensor]


def occupancy_iou(sdf_pred, sdf_gt) -> float:
    """IoU of the occupancies {sdf < 0} of two equally shaped arrays / tensors."""
    a, b = np.asarray(sdf_pred) < 0, np.asarray(sdf_gt) < 0
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def sdf_mae(sdf_pred, sdf_gt, delta: float = 0.1) -> float:
    """Mean |pred - gt| over entries with |gt| < delta (NaN if there are none)."""
    pred, gt = np.asarray(sdf_pred, dtype=np.float64), np.asarray(sdf_gt, dtype=np.float64)
    band = np.abs(gt) < delta
    return float(np.abs(pred[band] - gt[band]).mean()) if band.any() else float("nan")


def grid_metrics(
    pred_fn: SDFFn, gt_fn: SDFFn, res: int = 64, bound: float = 1.0, delta: float = 0.1, device=None
) -> dict:
    """IoU and band-MAE of two SDF callables on the same res^3 grid over [-bound,bound]^3."""
    pred = eval_grid(pred_fn, res, bound, device)
    gt = eval_grid(gt_fn, res, bound, device)
    return {"iou": occupancy_iou(pred, gt), "mae": sdf_mae(pred, gt, delta)}


def grid_iou(sdf_a: SDFFn, sdf_b: SDFFn, res: int = 64, bound: float = 1.0, device=None) -> float:
    return occupancy_iou(eval_grid(sdf_a, res, bound, device), eval_grid(sdf_b, res, bound, device))


def surface_samples(
    sdf_fn: SDFFn, n: int, res: int = 128, bound: float = 1.0, device=None, seed: int = 0
) -> np.ndarray | None:
    """n area-uniform points on the marching-cubes surface of an SDF (None if no surface)."""
    mesh = extract_mesh(eval_grid(sdf_fn, res, bound, device), bound)
    if mesh is None:
        return None
    rng = np.random.default_rng(seed)
    areas = mesh.area_faces
    tri = mesh.triangles[rng.choice(len(areas), size=n, p=areas / areas.sum())]
    u, v = rng.random((2, n, 1))
    flip = (u + v) > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    return tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])


def chamfer_points(a, b, device=None, chunk: int = 2048) -> float:
    """Symmetric chamfer-L2: mean_a min_b |a-b|^2 + mean_b min_a |a-b|^2."""
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)

    def one_way(p, q):
        # exact (non-matmul) distances: the mm shortcut adds ~1e-7 error to |a-b|^2
        d = [torch.cdist(c, q, compute_mode="donot_use_mm_for_euclid_dist").min(dim=1).values for c in p.split(chunk)]
        return torch.cat(d).pow(2).mean()

    return float(one_way(a, b) + one_way(b, a))


def chamfer_l2(
    sdf_a: SDFFn, sdf_b: SDFFn, n: int = 30000, res: int = 128, bound: float = 1.0, device=None, seed: int = 0
) -> float:
    """Chamfer-L2 between surface samples of two SDFs (inf if either has no surface)."""
    pa = surface_samples(sdf_a, n, res, bound, device, seed)
    pb = surface_samples(sdf_b, n, res, bound, device, seed + 1)
    if pa is None or pb is None:
        return float("inf")
    return chamfer_points(pa, pb, device)
