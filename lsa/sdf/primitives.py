"""Exact, batched signed distance functions of the analytic primitives.

All primitives live in the canonical frame: centred at the origin, axis-aligned and
upright (symmetry axis = y). Formulas follow Inigo Quilez
(https://iquilezles.org/articles/distfunctions/).

Signature of every primitive: ``fn(points (B,N,3), params (B,k)) -> sdf (B,N)``.
Negative inside, positive outside, exact Euclidean distance, differentiable
(autograd gives zero sub-gradients on measure-zero medial sets).
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def _norm(x: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x, dim=-1)


def _p(params: Tensor, i: int) -> Tensor:
    """Column i of params (B,k) broadcastable against (B,N)."""
    return params[:, i : i + 1]


def sphere(points: Tensor, params: Tensor) -> Tensor:
    """params = [r]."""
    return _norm(points) - _p(params, 0)


def box(points: Tensor, params: Tensor) -> Tensor:
    """params = [hx, hy, hz] half extents."""
    q = points.abs() - params[:, None, :]
    outside = _norm(q.clamp(min=0.0))
    inside = q.max(dim=-1).values.clamp(max=0.0)
    return outside + inside


def cylinder(points: Tensor, params: Tensor) -> Tensor:
    """Capped cylinder along y. params = [r, h], h = half height."""
    radial = _norm(points[..., [0, 2]])
    d = torch.stack(
        [radial - _p(params, 0), points[..., 1].abs() - _p(params, 1)], dim=-1
    )
    inside = d.max(dim=-1).values.clamp(max=0.0)
    outside = _norm(d.clamp(min=0.0))
    return inside + outside


def capsule(points: Tensor, params: Tensor) -> Tensor:
    """Vertical capsule: segment (0,-h,0)-(0,h,0) swept by radius r. params = [r, h]."""
    r, h = _p(params, 0), _p(params, 1)
    y = points[..., 1]
    offset = torch.stack(
        [torch.zeros_like(y), torch.maximum(torch.minimum(y, h), -h), torch.zeros_like(y)],
        dim=-1,
    )
    return _norm(points - offset) - r


def torus(points: Tensor, params: Tensor) -> Tensor:
    """Torus in the xz-plane (axis y). params = [R, r], major / minor radius."""
    q = torch.stack(
        [_norm(points[..., [0, 2]]) - _p(params, 0), points[..., 1]], dim=-1
    )
    return _norm(q) - _p(params, 1)


PRIMITIVES: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "sphere": sphere,
    "box": box,
    "cylinder": cylinder,
    "capsule": capsule,
    "torus": torus,
}

N_PARAMS: dict[str, int] = {"sphere": 1, "box": 3, "cylinder": 2, "capsule": 2, "torus": 2}


def sdf_grad(fn: Callable[[Tensor], Tensor], x: Tensor, create_graph: bool = False) -> Tensor:
    """Spatial gradient of an SDF callable ``fn(x) -> (...,)`` at x (...,3).

    For exact SDFs this is the unit outward normal away from medial sets (|grad| = 1).
    Works for learned SDFs too; pass ``create_graph=True`` to backprop through it.
    """
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(fn(x).sum(), x, create_graph=create_graph)
    return g
