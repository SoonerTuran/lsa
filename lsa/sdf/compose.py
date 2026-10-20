"""Exact SDF composition: union and Sim(3) transforms."""
from __future__ import annotations

import functools
from typing import Callable

import torch
from torch import Tensor

SDFFn = Callable[[Tensor], Tensor]


def union(*sdf_values: Tensor) -> Tensor:
    """Union of shapes = pointwise min of their SDF values (all of equal shape)."""
    if not sdf_values:
        raise ValueError("union() needs at least one SDF tensor")
    return functools.reduce(torch.minimum, sdf_values)


def transform_sdf(sdf_fn: SDFFn, T) -> SDFFn:
    """SDF of the shape moved by a similarity T: x -> s R x + t.

    ``T`` is a :class:`lsa.geom.sim3.Sim3` (or anything with batched ``R (B,3,3)``,
    ``t (B,3)``, ``s (B,)``; a tuple ``(R, t, s)`` also works). ``sdf_fn`` maps
    (B,N,3) -> (B,N). The returned callable evaluates ``s * sdf_fn(T^{-1} y)``, which is
    exact because similarities scale all distances by s.
    """
    R, t, s = (T.R, T.t, T.s) if hasattr(T, "R") else T

    def fn(y: Tensor) -> Tensor:
        local = torch.einsum("bji,bnj->bni", R, y - t[:, None, :]) / s[:, None, None]
        return s[:, None] * sdf_fn(local)

    return fn
