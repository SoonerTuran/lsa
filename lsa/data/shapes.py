"""Shape specifications for the analytic families, canonicalised to radius 0.4."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from lsa.sdf.primitives import N_PARAMS, PRIMITIVES

FAMILIES: tuple[str, ...] = ("sphere", "box", "cylinder", "capsule", "torus")
CANONICAL_RADIUS = 0.4
MAX_PARAMS = max(N_PARAMS.values())


@dataclass(frozen=True)
class ShapeSpec:
    kind: str
    params: tuple[float, ...]

    def __post_init__(self):
        if self.kind not in PRIMITIVES:
            raise ValueError(f"unknown shape kind {self.kind!r}")
        if len(self.params) != N_PARAMS[self.kind]:
            raise ValueError(f"{self.kind} takes {N_PARAMS[self.kind]} params, got {len(self.params)}")


def bounding_radius(spec: ShapeSpec) -> float:
    """Max distance from the origin to any point of the (centred) shape."""
    p = spec.params
    if spec.kind == "sphere":
        return p[0]
    if spec.kind == "box":
        return math.sqrt(sum(v * v for v in p))
    if spec.kind == "cylinder":
        return math.hypot(p[0], p[1])
    if spec.kind == "capsule":
        return p[0] + p[1]
    if spec.kind == "torus":
        return p[0] + p[1]
    raise ValueError(spec.kind)


def normalize_spec(spec: ShapeSpec, radius: float = CANONICAL_RADIUS) -> ShapeSpec:
    """Uniformly rescale (all params are lengths) so the bounding radius equals `radius`."""
    scale = radius / bounding_radius(spec)
    return ShapeSpec(spec.kind, tuple(float(v * scale) for v in spec.params))


def _raw_params(kind: str, rng: np.random.Generator) -> tuple[float, ...]:
    """Un-normalised aspect parameters; ranges keep the thinnest feature >= ~0.09 after
    normalisation so shapes stay resolvable on a 64^3 grid over [-1,1]^3."""
    u = lambda lo, hi, n=None: rng.uniform(lo, hi, n)  # noqa: E731
    if kind == "sphere":
        return (1.0,)
    if kind == "box":
        return tuple(u(0.35, 1.0, 3))
    if kind == "cylinder":
        return (u(0.35, 1.0), u(0.35, 1.0))  # r, half height
    if kind == "capsule":
        return (u(0.35, 1.0), u(0.0, 1.0))  # r, half segment length
    if kind == "torus":
        return (1.0, u(0.3, 0.6))  # R, r
    raise ValueError(kind)


def random_spec(rng: np.random.Generator, kinds: Sequence[str] = FAMILIES) -> ShapeSpec:
    """Family uniform over `kinds`, then random aspect, normalised to radius 0.4."""
    kind = kinds[rng.integers(len(kinds))]
    raw = ShapeSpec(kind, tuple(float(v) for v in _raw_params(kind, rng)))
    return normalize_spec(raw)


def random_specs(n: int, seed: int, kinds: Sequence[str] = FAMILIES) -> list[ShapeSpec]:
    rng = np.random.default_rng(seed)
    return [random_spec(rng, kinds) for _ in range(n)]


def sdf_of(spec: ShapeSpec) -> Callable[[Tensor], Tensor]:
    """Exact SDF callable (N,3) -> (N,) for one shape (any dtype/device)."""
    fn = PRIMITIVES[spec.kind]

    def f(x: Tensor) -> Tensor:
        p = torch.tensor(spec.params, dtype=x.dtype, device=x.device)[None]
        return fn(x[None], p)[0]

    return f


def specs_to_arrays(specs: Sequence[ShapeSpec]) -> tuple[np.ndarray, np.ndarray]:
    """Pack specs to (kind index (K,) int64, params (K, MAX_PARAMS) float64, zero padded)."""
    kinds = np.array([FAMILIES.index(s.kind) for s in specs], dtype=np.int64)
    params = np.zeros((len(specs), MAX_PARAMS), dtype=np.float64)
    for i, s in enumerate(specs):
        params[i, : len(s.params)] = s.params
    return kinds, params


def specs_from_arrays(kinds: np.ndarray, params: np.ndarray) -> list[ShapeSpec]:
    out = []
    for k, p in zip(kinds.tolist(), params):
        kind = FAMILIES[k]
        out.append(ShapeSpec(kind, tuple(float(v) for v in p[: N_PARAMS[kind]])))
    return out
