"""Ground-truth SDF point sampling, on-disk caches and the training Dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lsa.data.shapes import ShapeSpec, sdf_of, specs_from_arrays, specs_to_arrays
from lsa.sdf.primitives import sdf_grad

SAMPLING_DEFAULTS = dict(uniform_frac=0.4, sigmas=(0.005, 0.03), n_iters=5)


def project_to_surface(fn: Callable[[Tensor], Tensor], x: Tensor, n_iters: int = 5) -> Tensor:
    """Newton-style projection x <- x - sdf(x) * grad sdf(x) (exact SDF: one step suffices
    away from medial sets)."""
    for _ in range(n_iters):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            d = fn(xg)
            (g,) = torch.autograd.grad(d.sum(), xg)
        x = (xg - d[:, None] * g).detach()
    return x


def sample_points(
    spec: ShapeSpec,
    M: int,
    rng: np.random.Generator,
    uniform_frac: float = 0.4,
    sigmas: Sequence[float] = (0.005, 0.03),
    n_iters: int = 5,
) -> tuple[Tensor, Tensor]:
    """M points with exact GT SDF: `uniform_frac` uniform in [-1,1]^3, the rest projected
    onto the surface and perturbed by isotropic Gaussian noise, split evenly over `sigmas`.

    Returns float32 ``x (M,3)`` and ``sdf (M,)`` where sdf is the exact SDF evaluated (in
    float64) at the stored float32 points.
    """
    fn = sdf_of(spec)
    n_uniform = int(round(uniform_frac * M))
    n_near = M - n_uniform
    uniform = rng.uniform(-1.0, 1.0, size=(n_uniform, 3))

    start = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(n_near, 3)))
    surface = project_to_surface(fn, start, n_iters).numpy()
    sigma_idx = np.arange(n_near) % len(sigmas)
    noise = rng.normal(size=(n_near, 3)) * np.asarray(sigmas)[sigma_idx][:, None]
    near = surface + noise

    pts = np.concatenate([uniform, near], axis=0)[rng.permutation(M)]
    x = torch.from_numpy(pts.astype(np.float32))
    sdf = fn(x.double()).float()
    return x, sdf


def precompute(
    specs: Sequence[ShapeSpec],
    M: int,
    path: str | os.PathLike,
    seed: int,
    uniform_frac: float = 0.4,
    sigmas: Sequence[float] = (0.005, 0.03),
    n_iters: int = 5,
    log_every: int = 0,
) -> dict:
    """Sample K shapes x M points and save to an .npz (points f32 (K,M,3), sdf f32 (K,M),
    kinds, params, meta). Shape i uses rng seeded by (seed, i), independent of K."""
    K = len(specs)
    points = np.empty((K, M, 3), dtype=np.float32)
    sdf = np.empty((K, M), dtype=np.float32)
    for i, spec in enumerate(specs):
        rng = np.random.default_rng([seed, i])
        x, d = sample_points(spec, M, rng, uniform_frac, sigmas, n_iters)
        points[i], sdf[i] = x.numpy(), d.numpy()
        if log_every and (i + 1) % log_every == 0:
            print(f"  sampled {i + 1}/{K} shapes", flush=True)
    kinds, params = specs_to_arrays(specs)
    meta = dict(M=M, seed=seed, uniform_frac=uniform_frac, sigmas=list(sigmas), n_iters=n_iters)
    data = dict(points=points, sdf=sdf, kinds=kinds, params=params, meta=np.array(json.dumps(meta)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, path)
    return data


def load_samples(path: str | os.PathLike) -> dict:
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    data["meta"] = json.loads(str(data["meta"]))
    data["specs"] = specs_from_arrays(data["kinds"], data["params"])
    return data


def load_or_precompute(specs: Sequence[ShapeSpec], M: int, path, seed: int, **kw) -> dict:
    """Reuse a cache at `path` if it matches (specs, M, seed, sampling params), else rebuild."""
    want = dict(M=M, seed=seed, **{**SAMPLING_DEFAULTS, **kw})
    want["sigmas"] = list(want["sigmas"])
    path = Path(path)
    if path.exists():
        try:
            data = load_samples(path)
            if data["meta"] == want and data["specs"] == list(specs):
                return data
        except Exception as e:  # corrupt / partial cache: rebuild
            print(f"ignoring unreadable cache {path}: {e}")
    precompute(specs, M, path, seed, log_every=max(1, len(specs) // 10), **kw)
    return load_samples(path)


class SDFSamples(Dataset):
    """Per item: (shape index, random subset of `n_points` points (n,3), their GT sdf (n,)).

    A fresh subset (without replacement) is drawn on every access from torch's RNG.
    """

    def __init__(self, data: dict | str | os.PathLike, n_points: int):
        if not isinstance(data, dict):
            data = load_samples(data)
        self.points = torch.from_numpy(np.ascontiguousarray(data["points"]))
        self.sdf = torch.from_numpy(np.ascontiguousarray(data["sdf"]))
        self.n_points = n_points
        if n_points > self.points.shape[1]:
            raise ValueError(f"n_points={n_points} > stored points {self.points.shape[1]}")

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, i: int):
        idx = torch.randperm(self.points.shape[1])[: self.n_points]
        return i, self.points[i, idx], self.sdf[i, idx]
