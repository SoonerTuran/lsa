"""Cross-module consistency: sampled GT, Sim(3) transforms and unions agree exactly."""
import numpy as np
import torch

from lsa.data.sampling import sample_points
from lsa.data.shapes import random_specs, sdf_of
from lsa.geom.sim3 import Sim3
from lsa.sdf.compose import transform_sdf, union
from lsa.sdf.primitives import PRIMITIVES

DT = torch.float64


def _batched(spec):
    p = torch.tensor([spec.params], dtype=DT)
    return lambda x: PRIMITIVES[spec.kind](x, p)


def test_scene_of_transformed_shapes_matches_pointwise_definition():
    specs = random_specs(3, seed=7)
    Ts = [Sim3.random(1, generator=torch.Generator().manual_seed(i), dtype=DT) for i in range(3)]
    y = torch.rand(1, 2000, 3, dtype=DT) * 2 - 1
    scene = union(*[transform_sdf(_batched(s), T)(y) for s, T in zip(specs, Ts)])
    # reference: map world points into each local frame by hand
    ref = []
    for s, T in zip(specs, Ts):
        local = T.inverse().apply(y)
        ref.append(T.s[:, None] * sdf_of(s)(local[0])[None])
    assert torch.allclose(scene, torch.stack(ref).min(0).values, atol=1e-12)


def test_sampled_points_map_through_pose_exactly():
    spec = random_specs(1, seed=8)[0]
    x, d = sample_points(spec, 2000, np.random.default_rng(0))
    T = Sim3.random(1, generator=torch.Generator().manual_seed(9), dtype=DT)
    y = T.apply(x.double()[None])
    moved = transform_sdf(_batched(spec), T)(y)[0]
    assert torch.allclose(moved, T.s * d.double(), atol=1e-6)
