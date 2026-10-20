import collections

import numpy as np
import pytest
import torch

from lsa.data.shapes import (
    CANONICAL_RADIUS,
    FAMILIES,
    ShapeSpec,
    bounding_radius,
    normalize_spec,
    random_spec,
    random_specs,
    sdf_of,
    specs_from_arrays,
    specs_to_arrays,
)
from lsa.sdf.primitives import PRIMITIVES

DT = torch.float64


def test_spec_validation():
    with pytest.raises(ValueError):
        ShapeSpec("cone", (1.0,))
    with pytest.raises(ValueError):
        ShapeSpec("box", (1.0, 2.0))


def test_random_specs_are_seeded_and_cover_families():
    a, b = random_specs(500, seed=3), random_specs(500, seed=3)
    assert a == b
    counts = collections.Counter(s.kind for s in a)
    assert set(counts) == set(FAMILIES)
    assert min(counts.values()) > 60  # ~100 each


@pytest.mark.parametrize("kind", FAMILIES)
def test_normalised_to_canonical_radius(kind):
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = random_spec(rng, kinds=[kind])
        assert abs(bounding_radius(s) - CANONICAL_RADIUS) < 1e-12


@pytest.mark.parametrize("kind", FAMILIES)
def test_shape_fits_inside_ball_and_touches_it(kind):
    rng = np.random.default_rng(1)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(20000, 3, generator=g, dtype=DT) * 2 - 1
    for _ in range(20):
        s = random_spec(rng, kinds=[kind])
        d = sdf_of(s)(x)
        # distance to a subset of the ball is >= distance to the ball
        assert (d >= x.norm(dim=-1) - CANONICAL_RADIUS - 1e-12).all()
        # the shape reaches the ball: some point at radius 0.4 is on the surface
        far_dirs = torch.nn.functional.normalize(x, dim=-1) * CANONICAL_RADIUS
        assert sdf_of(s)(far_dirs).min() < 0.02


@pytest.mark.parametrize("kind", FAMILIES)
def test_minimum_thickness(kind):
    # thinnest feature (half-thickness) stays resolvable on the 64^3 eval grid
    rng = np.random.default_rng(2)
    for _ in range(200):
        s = random_spec(rng, kinds=[kind])
        p = s.params
        thin = p[1] if kind == "torus" else (min(p) if kind in ("box", "cylinder") else p[0])
        assert thin >= 0.09


def test_sdf_of_matches_primitive():
    s = ShapeSpec("torus", (0.3, 0.1))
    x = torch.rand(100, 3, dtype=DT) * 2 - 1
    want = PRIMITIVES["torus"](x[None], torch.tensor([[0.3, 0.1]], dtype=DT))[0]
    assert torch.equal(sdf_of(s)(x), want)
    assert sdf_of(s)(x.float()).dtype == torch.float32


def test_normalize_scales_uniformly():
    s = normalize_spec(ShapeSpec("box", (1.0, 2.0, 2.0)))
    assert np.allclose(s.params, (0.4 / 3, 0.8 / 3, 0.8 / 3))


def test_array_roundtrip():
    specs = random_specs(40, seed=5)
    kinds, params = specs_to_arrays(specs)
    assert kinds.shape == (40,) and params.shape == (40, 3)
    assert specs_from_arrays(kinds, params) == specs
