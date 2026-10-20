import math

import pytest
import torch

from lsa.sdf.primitives import N_PARAMS, PRIMITIVES, box, capsule, cylinder, sdf_grad, sphere, torus

DT = torch.float64


def _eval(fn, params, pts):
    p = torch.tensor([params], dtype=torch.float64)
    x = torch.tensor([pts], dtype=torch.float64)
    return fn(x, p)[0]


def test_sphere_known_distances():
    d = _eval(sphere, [0.3], [[0, 0, 0], [0.3, 0, 0], [0, 1.0, 0], [0.1, 0.2, 0.2]])
    assert torch.allclose(d, torch.tensor([-0.3, 0.0, 0.7, 0.0], dtype=DT), atol=1e-12)


def test_box_face_edge_corner():
    h = [0.2, 0.3, 0.4]
    pts = [
        [0.5, 0.0, 0.0],  # face +x
        [0.0, -0.5, 0.0],  # face -y
        [0.5, 0.5, 0.0],  # edge x/y
        [0.5, 0.5, 0.5],  # corner
        [0.0, 0.0, 0.0],  # centre: nearest face is x
        [0.15, 0.0, 0.0],  # inside, near +x face
        [0.2, 0.3, 0.4],  # on the corner
    ]
    want = [
        0.3,
        0.2,
        math.sqrt(0.3**2 + 0.2**2),
        math.sqrt(0.3**2 + 0.2**2 + 0.1**2),
        -0.2,
        -0.05,
        0.0,
    ]
    assert torch.allclose(_eval(box, h, pts), torch.tensor(want, dtype=DT), atol=1e-12)


def test_cylinder_known_distances():
    pts = [[0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0], [0, 0.25, 0.0], [0.0, 0.0, -0.2]]
    want = [0.3, 0.2, math.sqrt(0.3**2 + 0.2**2), -0.2, -0.05, 0.0]
    assert torch.allclose(_eval(cylinder, [0.2, 0.3], pts), torch.tensor(want, dtype=DT), atol=1e-12)


def test_capsule_known_distances():
    pts = [[0, 0.6, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0, -0.4, 0]]
    want = [0.2, 0.4, math.sqrt(0.25 + 0.04) - 0.1, -0.1, 0.0]
    assert torch.allclose(_eval(capsule, [0.1, 0.3], pts), torch.tensor(want, dtype=DT), atol=1e-12)


def test_torus_known_distances():
    pts = [[0.3, 0, 0], [0, 0, 0], [0, 0.5, 0], [0.6, 0, 0], [0, 0, -0.4]]
    want = [-0.1, 0.2, math.sqrt(0.09 + 0.25) - 0.1, 0.2, 0.0]
    assert torch.allclose(_eval(torus, [0.3, 0.1], pts), torch.tensor(want, dtype=DT), atol=1e-12)


EXAMPLE_PARAMS = {
    "sphere": [0.35],
    "box": [0.2, 0.3, 0.15],
    "cylinder": [0.2, 0.3],
    "capsule": [0.12, 0.25],
    "torus": [0.28, 0.1],
}


@pytest.mark.parametrize("kind", list(PRIMITIVES))
def test_shapes_and_param_count(kind):
    fn = PRIMITIVES[kind]
    assert len(EXAMPLE_PARAMS[kind]) == N_PARAMS[kind]
    B, N = 3, 17
    params = torch.tensor([EXAMPLE_PARAMS[kind]] * B, dtype=DT)
    out = fn(torch.rand(B, N, 3, dtype=DT) * 2 - 1, params)
    assert out.shape == (B, N)


@pytest.mark.parametrize("kind", list(PRIMITIVES))
def test_eikonal_away_from_medial_axes(kind):
    fn = PRIMITIVES[kind]
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 4000, 3, generator=g, dtype=DT) * 2 - 1
    params = torch.tensor([EXAMPLE_PARAMS[kind]], dtype=DT)
    grad = sdf_grad(lambda p: fn(p, params), x)
    norm = grad.norm(dim=-1)
    assert torch.isfinite(grad).all()
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)


@pytest.mark.parametrize("kind", list(PRIMITIVES))
def test_gradient_matches_finite_differences(kind):
    fn = PRIMITIVES[kind]
    params = torch.tensor([EXAMPLE_PARAMS[kind]], dtype=DT)
    x = torch.rand(1, 64, 3, generator=torch.Generator().manual_seed(1), dtype=DT) * 2 - 1
    grad = sdf_grad(lambda p: fn(p, params), x)
    eps = 1e-6
    fd = torch.stack(
        [
            (fn(x + eps * e, params) - fn(x - eps * e, params)) / (2 * eps)
            for e in torch.eye(3, dtype=DT)
        ],
        dim=-1,
    )
    assert torch.allclose(grad, fd, atol=1e-5)


@pytest.mark.parametrize("kind", list(PRIMITIVES))
def test_gradient_finite_on_degenerate_points(kind):
    # centres, symmetry axis, and points with zero clamped components
    fn = PRIMITIVES[kind]
    params = torch.tensor([EXAMPLE_PARAMS[kind]], dtype=DT)
    x = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.1, 0.0], [0.05, 0.0, 0.0]]], dtype=DT)
    assert torch.isfinite(sdf_grad(lambda p: fn(p, params), x)).all()


def test_sdf_grad_float32_and_create_graph():
    params = torch.tensor([[0.3]], dtype=torch.float32, requires_grad=True)
    x = torch.randn(2, 5, 3, dtype=torch.float32)
    g = sdf_grad(lambda p: sphere(p, params.expand(2, 1)), x, create_graph=True)
    assert g.dtype == torch.float32 and g.shape == x.shape
    # unit normals of a sphere point radially
    assert torch.allclose(g, x / x.norm(dim=-1, keepdim=True), atol=1e-6)
