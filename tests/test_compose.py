import math

import torch

from lsa.sdf.compose import transform_sdf, union
from lsa.sdf.primitives import box, sdf_grad, sphere, torus

DT = torch.float64


def _rot_y(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=DT)


def test_union_is_min():
    a, b, c = (torch.randn(2, 50, dtype=DT) for _ in range(3))
    assert torch.equal(union(a, b, c), torch.minimum(torch.minimum(a, b), c))
    assert torch.equal(union(a), a)


def test_union_of_spheres_exact_outside_overlap():
    params = torch.tensor([[0.2]], dtype=DT)
    x = torch.rand(1, 500, 3, dtype=DT) * 2 - 1
    shift = torch.tensor([0.5, 0, 0], dtype=DT)
    d = union(sphere(x - shift, params), sphere(x + shift, params))
    want = torch.minimum((x - shift).norm(dim=-1), (x + shift).norm(dim=-1)) - 0.2
    assert torch.allclose(d, want)


def _manual_T(B=2):
    R = torch.stack([_rot_y(0.7), _rot_y(-1.9)])[:B]
    t = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.05, 0.0]], dtype=DT)[:B]
    s = torch.tensor([0.6, 1.2], dtype=DT)[:B]
    return R, t, s


def test_transformed_sphere_is_moved_scaled_sphere():
    R, t, s = _manual_T()
    params = torch.tensor([[0.3], [0.3]], dtype=DT)
    f = transform_sdf(lambda p: sphere(p, params), (R, t, s))
    y = torch.rand(2, 300, 3, dtype=DT) * 2 - 1
    want = (y - t[:, None]).norm(dim=-1) - s[:, None] * 0.3
    assert torch.allclose(f(y), want, atol=1e-12)


def test_transform_points_then_evaluate_includes_scale():
    R, t, s = _manual_T()
    params = torch.tensor([[0.2, 0.3, 0.1], [0.3, 0.1, 0.25]], dtype=DT)
    local_fn = lambda p: box(p, params)  # noqa: E731
    x = torch.rand(2, 300, 3, dtype=DT) * 2 - 1
    y = s[:, None, None] * torch.einsum("bij,bnj->bni", R, x) + t[:, None]
    f = transform_sdf(local_fn, (R, t, s))
    assert torch.allclose(f(y), s[:, None] * local_fn(x), atol=1e-12)


def test_transformed_sdf_stays_eikonal():
    R, t, s = _manual_T()
    params = torch.tensor([[0.3, 0.1], [0.25, 0.08]], dtype=DT)
    f = transform_sdf(lambda p: torus(p, params), (R, t, s))
    y = torch.rand(2, 500, 3, dtype=DT) * 2 - 1
    g = sdf_grad(f, y).norm(dim=-1)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)
