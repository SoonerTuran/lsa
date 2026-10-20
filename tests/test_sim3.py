import torch

from lsa.geom.sim3 import Sim3, matrix_to_rot6d, quaternion_to_matrix, random_rotation, rot6d_to_matrix
from lsa.sdf.compose import transform_sdf
from lsa.sdf.primitives import box, sphere

DT = torch.float64


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _assert_rotation(R, atol=1e-10):
    I = torch.eye(3, dtype=R.dtype).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, I, atol=atol)
    assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[:-2], dtype=R.dtype), atol=atol)


def test_random_ranges_and_rotation():
    T = Sim3.random(500, generator=_gen(), dtype=DT)
    _assert_rotation(T.R)
    assert T.t.abs().max() <= 0.5
    assert T.s.min() >= 0.5 and T.s.max() <= 1.25
    assert T.R.shape == (500, 3, 3) and T.t.shape == (500, 3) and T.s.shape == (500,)


def test_random_rotation_is_roughly_haar():
    # E[R] = 0 for the Haar measure on SO(3)
    R = random_rotation(20000, generator=_gen(1), dtype=DT)
    assert R.mean(0).abs().max() < 0.03


def test_random_is_seeded():
    a, b = Sim3.random(4, generator=_gen(3)), Sim3.random(4, generator=_gen(3))
    assert torch.equal(a.R, b.R) and torch.equal(a.t, b.t) and torch.equal(a.s, b.s)


def test_identity_apply():
    x = torch.randn(3, 10, 3, dtype=DT)
    assert torch.allclose(Sim3.identity(3, dtype=DT).apply(x), x)


def test_inverse_roundtrip_points():
    T = Sim3.random(8, generator=_gen(2), dtype=DT)
    x = torch.randn(8, 100, 3, dtype=DT)
    assert torch.allclose(T.inverse().apply(T.apply(x)), x, atol=1e-12)
    assert torch.allclose(T.apply(T.inverse().apply(x)), x, atol=1e-12)


def test_compose_with_inverse_is_identity():
    T = Sim3.random(8, generator=_gen(4), dtype=DT)
    I = T.compose(T.inverse())
    assert torch.allclose(I.R, torch.eye(3, dtype=DT).expand(8, 3, 3), atol=1e-12)
    assert torch.allclose(I.t, torch.zeros(8, 3, dtype=DT), atol=1e-12)
    assert torch.allclose(I.s, torch.ones(8, dtype=DT), atol=1e-12)


def test_compose_matches_sequential_apply_and_matrix():
    A = Sim3.random(5, generator=_gen(5), dtype=DT)
    B = Sim3.random(5, generator=_gen(6), dtype=DT)
    C = Sim3.random(5, generator=_gen(7), dtype=DT)
    x = torch.randn(5, 50, 3, dtype=DT)
    assert torch.allclose(A.compose(B).apply(x), A.apply(B.apply(x)), atol=1e-12)
    assert torch.allclose(A.compose(B).matrix(), A.matrix() @ B.matrix(), atol=1e-12)
    # associativity
    lhs, rhs = A.compose(B).compose(C), A.compose(B.compose(C))
    assert torch.allclose(lhs.matrix(), rhs.matrix(), atol=1e-12)


def test_rot6d_outputs_are_rotations():
    R = rot6d_to_matrix(torch.randn(1000, 6, generator=_gen(8), dtype=DT))
    _assert_rotation(R)


def test_rot6d_roundtrip():
    R = random_rotation(100, generator=_gen(9), dtype=DT)
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(R)), R, atol=1e-12)


def test_quaternion_identity():
    q = torch.tensor([[1.0, 0, 0, 0]], dtype=DT)
    assert torch.allclose(quaternion_to_matrix(q), torch.eye(3, dtype=DT)[None])


def test_transform_sdf_with_sim3():
    T = Sim3.random(4, generator=_gen(10), dtype=DT)
    p_sphere = torch.full((4, 1), 0.3, dtype=DT)
    f = transform_sdf(lambda p: sphere(p, p_sphere), T)
    y = torch.rand(4, 200, 3, dtype=DT) * 2 - 1
    assert torch.allclose(f(y), (y - T.t[:, None]).norm(dim=-1) - 0.3 * T.s[:, None], atol=1e-12)

    p_box = torch.tensor([[0.2, 0.3, 0.1]] * 4, dtype=DT)
    x = torch.rand(4, 200, 3, dtype=DT) * 2 - 1
    g = transform_sdf(lambda p: box(p, p_box), T)
    assert torch.allclose(g(T.apply(x)), T.s[:, None] * box(x, p_box), atol=1e-12)
