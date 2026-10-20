import numpy as np
import torch
import trimesh

from lsa.data.shapes import ShapeSpec, sdf_of
from lsa.geom.sim3 import Sim3
from lsa.sdf.compose import transform_sdf
from lsa.sdf.primitives import box
from lsa.viz.mesh import eval_grid, grid_points, mesh_from_sdf, plane_slice, save_ply, slice_png

SPHERE = sdf_of(ShapeSpec("sphere", (0.4,)))


def test_grid_points_layout():
    p = grid_points(5)
    assert p.shape == (125, 3)
    assert torch.equal(p[0], torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.equal(p[1], torch.tensor([-1.0, -1.0, -0.5]))  # z fastest


def test_eval_grid_values():
    g = eval_grid(SPHERE, res=33)
    assert g.shape == (33, 33, 33) and g.dtype == np.float32
    assert abs(g[16, 16, 16] + 0.4) < 1e-6
    assert abs(g[32, 16, 16] - 0.6) < 1e-6  # (1,0,0)


def test_marching_cubes_recovers_sphere_radius(tmp_path):
    path = tmp_path / "sphere.ply"
    mesh = save_ply(SPHERE, path, res=64)
    loaded = trimesh.load(path)
    radii = np.linalg.norm(loaded.vertices, axis=1)
    assert abs(radii.mean() - 0.4) / 0.4 < 0.02
    assert abs(radii.min() - 0.4) / 0.4 < 0.02 and abs(radii.max() - 0.4) / 0.4 < 0.02
    assert mesh.volume > 0  # outward-facing winding
    assert abs(mesh.volume - 4 / 3 * np.pi * 0.4**3) / (4 / 3 * np.pi * 0.4**3) < 0.03


def test_no_surface_returns_none(tmp_path):
    path = tmp_path / "none.ply"
    assert save_ply(lambda x: x.norm(dim=-1) + 1.0, path, res=16) is None
    assert not path.exists()


def test_mesh_of_transformed_box():
    T = Sim3.random(1, generator=torch.Generator().manual_seed(0))
    params = torch.tensor([[0.2, 0.15, 0.1]])
    f = transform_sdf(lambda p: box(p, params), T)
    mesh = mesh_from_sdf(lambda x: f(x[None])[0], res=64)
    want = 8 * 0.2 * 0.15 * 0.1 * T.s.item() ** 3
    assert abs(mesh.volume - want) / want < 0.1


def test_plane_slice_orientation():
    # sdf = x coordinate: on the xy-plane slice, value varies along u (= x)
    s = plane_slice(lambda p: p[:, 0], normal_axis=2, res=11)
    assert np.allclose(s[:, 0], np.linspace(-1, 1, 11), atol=1e-6)
    assert np.allclose(s[0, :], -1.0)


def test_slice_png_analytic_and_learned(tmp_path):
    p = tmp_path / "slices.png"
    slice_png(SPHERE, p, res=32)
    assert p.exists() and p.stat().st_size > 1000

    net = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    learned = lambda x: net(x)[:, 0] - 0.1  # noqa: E731
    p2 = tmp_path / "cmp.png"
    slice_png([SPHERE, learned], p2, labels=["gt", "pred"], res=32, title="cmp")
    assert p2.exists() and p2.stat().st_size > 1000
