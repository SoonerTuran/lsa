import math

import numpy as np
import torch

from lsa.data.shapes import ShapeSpec, sdf_of
from lsa.eval.metrics import (
    chamfer_l2,
    chamfer_points,
    grid_iou,
    grid_metrics,
    occupancy_iou,
    sdf_mae,
    surface_samples,
)

S3 = sdf_of(ShapeSpec("sphere", (0.3,)))
S4 = sdf_of(ShapeSpec("sphere", (0.4,)))


def test_occupancy_iou_basic():
    a = np.array([-1.0, -1.0, 1.0, 1.0])
    b = np.array([-1.0, 1.0, -1.0, 1.0])
    assert occupancy_iou(a, a) == 1.0
    assert occupancy_iou(a, b) == 1 / 3
    assert occupancy_iou(np.ones(4), np.ones(4)) == 1.0  # both empty


def test_grid_iou_nested_spheres():
    assert grid_iou(S4, S4, res=48) == 1.0
    iou = grid_iou(S3, S4, res=64)
    assert abs(iou - (0.3 / 0.4) ** 3) < 0.03


def test_sdf_mae_band():
    gt = np.linspace(-0.5, 0.5, 101)
    assert sdf_mae(gt, gt) == 0.0
    assert math.isclose(sdf_mae(gt + 0.01, gt, delta=0.1), 0.01, rel_tol=1e-9)
    # errors outside the band are ignored
    pred = gt.copy()
    pred[np.abs(gt) >= 0.1] += 5.0
    assert sdf_mae(pred, gt, delta=0.1) == 0.0


def test_grid_metrics_offset_sphere():
    m = grid_metrics(lambda x: S4(x) + 0.01, S4, res=32)
    assert abs(m["mae"] - 0.01) < 1e-6 and m["iou"] < 1.0


def test_surface_samples_on_surface():
    p = surface_samples(S4, 5000, res=64)
    assert p.shape == (5000, 3)
    r = np.linalg.norm(p, axis=1)
    assert np.abs(r - 0.4).max() < 0.005
    # roughly uniform on the sphere: centroid near the origin
    assert np.abs(p.mean(0)).max() < 0.02


def test_chamfer():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 3))
    assert chamfer_points(a, a) == 0.0
    one = np.zeros((1, 3))
    assert math.isclose(chamfer_points(one, one + [0.1, 0, 0]), 2 * 0.01, rel_tol=1e-5)
    assert 0 < chamfer_points(a, a + [0.1, 0, 0]) <= 2 * 0.01 + 1e-6  # NN may be closer
    assert chamfer_l2(S4, S4, n=8000, res=64) < 2e-4
    cd = chamfer_l2(S3, S4, n=8000, res=64)
    assert abs(cd - 2 * 0.1**2) < 1e-3  # 0.01 each way


def test_chamfer_no_surface_is_inf():
    assert chamfer_l2(lambda x: x.norm(dim=-1) + 1.0, S4, n=100, res=16) == float("inf")


def test_metrics_accept_learned_callable():
    net = torch.nn.Linear(3, 1)
    learned = lambda x: net(x)[:, 0]  # noqa: E731  (a half-space)
    assert 0.0 <= grid_iou(learned, S4, res=16) <= 1.0
