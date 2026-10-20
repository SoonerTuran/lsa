import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lsa.data.sampling import (
    SDFSamples,
    load_or_precompute,
    load_samples,
    precompute,
    project_to_surface,
    sample_points,
)
from lsa.data.shapes import FAMILIES, random_spec, random_specs, sdf_of


@pytest.mark.parametrize("kind", FAMILIES)
def test_projection_lands_on_surface(kind):
    spec = random_spec(np.random.default_rng(0), kinds=[kind])
    fn = sdf_of(spec)
    x0 = torch.rand(2000, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 2 - 1
    x = project_to_surface(fn, x0, 5)
    assert fn(x).abs().max() < 1e-9


@pytest.mark.parametrize("kind", FAMILIES)
def test_sampled_gt_matches_reevaluation(kind):
    spec = random_spec(np.random.default_rng(1), kinds=[kind])
    x, d = sample_points(spec, 5000, np.random.default_rng(2))
    assert x.shape == (5000, 3) and d.shape == (5000,)
    assert x.dtype == torch.float32 and d.dtype == torch.float32
    assert torch.equal(d, sdf_of(spec)(x.double()).float())
    assert torch.allclose(d, sdf_of(spec)(x), atol=1e-6)


def test_sample_distribution():
    spec = random_spec(np.random.default_rng(3), kinds=["box"])
    x, d = sample_points(spec, 20000, np.random.default_rng(4))
    assert x.abs().max() <= 1.0 + 0.2  # uniform part in the cube, near part close to shape
    near = (d.abs() < 0.02).float().mean().item()
    assert 0.35 < near < 0.6  # ~30% (sigma .005) + ~15% (sigma .03) + few uniform
    # projection piles outside points onto edges/corners, where isotropic noise mostly
    # lands outside, so boxes get the fewest inside samples (~13%; spheres ~30%)
    assert (d < 0).float().mean() > 0.1


def test_sampling_is_seeded():
    spec = random_spec(np.random.default_rng(5))
    a = sample_points(spec, 1000, np.random.default_rng(6))
    b = sample_points(spec, 1000, np.random.default_rng(6))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_precompute_roundtrip_and_dataset(tmp_path):
    specs = random_specs(6, seed=0)
    path = tmp_path / "cache" / "train.npz"
    precompute(specs, 512, path, seed=0)
    data = load_samples(path)
    assert data["points"].shape == (6, 512, 3) and data["sdf"].shape == (6, 512)
    assert data["specs"] == specs and data["meta"]["M"] == 512
    for i in (0, 5):
        re = sdf_of(specs[i])(torch.from_numpy(data["points"][i]).double()).float().numpy()
        assert np.array_equal(re, data["sdf"][i])

    ds = SDFSamples(data, n_points=100)
    assert len(ds) == 6
    i, pts, sdf = ds[3]
    assert i == 3 and pts.shape == (100, 3) and sdf.shape == (100,)
    stored = torch.from_numpy(data["points"][3])
    # subset without replacement of the stored rows, with matching sdf
    match = (pts[:, None, :] == stored[None]).all(-1)
    assert (match.sum(1) == 1).all()
    rows = match.float().argmax(1)
    assert rows.unique().numel() == 100
    assert torch.equal(sdf, torch.from_numpy(data["sdf"][3])[rows])

    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
    assert batch[1].shape == (4, 100, 3) and batch[2].shape == (4, 100)


def test_load_or_precompute_reuses_and_invalidates(tmp_path):
    specs = random_specs(3, seed=1)
    path = tmp_path / "c.npz"
    load_or_precompute(specs, 256, path, seed=0)
    mtime = path.stat().st_mtime_ns
    load_or_precompute(specs, 256, path, seed=0)
    assert path.stat().st_mtime_ns == mtime  # reused
    data = load_or_precompute(specs, 128, path, seed=0)  # M changed -> rebuilt
    assert data["points"].shape == (3, 128, 3)
