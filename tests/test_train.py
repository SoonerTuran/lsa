"""Tiny end-to-end runs of the phase-1 trainer and test-time optimisation (CPU, seconds)."""
import csv
import json

import torch

from lsa.data.shapes import random_specs
from lsa.infer import clamped_l1, fit_latents, main as infer_main
from lsa.models.decoder import Decoder
from lsa.train.__main__ import main as train_main
from lsa.train.phase1 import pick_interp_pair, pick_viz_indices, run
from lsa.utils import apply_overrides, load_config

TINY = [
    "device=cpu", "data.n_shapes=6", "data.points_per_shape=512", "data.points_per_step=128",
    "model.latent_dim=8", "model.hidden_dim=32", "train.steps=6", "train.shapes_per_step=3",
    "train.lr_decay_every=4", "log.every=2", "log.eval_every=3", "log.ckpt_every=3",
    "log.n_eval_shapes=2", "log.n_viz_shapes=2", "log.grid_res=12", "log.slice_res=16",
    "infer.n_shapes=2", "infer.steps=3", "infer.points_per_step=64", "infer.points_per_shape=256",
    "infer.n_viz=1", "final.n_train_eval=3", "final.n_chamfer=1", "final.chamfer_points=200",
    "final.chamfer_res=16", "final.mesh_res=16", "final.interp_steps=3",
]


def test_overrides_parse_yaml_values():
    cfg = apply_overrides({"train": {"steps": 1}}, ["train.steps=5", "train.amp=false", "data.sigmas=[0.1, 0.2]", "x=cpu"])
    assert cfg == {"train": {"steps": 5, "amp": False}, "data": {"sigmas": [0.1, 0.2]}, "x": "cpu"}


def test_clamped_l1():
    pred = torch.tensor([[0.5, -0.05, 0.0]])
    gt = torch.tensor([[0.3, -0.5, 0.02]])
    assert torch.allclose(clamped_l1(pred, gt, 0.1), torch.tensor([(0.0 + 0.05 + 0.02) / 3]))


def test_pickers():
    specs = random_specs(30, seed=0)
    v = pick_viz_indices(specs, 4)
    assert len(set(v)) == 4 and {specs[i].kind for i in v} <= {"box", "cylinder", "capsule", "torus"}
    a, b = pick_interp_pair(specs)
    assert specs[a].kind == "box" and specs[b].kind == "torus"


def test_phase1_end_to_end_and_resume(tmp_path):
    out = tmp_path / "run"
    train_main(["--config", "configs/phase1_smoke.yaml", "--out", str(out), "--set", *TINY])
    m = json.loads((out / "metrics.json").read_text())
    assert m["step"] == 6 and 0.0 <= m["train"]["iou_mean"] <= 1.0 and m["unseen"]["n"] == 2
    for p in ["checkpoints/last.pt", "config.yaml", "train_log.csv", "eval_log.csv", "data/train.npz",
              "final/per_shape.json", "interp/interp_xz.png"]:
        assert (out / p).exists(), p
    assert list((out / "viz").rglob("*.png")) and list((out / "viz" / "gt").glob("*.ply"))

    # resume: extend the run to 10 steps; logging continues from step 6
    cfg = load_config("configs/phase1_smoke.yaml", TINY + ["train.steps=10"])
    m2 = run(cfg, out)
    assert m2["step"] == 10
    steps = [int(r["step"]) for r in csv.DictReader(open(out / "train_log.csv"))]
    assert steps == [2, 4, 6, 8, 10]
    ckpt = torch.load(out / "checkpoints" / "last.pt", weights_only=False)
    assert ckpt["step"] == 10

    # rerun of a finished run is a no-op returning the stored metrics
    assert run(cfg, out)["step"] == 10

    # standalone inference CLI on the checkpoint
    infer_main(["--ckpt", str(out / "checkpoints" / "last.pt"), "--out", str(tmp_path / "inf"),
                "--device", "cpu", "--n-shapes", "2", "--steps", "2", "--interp"])
    assert json.loads((tmp_path / "inf" / "infer_metrics.json").read_text())["summary"]["n"] == 2


def test_fit_latents_reduces_loss_and_freezes_decoder():
    torch.manual_seed(0)
    dec = Decoder(latent_dim=8, hidden_dim=32)
    with torch.no_grad():  # start inside the +-delta band so the clamped loss has gradients
        dec.out.weight.mul_(0.05)
        dec.out.bias.zero_()
    before = {k: v.clone() for k, v in dec.state_dict().items()}
    specs = random_specs(2, seed=1)
    z0 = fit_latents(dec, specs, steps=0, points_per_step=256, points_per_shape=512)
    z = fit_latents(dec, specs, steps=60, lr=1e-2, points_per_step=256, points_per_shape=512)
    assert z.shape == (2, 8)
    assert all(torch.equal(before[k], v) for k, v in dec.state_dict().items())
    assert all(p.requires_grad for p in dec.parameters())  # flags restored
    from lsa.infer import gt_samples

    x, d = gt_samples(specs, 512, seed=0)
    with torch.no_grad():
        l0 = clamped_l1(dec(z0, x), d, 0.1).mean()
        l1 = clamped_l1(dec(z, x), d, 0.1).mean()
    assert l1 < l0
