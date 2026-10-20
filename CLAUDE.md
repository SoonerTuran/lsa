# lsa — Latent Scene Algebra

Research codebase (PyTorch). A DeepSDF-style decoder whose latent space will support
LEARNED OPERATORS — pose `g(z,T)`, union `f(z1..zn)`, deformation `h(z,d)` — and later a
graph-network scene-dynamics model. All ground truth comes from ANALYTIC SDFs (sphere,
box, cylinder, capsule, torus), so GT is exact everywhere and transforms/unions are exact.

Phases: **0** infrastructure, **1** DeepSDF auto-decoder baseline (done here).
Operators go in `lsa/models/ops/{pose,union,deform}.py` (not implemented yet).

## Conventions
- Python 3.10+, torch 2.x, numpy, scikit-image (marching cubes), trimesh, pyyaml,
  matplotlib, pytest. Minimal dependencies.
- Canonical frame: shape centered at origin, axis-aligned, upright (height along y),
  scaled so it fits inside a ball of radius 0.4. Query/world domain is the cube [-1,1]^3.
- SDF convention: negative inside, positive outside, exact Euclidean distance for
  primitives. Union = min. Losses clamp SDF to ±delta, delta=0.1 (DeepSDF).
- Everything batched in torch: points (B,N,3) -> sdf (B,N). All SDFs differentiable.
  "SDF callables" used by viz/metrics take (N,3) -> (N,) (analytic or learned alike).
- Fixed seeds; YAML configs; single entry point
  `python -m lsa.train --config configs/<name>.yaml --out <dir>`.
- Must run on CPU with tiny "smoke" configs and on Colab (T4/A100) with full configs.
  No hard-coded paths; `--out` may be a Google Drive path; checkpoint/resume supported.
- Math first, tests first: every file in lsa/sdf, lsa/geom, lsa/data has tests in
  tests/. Run `pytest` before declaring a step done. Commit after each numbered item.

## Repo layout
```
lsa/sdf/{primitives,compose}.py   exact batched SDFs, union, Sim(3)-transformed SDFs
lsa/geom/sim3.py                  Sim(3) group (R,t,s), 6D rotation param
lsa/data/{shapes,sampling}.py     ShapeSpec + random families, GT point sampling, Dataset
lsa/models/{decoder,latent}.py    DeepSDF MLP, per-shape latent table
lsa/models/ops/                   (phase 2+) pose / union / deform operators
lsa/train/phase1.py               auto-decoder training; __main__ dispatches on config.phase
lsa/infer.py                      test-time latent optimisation, unseen-shape eval, interpolation
lsa/eval/metrics.py               grid IoU, SDF MAE in band, chamfer-L2
lsa/viz/mesh.py                   grid eval, marching cubes -> .ply, slice PNGs
configs/ tests/ scripts/smoke_test.sh colab/run_colab.py EXPERIMENTS.md
```

## Commands
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e .[dev]   # (bin/ on Linux)
python -m pytest -q
bash scripts/smoke_test.sh                     # pytest + <=2 min CPU training run
python -m lsa.train --config configs/phase1.yaml --out runs/phase1 [--set train.steps=100]
python -m lsa.infer --ckpt runs/phase1/checkpoints/last.pt --out runs/phase1/infer
```
Re-running `lsa.train` with the same `--out` resumes from `<out>/checkpoints/last.pt`.

## Decisions
(Ambiguities resolved with the simplest option; append new ones here.)
