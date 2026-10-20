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
- **Sim(3) action / composition**: `T(x) = s R x + t`; `A.compose(B)` = A∘B (B applied
  first). Uniform rotations from normalised Gaussian quaternions.
- **6D rotation**: Zhou et al. Gram-Schmidt; the two 3-vectors become the first two
  *columns* of R (`matrix_to_rot6d` returns those columns).
- `transform_sdf` duck-types T (anything with `R,t,s`, or a tuple) so `lsa/sdf` does not
  depend on `lsa/geom`.
- **Canonical scale**: every shape is rescaled so its analytic bounding radius is exactly
  0.4 (scale variation belongs to the Sim(3) pose, not the latent). Consequence: the
  sphere family is a single shape (r = 0.4); per-family metrics are reported so this is
  visible.
- **Primitive params**: box = half extents; cylinder `(r, h)` and capsule `(r, h)` use
  half heights (capsule segment from (0,-h,0) to (0,h,0)); torus `(R, r)` lies in the
  xz-plane (axis y).
- **Raw aspect ranges** (before normalisation): box half extents U[0.35,1]^3; cylinder
  r,h ~ U[0.35,1]; capsule r ~ U[0.35,1], h ~ U[0,1]; torus R=1, r ~ U[0.3,0.6].
  Thinnest feature after normalisation >= 0.09 (~3 voxels at 64^3 over [-1,1]^3).
  Families are sampled uniformly.
- **Near-surface noise**: near-surface points alternate between sigma 0.005 and 0.03
  (isotropic 3D Gaussian), i.e. 30% / 30% of all samples. Projecting *uniform* cube points
  piles many samples on box edges/corners (inside fraction: box ~13%, sphere ~30%); kept
  as specified.
- **Sample storage**: points/sdf stored float32; GT is the exact SDF evaluated in float64
  at the stored float32 points. Shape i's samples use rng seed `(seed, i)`. Training cache
  lives at `<out>/data/train.npz` and is rebuilt if specs/M/seed/sampling params differ.
- **Metrics**: IoU = occupancy {sdf<0} IoU on a 64^3 grid over the world cube [-1,1]^3
  (1.0 if both empty). MAE = mean |pred-gt| over grid points with |gt| < delta.
  Chamfer-L2 = mean_a min_b |a-b|^2 + mean_b min_a |a-b|^2 (sum, as in DeepSDF) between
  area-uniform samples of marching-cubes surfaces of both SDFs (same procedure for
  analytic and learned SDFs); inf if a surface is missing.
- **Decoder layout** (DeepSDF): 8 hidden ReLU layers of width 512 + linear output (9
  linear layers). Hidden layer 4 receives concat([h, z, x]); layer 3 outputs
  `512 - (latent_dim+3)` features so the concat is 512 wide. Weight norm
  (`torch.nn.utils.parametrizations.weight_norm`) on hidden layers only; no dropout, no
  output tanh. Hidden width is configurable (smoke config uses 128 to fit the 2-min CPU
  budget).
- **Training data**: `points_per_shape` (32768 full / 4096 smoke) precomputed GT samples
  per shape; each step draws a fresh random subset without replacement (`randperm`).
  Loss = mean over shapes of clamped L1 + `latent_reg * mean_b ||z_b||^2`.
- **LR schedule**: StepLR (x0.5 every `lr_decay_every` steps) on both param groups.
  Full run: 30k steps, decay every 10k (~960 epochs over K=2000 at B=64).
- **AMP**: autocast on CUDA only — bf16 on sm80+ (A100), fp16 + GradScaler on T4;
  TF32 matmuls enabled. Loss and all evaluation run in fp32.
- **Evaluation during training**: IoU / MAE on the first `n_eval_shapes` training shapes;
  slices + PLYs for 4 fixed shapes = first box, cylinder, capsule, torus.
  Final: first 200 training shapes (+ chamfer on the first 20), 50 unseen shapes.
- **Test-time optimisation**: z ~ N(0, 0.01^2), Adam lr 1e-2 for 800 steps with a x0.1
  decay at step 400 (as DeepSDF), 8192 points/step drawn *with* replacement from 32768
  exact samples per shape, same clamped L1 + latent reg. All 50 shapes are optimised in one
  Adam with the SUM of per-shape losses (Adam is elementwise -> identical to 50 separate
  runs). Unseen shapes use seed `seed + 100000`; their spheres coincide with the training
  sphere (degenerate family).
- **Clamp caveat**: clamping the prediction zeroes its gradient once |pred| > delta, even
  with the wrong sign. The default-initialised decoder outputs ~0.01-0.04 everywhere, so
  training starts inside the band; kept as specified.
- **Checkpoints**: `<out>/checkpoints/last.pt` (model, latents, optimiser, scheduler,
  scaler, RNG, step, accumulated train time), written atomically every `ckpt_every`
  steps. Re-running resumes; when training is already finished and `metrics.json` exists
  for that step, the run is a no-op. Exact bitwise resume is not guaranteed (the
  DataLoader shuffle order restarts).
- **Interpolation demo**: first box -> first torus in the training set, 8 evenly spaced
  alphas, PLYs + xz / xy slice strips in `<out>/interp/`.
- **Colab**: Google's `colab-mcp` server is configured in `.mcp.json` (loads at session
  start); `colab/run_colab.py` is the in-notebook launcher.
