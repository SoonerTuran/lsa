# Experiments

Each entry: config, steps, wall time, final metrics, artifact paths. IoU is the grid
occupancy IoU at the config's `log.grid_res` over [-1,1]^3.

## 2026-10-20 — phase1_smoke (CPU)
- Command: `bash scripts/smoke_test.sh runs/smoke`
  (`python -m lsa.train --config configs/phase1_smoke.yaml --out runs/smoke --set device=cpu`)
- Machine: Intel Xeon E-2224 (4 cores), torch 2.14.1+cpu
- Config: K=50, latent 32, 8x128 MLP, 400 steps, B=16 x 1024 pts, grid 32^3
- Wall time: 93 s for the training run (59 s training, 14 s final eval, rest data + viz);
  pytest 105 passed in 17 s
- Metrics (sanity only; smoke sizes are far too small for the acceptance targets):
  - train IoU@32^3 (16 shapes): 0.653 (sphere 0.91, capsule 0.78, cylinder 0.56, box 0.44, torus 0.36)
  - eval-subset IoU during training: 0.718 (step 200) -> 0.730 (step 400)
  - unseen IoU@32^3 (4 shapes, 100 fit steps): 0.743
  - train chamfer-L2 (2 shapes): 0.0138
- Artifacts: `runs/smoke/{metrics.json,train_log.csv,eval_log.csv}`,
  `runs/smoke/viz/step_000400/*.{png,ply}`, `runs/smoke/final/{train,unseen}/*`,
  `runs/smoke/interp/*`
- Repeat runs give identical metrics (deterministic on CPU).

## phase1 (full, GPU) — pending
To be run on Colab through the `colab-mcp` server (configured in `.mcp.json`; loads
at the start of a new Claude session) or manually with the README's 3-cell snippet.
Acceptance targets: train IoU@64^3 > 0.95, unseen IoU@64^3 > 0.90.
