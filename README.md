# lsa — Latent Scene Algebra

DeepSDF-style auto-decoder trained on **analytic** SDFs (sphere, box, cylinder, capsule,
torus), so ground truth is exact everywhere — the base for learned latent operators
(pose, union, deformation) in later phases. See `CLAUDE.md` for conventions/decisions and
`EXPERIMENTS.md` for results.

## Install & test (local, CPU is fine)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pytest -q
bash scripts/smoke_test.sh                            # pytest + <=2 min CPU training run
```

## Train / evaluate
```bash
python -m lsa.train --config configs/phase1_smoke.yaml --out runs/smoke
python -m lsa.train --config configs/phase1.yaml --out runs/phase1          # GPU
python -m lsa.train --config configs/phase1.yaml --out runs/p1 --set train.steps=5000 model.latent_dim=64
python -m lsa.infer --ckpt runs/phase1/checkpoints/last.pt --out runs/phase1/infer   # unseen shapes
```
Re-running with the same `--out` resumes from `<out>/checkpoints/last.pt`. Outputs:
`config.yaml`, `train_log.csv`, `viz/step_XXXXXX/*.{png,ply}`, `final/` (train/unseen
meshes + slices), `interp/` (latent interpolation), `metrics.json`.

## Google Colab (T4 / A100)
Three cells (Runtime → Change runtime type → GPU). Cell 3 mounts Drive, installs the
package, trains with the full config into `MyDrive/lsa_runs/phase1`, and resumes
automatically after a disconnect — just re-run the cells.

```python
# Cell 1 — get / update the code
!test -d /content/lsa/.git && git -C /content/lsa pull --ff-only || git clone https://github.com/SoonerTuran/lsa.git /content/lsa
```
```python
# Cell 2 — check the GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
```python
# Cell 3 — mount Drive, pip install -e, train (or resume) with the full config
%run /content/lsa/colab/run_colab.py --config configs/phase1.yaml
```
Private repo: add a Colab secret `GITHUB_TOKEN` and replace Cell 1 by
`%run` of a copy of `colab/run_colab.py` (it clones with the token). Overrides:
`%run /content/lsa/colab/run_colab.py --config configs/phase1.yaml --set train.steps=30000`.
