# ImageNet ViT-L/16 TransDiff experiments (status 2026-11-12)

Student: 1 shared DiT block replaces ViT-L layers 22-24 (`--from-layer 21 --to-layer 24`), linear zero-init heads,
batch 512, lr 2e-4, cosine, no warm-up, 2xA100 bf16. 297.2M deployed params vs 304.3M original.
All numbers: ImageNet val, 50k images.

## Results so far (one pass unless noted)

| Model | Inference | Top-1 | NLL | Brier | ECE % |
|---|---|---:|---:|---:|---:|
| Original ViT-L | - | 81.97 | 0.6986 | 0.2612 | 4.58 |
| 1-block, 3 ep (seed 0) | 1 noisy sample | 81.55 | 0.7048 | 0.2625 | 2.11 |
| 1-block, 3 ep (seed 0) | noise off (`--eval-mean`) | 81.85 | 0.6976 | 0.2603 | 3.01 |
| 1-block, 3 ep (seed 0) | 10 samples averaged | 81.83 | 0.6950 | 0.2596 | 1.78 |
| 1-block, 5 ep, CE 0.25 | 10 samples averaged | 81.74 | 0.6945 | 0.2604 | 1.70 |
| 3-block, 3 ep (335.0M) | 1 noisy sample | 81.78 | 0.6968 | 0.2601 | 2.95 |
| CE on noise-free pass, CE 1.0 | noise off | 81.63 | 0.7017 | 0.2617 | 2.22 |
| CE on noise-free pass, CE 0.5 | noise off | 81.63 | 0.7038 | 0.2627 | 2.34 |
| SwiGLU MLP | 1 noisy sample | 81.44 | 0.7105 | 0.2636 | 2.73 |
| SwiGLU MLP | noise off | 81.66 | 0.7073 | 0.2625 | 3.73 |

Findings:
- Noise off (`out = mean` at each step) beats the original on NLL/Brier/ECE in one pass, but only by ~0.001 NLL
  (seed-to-seed spread is ~0.002), and ECE rises to 3.0 because the classifier was trained on noisy features.
- Training CE on the noise-free pass (`--ce-mean`) restores ECE ~2.2 but loses the NLL win.
- ECE tracks label fitting: it stays near 2 while train CE stays above the teacher's 0.518.
- SwiGLU fits the teacher better but overfits labels faster (worse NLL/ECE).

## Still running / pending

| Run | Change | Status at hand-off |
|---|---|---|
| `ep3-dit1-lnll0.5-...` | NLL weight 0.5 | epoch 2: 81.47 / NLL 0.7120 / ECE 2.10 (ahead of seed 0 by 0.002) |
| `ep3-dit1-seed1-...` | seed 1 | epoch 1: 80.70 / NLL 0.7429 (seed 0: 80.89 / 0.7412) |
| `ep3-dit1-seed2-...` | seed 2 | training |
| `ep3-dit1-mlp6-...` | MLP ratio 6 (301.4M) | queued |

To do: evaluate seeds 1/2 with `--eval-mean` to check whether the noise-off win holds across seeds.

## New switches (default off; old runs reproduce exactly)
- `--eval-mean`: validate with one noise-free pass through the transition means.
- `--ce-mean`: train CE on the noise-free pass; NLL stays on the sampled pass (test in `tests/test_experiment.py`).
- `submit.slurm`: per-job torch.compile cache. Note: nodes node03, node54, node55 segfault at startup; submit with
  `--exclude=node03,node54,node55`.

Example:
```
NPROC=2 sbatch -p A100 --exclude=node03,node54,node55 --gres=gpu:2 --cpus-per-task=8 imagenet/submit.slurm \
  --trans-depth 1 --var-head linear --head-init zero --epochs 3 --lr 2e-4 --warmup-epochs 0 \
  --batch-size 128 --effective-batch-size 512 --compile --output results/NAME --run-name NAME
```
