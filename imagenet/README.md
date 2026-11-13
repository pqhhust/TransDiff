# ViT-L ImageNet: Qwen-style likelihood experiment

New branch: `imagenet-vit-large-loglik`, based on `rebuttal-qwen` at
`490e9788ab772a7817148eb13d9f78f279acbb06`. The existing CIFAR/Qwen code is
unchanged. Run commands below from the **repository root**.

## Experiment

Use `google/vit-large-patch16-224`: 24 blocks, width 1024, 16 attention heads,
ImageNet-21k pretraining and ImageNet-1k fine-tuning. This is the **1000-class
classifier**, not the headless/in21k-only checkpoint. It is closer in total
parameter count to Qwen2.5-0.5B than ViT-B. The shared cluster already has
ImageNet; the checkpoint lives in `~/models/google/vit-large-patch16-224`.

Replace the last three blocks, zero-based `[21,24)`, with three recurrent
stochastic transitions using one **shared three-block DiT**. The two threes
are independent settings: `to_layer - from_layer` is the number of transitions;
`trans_depth` is the depth of their shared backbone. The rest of ViT stays frozen.

Settings carried over from the active command in `rebuttal-qwen/CIFAR/run.sh`:

| Setting | Qwen reference | ImageNet configuration |
|---|---|---|
| DiT depth / MLP ratio / head dropout | 3 / 4 / 0.1 | same |
| Hidden width / attention heads | 896 / 14 | 1024 / 16, from ViT-L |
| Loss weights | NLL 1 + CE 1 | same |
| AdamW LR / weight decay | 5e-5 / 5e-7 | same |
| Epochs / warmup / minimum LR | 20 / 5 / 1e-7 | same |
| Effective batch | 4 GPUs × 128 = 512 | 512 via gradient accumulation |
| Evaluation samples | 1 stochastic pass | 1 by default; optional MC averaging |

The replacement heads follow Qwen: `mean = DiT(x,t) + mean_head(DiT(x,t))`,
`std = exp(var_head(DiT(x,t)))`, then `x_next = mean + std * epsilon`.
The code evaluates DiT **once** per transition. The optimized Gaussian NLL is

```text
mean_over_layers_and_elements(log_std + 0.5 * ((target - mean) * exp(-log_std))²)
loss = lambda_nll * Gaussian_NLL + lambda_ce * classification_cross_entropy
```

This is the Qwen likelihood formula with the same omitted Gaussian constant,
not the older ImageNet MSE/variance matching objective. Negative continuous
NLL values are legitimate. Likelihood arithmetic uses float32, with log standard
deviations clamped to [-10,5] for numerical safety. Classification NLL reported
in validation is a separate quantity, in unscaled natural-log units.

### Teacher boundaries and source issues

The Qwen forward retains one final FFN after the stochastic chain. We preserve
that arrangement using the **last selected ViT block's pretrained FFN**. Targets
are full block outputs for the first two transitions and the pre-FFN attention
residual for the final transition. The retained FFN is applied exactly once,
then the untouched suffix (if any), final normalization, and classifier.
Substituting the teacher's final target reproduces the original teacher logits;
tests cover prefix, interior, and suffix replacement intervals.

This repairs inconsistencies rather than silently reproducing them:

- Qwen's NLL zips the complete `x_t` list (starting at embeddings) with the
  selected student transitions; it never slices next-state targets by
  `from_layer`/`to_layer`. This runner uses explicit matching targets.
- The Qwen forward copies the last model FFN regardless of the selected range.
  Here the FFN belongs to the final selected layer, so interior ranges work too.
- `vit-imagenet1k-single-run` has an attention-path error (`layer.dropout(x)`
  instead of the attention output), and uses MSE. We use HF's real layer forward.
- `vit_image_net_mixed_precision` provides AMP precedent, but unconditional
  bf16 is unsuitable for V100 tensor cores. `auto` uses fp16 on V100, bf16 on
  Ampere or newer, fp32 on CPU.
- `vit-cifar-final` uses the older CIFAR embedding/readout design; it is not
  substituted for the pretrained ImageNet classifier.

Reviewed branch heads: `vit-imagenet1k-single-run` (`fecb2f5`),
`vit_image_net_mixed_precision` (`ec41d77`), `vit-cifar-final` (`a0f93cd`),
and the Qwen base above. Other remote branches remain available locally.

## Refinements after review (2026-11-09)

**Variance-head floor.** The Qwen head is `std = exp(GELU(Linear(LN z)))`; GELU is
bounded below by -0.17, so std >= 0.84 elementwise and the [-10,5] clamp never binds
from below. Measured on ViT-L validation images, each of blocks 21-23 changes the
residual stream by only ~0.33 std per element (stream std ~4.5). Adding N(0, 0.84)
to the final state alone flips ~8% of teacher predictions, compounded over three
transitions. `--var-head linear` uses `log_std = Linear(LN z)` (no GELU/dropout).

**Near-identity init.** `--head-init zero` zero-initializes the mean head's Linear
(so `mean = DiT(x) = x`, the DiT blocks being adaLN-zero identities) and, with the
linear head, sets `log_std = --log-std-init` (default -2, std 0.135). Skipping all
three layers keeps 84% teacher agreement, a much better start than random heads.
The Qwen head with default init remains the default for reference reproduction.

**Pilot (150 updates of 512, no warmup, 5k validation images, 1 MC sample):**

| Head | LR | Top-1 | ECE |
|---|---:|---:|---:|
| baseline ViT-L (same 5k) | - | 82.14 | 4.40 |
| qwen, default init | 2e-4 | 74.68 | 23.97 |
| linear, zero init | 5e-5 | 77.32 | 18.68 |

**Weights & Biases.** `--wandb` logs train losses/LR/throughput and per-epoch
validation metrics; `--wandb-project`, `--wandb-entity` (or `WANDB_ENTITY`),
`--run-name`. The API key is read from `~/.netrc` / `WANDB_API_KEY`, never the code.

**Speed sweep** (`python -m imagenet.bench_speed`, V100S-32GB, synthetic images,
model + optimizer only). `--compile` matches eager to ~1e-6 relative gradient (fp32).

| Microbatch | Checkpointing | Compile | img/s | s / 512 | Peak GiB |
|---:|:-:|:-:|---:|---:|---:|
| 32 | on | off | 141 | 3.62 | 3.4 |
| 32 | off | off | 166 | 3.09 | 5.6 |
| 64 | off | off | 178 | 2.88 | 9.3 |
| 128 | off | off | 182 | 2.82 | 16.5 |
| 64 | off | on | 197 | 2.60 | 8.4 |
| 128 | off | on | 203 | 2.53 | 14.7 |

Recommended: `--compile --no-activation-checkpointing --batch-size 64` (16 GB GPUs) or
`128` (>= 32 GB): 1.4x faster. CPU decoding gives 757 img/s with 4 workers and 947 with 8,
so data loading is not the bottleneck on a V100.

**Throughput.** Real-data training: ~134 img/s on a V100-16GB (3.8 s per 512-image
update), i.e. ~2.6 h per ImageNet pass; 20 full epochs are ~53 GPU-hours. Runs here
use `--steps-per-epoch` to compress the schedule. Note: some cluster GPUs are shared;
two 16 GB pilots OOM'd beside another process holding 12 GB.

## Memory and execution strategy

`--split-backward` computes/backpropagates NLL first, releases that graph, then
recomputes/backpropagates CE, with **one optimizer step** after both. CPU/CUDA
RNG replay reproduces the same noise/dropout as a joint forward. Both paths
reuse the detached teacher prefix. Tests compare every trainable gradient and
the final RNG state, including nonzero dropout and activation checkpointing.
Use `--no-split-backward` for a joint-loss reference or if recomputation is slower.
Joint backward is the default: local benchmarks found that splitting duplicates
most of the graph and retains the first loss's parameter gradients during the
second forward, increasing both peak memory and runtime in this architecture.

Selected default: **joint backward + activation checkpointing, batch 32, effective
batch 512** (16 accumulation microbatches on one GPU). A three-iteration synthetic
benchmark on the allocated Tesla V100-PCIE-16GB, fp16, produced:

| Execution | Peak allocated GPU memory | Seconds / batch of 32 |
|---|---:|---:|
| Joint, no checkpointing | 5.64 GiB | 0.203 |
| **Joint, checkpointing** | **3.45 GiB** | **0.237** |
| Split, no checkpointing | 5.59 GiB | 0.317 |
| Split, checkpointing | 3.50 GiB | 0.384 |

Checkpointing saves about 39% peak memory for about 17% additional model runtime
relative to the fastest joint variant. This leaves headroom for real-data
training. Timings exclude image decoding and filesystem I/O; short benchmark
timings are indicative, not an end-to-end throughput guarantee. Both joint and
split alternatives remain flags for other hardware. FP32 unit tests verify
matching gradients; fp16 introduces normal small rounding differences.

- One frozen teacher instance; no second copy of the prefix/readout.
- Teacher prefix and selected targets run under `no_grad`; only three targets
  are retained. Gradients still propagate through the frozen readout to the student.
- SDPA attention; optional non-reentrant activation checkpointing of the shared
  DiT (`--no-activation-checkpointing` trades memory for speed).
- Native AMP plus fp16 gradient scaling; fused AdamW on CUDA.
- `zero_grad(set_to_none=True)`; DDP synchronization only at the final backward
  of an accumulated optimizer update.
- Direct, read-only Parquet batches. No 155 GB extraction or HF Arrow cache copy.
- Streaming metrics and MC probability accumulation; no full prediction tensor.
- Checkpoints contain student weights/optimizer state, not another ViT checkpoint.

`python -m imagenet.benchmark` compares the four joint/split and checkpointing
combinations on synthetic images, including optimizer memory. It does not
benchmark disk I/O or claim downstream model accuracy.

## Checkpoint and environment

The large checkpoint is pinned to revision
`9e2727f4250d3973839eecfa5c4b42e41b709a50` (about 1.2 GB).
Download only one weight format:

```bash
python -m imagenet.download_checkpoint
```

The download script records revision and SHA-256 in `provenance.json`.
`imagenet/requirements.txt` records the cluster versions used for validation.
Use the existing `pqhung` environment on this account; no new environment or
duplicate CUDA install is needed. For another machine, install a compatible
PyTorch CUDA build and those dependencies. Keep `HF_HOME`/`HF_HUB_CACHE` writable
and **do not** point them at the shared read-only directory.

## Run

```bash
# Tests: small random ViT, no checkpoint download or GPU needed.
python -m pytest imagenet/tests -q

# Original pretrained classifier; no student training. First use a short check.
python -m imagenet.run --baseline --val-limit 128 --output results/vit-large-baseline-check

# Real-data training smoke check; deliberately not a scientific result.
python -m imagenet.run --config imagenet/configs/vit_large_last3.json \
  --epochs 1 --warmup-epochs 0 --steps-per-epoch 2 --val-limit 8 \
  --batch-size 2 --effective-batch-size 4 --workers 0 \
  --output results/vit-large-smoke

# Full 20-epoch experiment on an already allocated GPU.
python -m imagenet.run --config imagenet/configs/vit_large_last3.json

# Or submit from repository root (activate your environment first).
sbatch imagenet/submit.slurm

# Same global batch of 512 across four GPUs; choose a partition you can access.
NPROC=4 sbatch --gres=gpu:4 imagenet/submit.slurm

# Resume at the next epoch with the original settings/topology.
python -m imagenet.run --config imagenet/configs/vit_large_last3.json \
  --resume results/imagenet-vit-large-last3/last.pt

# Final predictive metrics using ten stochastic samples.
python -m imagenet.run --config imagenet/configs/vit_large_last3.json \
  --evaluate --resume results/imagenet-vit-large-last3/best.pt --mc-samples 10

# Full teacher baseline on the same 50,000 labeled validation images.
python -m imagenet.run --baseline --output results/vit-large-baseline
```

Change `--batch-size` for available GPU memory; it must divide 512/world-size.
The runner computes accumulation automatically. All layer ranges are zero-based,
exclusive on the right. Config files use underscores; CLI flags use hyphens.
Outputs are `manifest.json`, `metrics.jsonl`, `last.pt`, and `best.pt`; evaluation
produces `evaluation.json` or `baseline.json`. Checkpoints are atomic and resume
at epoch boundaries. A 24-hour Slurm limit is not a promise that 20 ImageNet
epochs will fit; benchmark throughput and resume as necessary.

Each training epoch shuffles shards and the examples within bounded 32-image
read batches. Workers stream assigned shards repeatedly until the configured
optimizer-step budget is reached. Distributed shard lengths are unequal, so this
is an **approximate pass**, with some repeated/omitted images per epoch; it is
not a global sample permutation. Validation visits each shard once, without
padding duplicates. `--steps-per-epoch` is an explicit pilot/compute-budget knob;
zero means `ceil(training_rows / effective_batch_size)` optimizer updates.

Only training shards are used for fitting. All 50,000 labeled validation images
are used by default; ImageNet's unlabeled test split is never used. Best-epoch
selection uses validation top-1, so final metrics are validation metrics, not an
independent held-out test claim. Report top-1/top-5, ECE (15 bins), predictive
NLL, Brier score, entropy, seed, and MC count. Multiple seeds can be launched with
distinct `--seed` and `--output` values. AUROC/AURC from the old code are
computed from only confidence and correctness (two scalars per image), using
the same ranking functions as the Qwen/CIFAR code. These metrics, Brier, and NLL
are saved unscaled; accuracy and ECE are percentages. Old printed tables multiply
AURC/EAURC by 1000, NLL by 10, and Brier by 100. Undefined binary-ranking metrics
(all predictions correct or all wrong in a tiny smoke test) are saved as null.

Sources: [ViT-L checkpoint](https://huggingface.co/google/vit-large-patch16-224),
[Qwen reference](https://github.com/pqhhust/TransDiff/tree/rebuttal-qwen/CIFAR).
