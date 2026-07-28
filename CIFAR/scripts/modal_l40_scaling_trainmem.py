"""L40S (Modal): depth/timestep inference scaling + training-memory sweep.

Part 1 — inference time+memory scaling (vit-cifar-test worktree test.py,
warmed loader, bs=128): ViT depth 7/14/21/28 and DiT 7/14/21/28 timesteps
(depth-7 with trained checkpoints, deeper with random weights — timing is
weight-independent).

Part 2 — training peak memory (NeurIPS26 CIFAR profile_memory.py, real
train.py loops, 30 steps, bs=128): ViT / KEP-1/7 / KEP-7/7 / SGPA Stage-1 and
DIRECTOR Stage-2 from each teacher, plus split-backward (PROFILE_LOWMEM)
variants for from-ViT and from-SGPA.

Usage:
    modal run --detach scratchpad/modal_l40_scaling_trainmem.py
"""
from __future__ import annotations

import modal

app = modal.App("transdiff-l40-scaling-trainmem")

SRC_TIMING = "/mnt/disk1/aiotlab/pqhung/TransDiff/TransDiff-vit-cifar-test/CIFAR"
SRC_MAIN = "/mnt/disk1/aiotlab/pqhung/TransDiff/CIFAR"
DATA_SRC = "/mnt/disk1/aiotlab/pqhung/TransDiff/CIFAR/data/CIFAR10"
CKPT_SRC = "/tmp/claude-1002/-mnt-disk1-aiotlab-pqhung-TransDiff/15960f43-3667-4982-8e95-ea1790148216/scratchpad/l40_timing_ckpts"

IGNORE = ["__pycache__", "*.pyc", ".git", "*.pth", "*.pt", "*.npz", "*.tar.gz", "*.tar",
          "wandb", "*/wandb", "results", "*/results", "results_*", "*/results_*",
          "data", "*/data"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1", "torchvision==0.19.1", "timm>=1.0.0", "wandb>=0.18",
        "scikit-learn", "pandas", "einops", "warmup_scheduler", "gpytorch",
        "ema-pytorch", "matplotlib",
    )
    .add_local_dir(SRC_TIMING, remote_path="/root/timing", ignore=IGNORE)
    .add_local_dir(SRC_MAIN, remote_path="/root/main", ignore=IGNORE)
    .add_local_dir(DATA_SRC, remote_path="/root/data/CIFAR10")
    .add_local_dir(CKPT_SRC, remote_path="/root/ckpts")
)

TCOMMON = "--seed 0 --batch-size 128 --gpu 0 --nb-run 1"
DIT = ("--model diffusion --attn-type softmax --num_heads 12 --hdim 384 --nb-epochs 100 --lr 1e-3 "
       "--backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1.0 --trans_dropout 0.1")

SCALING = []
for d in (7, 14, 21, 28):
    sd = "/root/ckpts/vit_out" if d == 7 else f"./results_scaling/vit_out_d{d}"
    prep = "" if d == 7 else f"mkdir -p {sd}/cifar10_softmax_vit_cifar_0 && "
    SCALING.append((f"ViT-d{d}", f"{prep}python3 test.py {TCOMMON} --model vit_cifar --attn-type softmax --depth {d} --save-dir {sd} Cifar10"))
for d in (7, 14, 21, 28):
    sd = "/root/ckpts/diffusion" if d == 7 else f"./results_scaling/diffusion_t{d}"
    prep = "" if d == 7 else f"mkdir -p {sd}/cifar10_softmax_diffusion_0_transformer_1_12_1.0_0.1_0.001_100 && "
    SCALING.append((f"DiT-t{d}", f"{prep}python3 test.py {TCOMMON} {DIT} --depth {d} --save-dir {sd} Cifar10"))

MCOMMON = "--seed 0 --batch-size 128 --gpu 0 --nb-epochs 1 --nb-run 1 --weight-decay 5e-5"
DIFF_VIT = ("--model diffusion --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --lr 1e-3 "
            "--save-dir ./results/diffusion --backbone transformer --trans_depth 1 --trans_num_heads 12 "
            "--trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 "
            "--pretrained_dir /root/ckpts/vit_out --pretrained_seed 0")
DIFF_KEP = ("--model diffusion --depth 7 --attn-type kep_svgp --concate --eta-ksvd 10 --num_heads 12 --hdim 384 "
            "--lr 1e-3 --save-dir ./results/diffusion --backbone transformer --trans_depth 1 --trans_num_heads 12 "
            "--trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 "
            "--pretrained_dir /root/ckpts/vit_out_cat --pretrained_seed 0")
DIFF_SGPA = ("--model diffusion --depth 7 --attn-type sgpa --num_heads 12 --hdim 384 --lr 5e-4 "
             "--save-dir ./results/diffusion_sgpa --backbone mlp --mlp_hdim1 1024 --mlp_hdim2 1024 --mlp_hdim3 1024 "
             "--mlp_dropout 0.1 --clip 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 "
             "--pretrained_dir /root/ckpts/vit_out_sgpa --pretrained_seed 0")

TRAINMEM = [
    ("trainmem ViT", f"python3 profile_memory.py {MCOMMON} --attn-type softmax --model vit_cifar --lr 1e-3 --save-dir ./results/vit_out Cifar10"),
    ("trainmem KEP-1/7", f"python3 profile_memory.py {MCOMMON} --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --model vit_cifar --lr 1e-3 --save-dir ./results/vit_out_cat Cifar10"),
    ("trainmem KEP-7/7", f"python3 profile_memory.py {MCOMMON} --attn-type kep_svgp --concate --ksvd-layers 7 --eta-ksvd 10 --model vit_cifar --lr 1e-3 --save-dir ./results/vit_out_cat Cifar10"),
    ("trainmem SGPA", f"python3 profile_memory.py {MCOMMON} --attn-type sgpa --model vit_cifar --lr 1e-3 --save-dir ./results/vit_out_sgpa Cifar10"),
    ("trainmem DIR<-ViT", f"python3 profile_memory.py {MCOMMON} {DIFF_VIT} Cifar10"),
    ("trainmem DIR<-KEP-1/7", f"python3 profile_memory.py {MCOMMON} --ksvd-layers 1 {DIFF_KEP} Cifar10"),
    ("trainmem DIR<-KEP-7/7", f"python3 profile_memory.py {MCOMMON} --ksvd-layers 7 {DIFF_KEP} Cifar10"),
    ("trainmem DIR<-SGPA", f"python3 profile_memory.py {MCOMMON} {DIFF_SGPA} Cifar10"),
    ("trainmem DIR<-ViT lowmem", f"PROFILE_LOWMEM=1 python3 profile_memory.py {MCOMMON} {DIFF_VIT} Cifar10"),
    ("trainmem DIR<-SGPA lowmem", f"PROFILE_LOWMEM=1 python3 profile_memory.py {MCOMMON} {DIFF_SGPA} Cifar10"),
]


@app.function(image=image, gpu="L40S", timeout=3 * 3600,
              secrets=[modal.Secret.from_name("wandb-api", required_keys=["WANDB_API_KEY"])])
def run_all():
    import os
    import re
    import subprocess

    for root in ("/root/timing", "/root/main"):
        if not os.path.exists(f"{root}/data"):
            os.symlink("/root/data", f"{root}/data")

    summary = []

    os.chdir("/root/timing")
    for name, cmd in SCALING:
        print(f"===== {name} =====", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/root/timing")
        out = r.stdout + r.stderr
        start = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}).*Start inference", out)
        end = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}).*End inference", out)
        mem = re.findall(r"Inference memory: ([\d.]+) GB", out)
        secs = None
        if start and end:
            def t2s(t):
                h, m, s = t.replace(",", ".").split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            secs = round(t2s(end[-1]) - t2s(start[-1]), 3)
        row = f"{name}: time={secs} s  mem={mem[-1] if mem else 'NA'} GB  rc={r.returncode}"
        print(row, flush=True)
        if r.returncode != 0:
            print(out[-2500:], flush=True)
        summary.append(row)

    os.chdir("/root/main")
    for name, cmd in TRAINMEM:
        print(f"===== {name} =====", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/root/main")
        out = r.stdout + r.stderr
        mem = re.findall(r"Training memory: ([\d.]+) GB", out)
        row = f"{name}: train_mem={mem[-1] if mem else 'NA'} GB  rc={r.returncode}"
        print(row, flush=True)
        if r.returncode != 0:
            print(out[-2500:], flush=True)
        summary.append(row)

    print("=" * 20 + " SUMMARY (1x L40S) " + "=" * 20, flush=True)
    for row in summary:
        print(row, flush=True)


@app.local_entrypoint()
def main():
    run_all.remote()
