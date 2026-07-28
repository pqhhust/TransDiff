"""Inference time + memory on 1x L40S (Modal) for the tab:infer-time-mem methods.

Runs the vit-cifar-test branch's timed test.py (warmed 16-worker loader,
inference_time_seconds + inference_memory_gb, bs=128) for KEP-7/7, KEP-1/7,
SGPA, ViT, DIRECTOR — the same methodology used on the local A30, but on an
L40S to be comparable with the paper's original L40 numbers. Results print to
the log and sync to wandb via test.py's built-in logging.

Usage:
    modal run --detach scratchpad/modal_l40_timing.py
"""
from __future__ import annotations

import modal

app = modal.App("transdiff-l40-timing")

SRC = "/mnt/disk1/aiotlab/pqhung/TransDiff/TransDiff-vit-cifar-test/CIFAR"
DATA_SRC = "/mnt/disk1/aiotlab/pqhung/TransDiff/CIFAR/data/CIFAR10"
CKPT_SRC = "/tmp/claude-1002/-mnt-disk1-aiotlab-pqhung-TransDiff/15960f43-3667-4982-8e95-ea1790148216/scratchpad/l40_timing_ckpts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "timm>=1.0.0",
        "wandb>=0.18",
        "scikit-learn",
        "pandas",
        "einops",
        "warmup_scheduler",
        "gpytorch",
        "matplotlib",
    )
    .add_local_dir(
        SRC,
        remote_path="/root/transdiff",
        ignore=["__pycache__", "*.pyc", ".git", "*.pth", "*.pt",
                "wandb", "*/wandb", "results", "*/results", "data/CIFAR10"],
    )
    .add_local_dir(DATA_SRC, remote_path="/root/transdiff/data/CIFAR10")
    .add_local_dir(CKPT_SRC, remote_path="/root/transdiff/ckpts")
)

COMMON = "--seed 0 --batch-size 128 --gpu 0 --nb-run 1"
RUNS = [
    ("KEP-7/7", f"python3 test.py {COMMON} --model vit_cifar --attn-type kep_svgp --concate --ksvd-layers 7 --eta-ksvd 10 --save-dir ./ckpts/vit_out_cat Cifar10"),
    ("KEP-1/7", f"python3 test.py {COMMON} --model vit_cifar --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --save-dir ./ckpts/vit_out_cat Cifar10"),
    ("SGPA", f"python3 test.py {COMMON} --model vit_cifar --attn-type sgpa --save-dir ./ckpts/vit_out_sgpa Cifar10"),
    ("ViT", f"python3 test.py {COMMON} --model vit_cifar --attn-type softmax --save-dir ./ckpts/vit_out Cifar10"),
    ("DIRECTOR", f"python3 test.py {COMMON} --model diffusion --attn-type softmax --depth 7 --num_heads 12 --hdim 384 --nb-epochs 100 --lr 1e-3 --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1.0 --trans_dropout 0.1 --save-dir ./ckpts/diffusion Cifar10"),
]


@app.function(image=image, gpu="L40S", timeout=3600,
              secrets=[modal.Secret.from_name("wandb-api", required_keys=["WANDB_API_KEY"])])
def run_timing():
    import os
    import re
    import subprocess

    os.chdir("/root/transdiff")
    summary = []
    for name, cmd in RUNS:
        print(f"===== {name} =====", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = r.stdout + r.stderr
        times = re.findall(r"Start inference\.\n.*?(\d{2}:\d{2}:\d{2},\d{3})", out)
        mem = re.findall(r"Inference memory: ([\d.]+) GB", out)
        start = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}).*Start inference", out)
        end = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}).*End inference", out)
        secs = None
        if start and end:
            def t2s(t):
                h, m, s = t.replace(",", ".").split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            secs = t2s(end[-1]) - t2s(start[-1])
        row = f"{name}: time={secs if secs is not None else 'NA'} s  mem={mem[-1] if mem else 'NA'} GB  rc={r.returncode}"
        print(row, flush=True)
        if r.returncode != 0:
            print(out[-3000:], flush=True)
        summary.append(row)
    print("=" * 20 + " SUMMARY (1x L40S, bs=128, warmed) " + "=" * 20, flush=True)
    for row in summary:
        print(row, flush=True)


@app.local_entrypoint()
def main():
    run_timing.remote()
