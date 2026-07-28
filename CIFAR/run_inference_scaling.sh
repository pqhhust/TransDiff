#!/bin/bash
# Inference time + peak-memory scaling experiment (NeurIPS26 rebuttal).
#
# Runs all five methods at depths 7/14/21/28 with the instrumented test.py
# (warmed 16-worker loader, untimed warm-up pass, then a timed pass over the
# 10k CIFAR-10 test set; logs "Inference time: X s" and
# "Inference memory: X GB" per run; N=1 everywhere):
#
#   ViT       (softmax)                     depth d
#   KEP-1/d   (kep_svgp, GP on last block)  depth d
#   KEP-d/d   (kep_svgp, all blocks GP)     depth d
#   SGPA      (sgpa)                        depth d
#   DIRECTOR  (diffusion, transformer DiT)  d timesteps
#
# Checkpoints: depth-7 rows load best_acc_net_1.pth from $CKPT_ROOT if
# present; any missing checkpoint (all deeper rows) falls back to RANDOM
# weights automatically — timing and peak memory are weight-independent for
# these dense models, so deeper rows need no training.
#
# Usage (from CIFAR/, with ./data/CIFAR10/{train,val,test} prepared):
#   GPU=0 CKPT_ROOT=./results bash run_inference_scaling.sh
#
#   CKPT_ROOT layout (optional, for trained depth-7 rows):
#     $CKPT_ROOT/vit_out/cifar10_softmax_vit_cifar_0/best_acc_net_1.pth
#     $CKPT_ROOT/vit_out_cat/cifar10_kep_svgp_vit_cifar_ksvdlayer{1,7}_ksvd10.0_kl1.0_0/best_acc_net_1.pth
#     $CKPT_ROOT/vit_out_sgpa/cifar10_sgpa_vit_cifar_0/best_acc_net_1.pth
#     $CKPT_ROOT/diffusion/cifar10_softmax_diffusion_0_transformer_1_12_1.0_0.1_0.001_100/best_acc_net_1_diffusion_transformer.pth
#
# Results: per-run logs under ./results_scaling/, summary CSV at
# ./results_scaling/scaling_summary.csv. Set WANDB_MODE=online to sync runs.

set -u
GPU=${GPU:-0}
CKPT_ROOT=${CKPT_ROOT:-./results}
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU

DEPTHS=(7 14 21 28)
OUT=./results_scaling
SUMMARY=$OUT/scaling_summary.csv
mkdir -p $OUT
echo "method,depth,time_s,mem_GB,weights" > $SUMMARY

COMMON="--seed 0 --batch-size 128 --gpu 0 --nb-run 1"

run_one () {  # $1 label  $2 save-dir  $3 save_path-subdir  $4... test.py args
    local label=$1 sd=$2 sub=$3; shift 3
    mkdir -p "$sd/$sub"
    local log=$OUT/$(echo "$label" | tr '/' '_').log
    echo "===== $label ====="
    python3 test.py $COMMON "$@" --save-dir "$sd" Cifar10 > "$log" 2>&1
    local t m w
    t=$(grep -a "Inference time:" "$log" | tail -1 | grep -oE "[0-9.]+ s" | grep -oE "[0-9.]+")
    m=$(grep -a "Inference memory:" "$log" | tail -1 | grep -oE "[0-9.]+ GB" | grep -oE "[0-9.]+")
    w=$(grep -aq "RANDOM weights" "$log" && echo random || echo trained)
    echo "$label: time=${t:-NA} s  mem=${m:-NA} GB  ($w)"
    echo "$label,${label##*-},${t:-NA},${m:-NA},$w" >> $SUMMARY
}

for d in "${DEPTHS[@]}"; do
    if [ "$d" = 7 ]; then sd=$CKPT_ROOT/vit_out; else sd=$OUT/vit_d$d; fi
    run_one "ViT-$d" "$sd" cifar10_softmax_vit_cifar_0 \
        --model vit_cifar --attn-type softmax --depth $d
done

for d in "${DEPTHS[@]}"; do
    if [ "$d" = 7 ]; then sd=$CKPT_ROOT/vit_out_cat; else sd=$OUT/kep1_d$d; fi
    run_one "KEP-1of-$d" "$sd" cifar10_kep_svgp_vit_cifar_ksvdlayer1_ksvd10.0_kl1.0_0 \
        --model vit_cifar --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --depth $d
done

for d in "${DEPTHS[@]}"; do
    if [ "$d" = 7 ]; then sd=$CKPT_ROOT/vit_out_cat; else sd=$OUT/kepall_d$d; fi
    run_one "KEP-allof-$d" "$sd" cifar10_kep_svgp_vit_cifar_ksvdlayer${d}_ksvd10.0_kl1.0_0 \
        --model vit_cifar --attn-type kep_svgp --concate --ksvd-layers $d --eta-ksvd 10 --depth $d
done

for d in "${DEPTHS[@]}"; do
    if [ "$d" = 7 ]; then sd=$CKPT_ROOT/vit_out_sgpa; else sd=$OUT/sgpa_d$d; fi
    run_one "SGPA-$d" "$sd" cifar10_sgpa_vit_cifar_0 \
        --model vit_cifar --attn-type sgpa --depth $d
done

for d in "${DEPTHS[@]}"; do
    if [ "$d" = 7 ]; then sd=$CKPT_ROOT/diffusion; else sd=$OUT/dit_t$d; fi
    run_one "DIRECTOR-$d" "$sd" cifar10_softmax_diffusion_0_transformer_1_12_1.0_0.1_0.001_100 \
        --model diffusion --attn-type softmax --depth $d --num_heads 12 --hdim 384 \
        --nb-epochs 100 --lr 1e-3 --backbone transformer --trans_depth 1 \
        --trans_num_heads 12 --trans_mlp_ratio 1.0 --trans_dropout 0.1
done

echo
echo "==================== SUMMARY ===================="
column -s, -t < $SUMMARY
echo "CSV: $SUMMARY"
