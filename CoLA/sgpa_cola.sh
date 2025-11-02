#!/bin/bash

# SGPA Training and Testing for CoLA Dataset
# This script provides comprehensive SGPA support for the CoLA task

################ SGPA Base Model Training ################
echo "Training SGPA base model..."
python3 main.py \
    --depth 5 \
    --attn-type sgpa \
    --batch-size 32 \
    --gpu 0 \
    --nb-epochs 100 \
    --nb-run 1 \
    --model vit_cola \
    --lr 5e-4 \
    --seed 0 \
    --save-dir ./results/vit_out_sgpa

################ SGPA Base Model Testing ################
echo "Testing SGPA base model..."
python3 test.py \
    --attn-type sgpa \
    --batch-size 32 \
    --gpu 0 \
    --nb-run 1 \
    --model vit_cola \
    --seed 0 \
    --save-dir ./results/vit_out_sgpa

################ SGPA Diffusion Model Training ################
echo "Training SGPA diffusion model..."
python3 main.py \
    --model diffusion \
    --seed 0 \
    --depth 5 \
    --attn-type sgpa \
    --batch-size 32 \
    --gpu 0 \
    --nb-epochs 100 \
    --nb-run 1 \
    --lr 5e-4 \
    --weight-decay 5e-5 \
    --save-dir ./results/diffusion \
    --backbone transformer \
    --pretrained_dir ./results/vit_out_sgpa \
    --pretrained_seed 0 \
    --trans_depth 1 \
    --trans_num_heads 4 \
    --trans_mlp_ratio 1 \
    --trans_dropout 0.1 \
    --lambda_mean 0.5 \
    --lambda_var 0.2 \
    --lambda_ce 0.3

################ SGPA Diffusion Model Testing ################
echo "Testing SGPA diffusion model..."
python3 test.py \
    --model diffusion \
    --seed 0 \
    --depth 5 \
    --attn-type sgpa \
    --batch-size 32 \
    --gpu 0 \
    --nb-run 1 \
    --save-dir ./results/diffusion \
    --backbone transformer \
    --trans_depth 1 \
    --trans_num_heads 4 \
    --trans_mlp_ratio 1 \
    --trans_dropout 0.1

echo "SGPA experiments completed!"