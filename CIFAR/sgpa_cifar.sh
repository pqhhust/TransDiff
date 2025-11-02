#!/bin/bash

### SGPA Training for CIFAR-10
python3 main.py \
--seed 0 \
--attn-type sgpa \
--batch-size 128 \
--gpu 0 \
--nb-epochs 300 \
--nb-run 1 \
--model vit_cifar \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ./results/vit_out_sgpa \
Cifar10

### SGPA Diffusion Training (matching with SGPA base model)
python3 main.py \
--model diffusion \
--seed 0 \
--depth 7 \
--attn-type sgpa \
--num_heads 12 \
--hdim 384 \
--batch-size 128 \
--gpu 0 \
--nb-epochs 100 \
--nb-run 1 \
--lr 5e-4 \
--weight-decay 5e-5 \
--save-dir ./results/diffusion \
--backbone mlp \
--pretrained_dir ./results/vit_out_sgpa \
--clip 0.1 \
--mlp_hdim1 1024 \
--mlp_hdim2 1024 \
--mlp_hdim3 1024 \
--pretrained_seed 0 \
--mlp_dropout 0.1 \
--lambda_mean 0.4 \
--lambda_var 0.2 \
--lambda_ce 0.4 \
Cifar10

### SGPA Testing
python3 test.py \
--seed 0 \
--attn-type sgpa \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out_sgpa \
Cifar10

### SGPA Diffusion Testing
python3 test.py \
--model diffusion \
--seed 0 \
--depth 7 \
--attn-type sgpa \
--num_heads 12 \
--hdim 384 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--save-dir ./results/diffusion \
--backbone mlp \
--mlp_hdim1 1024 \
--mlp_hdim2 1024 \
--mlp_hdim3 1024 \
--mlp_dropout 0.1 \
Cifar10