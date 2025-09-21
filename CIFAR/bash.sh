# python3 main.py \
# --seed 0 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-epochs 0 \
# --nb-run 1 \
# --model vit_cifar \
# --lr 1e-3 \
# --weight-decay 5e-5 \
# --save-dir ./results/vit_out \
# Cifar10 

# python3 main.py \
# --seed 1 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-epochs 300 \
# --nb-run 1 \
# --model vit_cifar \
# --lr 1e-3 \
# --weight-decay 5e-5 \
# --save-dir ./results/vit_out \
# Cifar10

# python3 main.py \
# --seed 2 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-epochs 300 \
# --nb-run 1 \
# --model vit_cifar \
# --lr 1e-3 \
# --weight-decay 5e-5 \
# --save-dir ./results/vit_out \
# Cifar10 

# python3 main.py \
# --seed 3 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-epochs 300 \
# --nb-run 1 \
# --model vit_cifar \
# --lr 1e-3 \
# --weight-decay 5e-5 \
# --save-dir ./results/vit_out \
# Cifar10

# python3 main.py \
# --seed 4 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-epochs 300 \
# --nb-run 1 \
# --model vit_cifar \
# --lr 1e-3 \
# --weight-decay 5e-5 \
# --save-dir ./results/vit_out \
# Cifar10


### Test
# python3 test.py \
# --seed 0 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 1 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 2 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 3 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 4 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10


# ### KEP-SVGP
# python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 

# python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 7 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 7 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 7 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 7 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 7 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 

# python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 2 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 2 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 2 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 2 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 2 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 

# python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 3 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 3 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 3 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 3 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 3 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 

# python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 4 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 4 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 4 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 4 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 4 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 

# python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 5 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 5 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 5 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 5 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 5 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 

# python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 6 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 6 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 6 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 6 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 \
# &python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 6 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10 


### Test
# python3 test.py \
# --seed 0 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 1 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 2 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 3 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10 \
# &python3 test.py \
# --seed 4 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cifar \
# --save-dir ./results/vit_out \
# Cifar10

# python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 0 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 1 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 2 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 3 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 4 Cifar10 

# python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 2 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 0 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 2 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 1 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 2 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 2 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 2 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 3 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 2 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 4 Cifar10 

# python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 0 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 1 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 2 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 3 Cifar10 \
# &python3 test.py --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-run 1 --model vit_cifar --save-dir ./results/vit_out_cat --seed 4 Cifar10 


### Train Diffusion
## 7 layers KEP-SVGP
## dropout 0.1
# python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.1 --mlp_hdim1 1024 --mlp_hdim2 1024 --mlp_hdim3 1024 --mlp_hdim4 64 --pretrained_seed 3 --mlp_dropout 0.1 --mlp_gamma 1.0 Cifar10 

# ## dropout 0
# python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.1 --mlp_hdim1 64 --mlp_hdim2 64 --mlp_hdim3 64 --pretrained_seed 3 --mlp_dropout 0 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.1 --mlp_hdim1 128 --mlp_hdim2 128 --mlp_hdim3 128 --pretrained_seed 3 --mlp_dropout 0 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.1 --mlp_hdim1 256 --mlp_hdim2 256 --mlp_hdim3 256 --pretrained_seed 3 --mlp_dropout 0 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.1 --mlp_hdim1 128 --mlp_hdim2 64 --mlp_hdim3 128 --pretrained_seed 3 --mlp_dropout 0 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.1 --mlp_hdim1 256 --mlp_hdim2 64 --mlp_hdim3 256 --pretrained_seed 3 --mlp_dropout 0 Cifar10

# ## 1 layers KEP-SVGP
# ## 64 64 64
# python3 main.py --model diffusion --seed 0 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 64 --mlp_hdim2 64 --mlp_hdim3 64 --pretrained_seed 0 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 1 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 64 --mlp_hdim2 64 --mlp_hdim3 64 --pretrained_seed 1 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 2 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 64 --mlp_hdim2 64 --mlp_hdim3 64 --pretrained_seed 2 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 64 --mlp_hdim2 64 --mlp_hdim3 64 --pretrained_seed 3 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 4 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 64 --mlp_hdim2 64 --mlp_hdim3 64 --pretrained_seed 4 --mlp_dropout 0.1 Cifar10

# ## 128 128 128
# python3 main.py --model diffusion --seed 0 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 128 --mlp_hdim3 128 --pretrained_seed 0 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 1 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 128 --mlp_hdim3 128 --pretrained_seed 1 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 2 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 128 --mlp_hdim3 128 --pretrained_seed 2 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 128 --mlp_hdim3 128 --pretrained_seed 3 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 4 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 128 --mlp_hdim3 128 --pretrained_seed 4 --mlp_dropout 0.1 Cifar10

# ## 256 256 256
# python3 main.py --model diffusion --seed 0 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 256 --mlp_hdim3 256 --pretrained_seed 0 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 1 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 256 --mlp_hdim3 256 --pretrained_seed 1 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 2 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 256 --mlp_hdim3 256 --pretrained_seed 2 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 256 --mlp_hdim3 256 --pretrained_seed 3 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 4 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 256 --mlp_hdim3 256 --pretrained_seed 4 --mlp_dropout 0.1 Cifar10

# ## 128 64 128
# python3 main.py --model diffusion --seed 0 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 64 --mlp_hdim3 128 --pretrained_seed 0 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 1 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 64 --mlp_hdim3 128 --pretrained_seed 1 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 2 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 64 --mlp_hdim3 128 --pretrained_seed 2 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 64 --mlp_hdim3 128 --pretrained_seed 3 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 4 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 128 --mlp_hdim2 64 --mlp_hdim3 128 --pretrained_seed 4 --mlp_dropout 0.1 Cifar10

# ## 256 64 256
# python3 main.py --model diffusion --seed 0 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 64 --mlp_hdim3 256 --pretrained_seed 0 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 1 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 64 --mlp_hdim3 256 --pretrained_seed 1 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 2 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 64 --mlp_hdim3 256 --pretrained_seed 2 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 3 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 64 --mlp_hdim3 256 --pretrained_seed 3 --mlp_dropout 0.1 Cifar10 \
# &python3 main.py --model diffusion --seed 4 --depth 1 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.01 --mlp_hdim1 256 --mlp_hdim2 64 --mlp_hdim3 256 --pretrained_seed 4 --mlp_dropout 0.1 Cifar10

# python3 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/imagenet/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed ImageNet

# # distributed training
# CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 100 --nb-run 1 --lr 3e-4 --weight-decay 5e-5 --warmup-epoch 0 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed ImageNet

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 32 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 1.0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 32 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 1.0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.2 --lambda_var 0 --lambda_ce 0.8 --run_name DiT-5-seed Cifar10

# merge 2 layers
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 32 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.5 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_augment_224_fit_residual_clip Cifar10
CUDA_VISIBLE_DEVICES=2,7 torchrun --nproc_per_node=2 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.5 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_augment_224_fit_residual_both_mean_and_ce_clip Cifar10


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 36 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_3_sublayers_t_[0,1,..]forinference_[0,0.33,0.66,1,...]fortrain_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 36 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 32 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_3_sublayers_t_[0,1,..]forinference_[0,0.33,0.66,1,...]fortrain_random_select_12_sublayers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 36 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_3_sublayers_t_[0,1,..]forinference_[0,0.33,0.66,1,...]fortrain_from_epoch_5_ce_w_0.8_mean_w_0.2_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 36 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 1.0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_3_sublayers_t_[0,1,..]forinference_[0,0.33,0.66,1,...]fortrain_from_epoch_5_ce_w_1_mean_w_0.2_clipgrad_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 24 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 1.0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_2_sublayers_t_[0,1,..]forinference_[0,0.5,1,...]fortrain_sum_predictions_feeding_intermediate_reps_through_solution_head_augment_224 Cifar10



CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_6_layers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name fix_att_code_merge_into_6_layers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.2 --lambda_var 0 --lambda_ce 0.8 --run_name fix_att_code_merge_into_6_layers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 10 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_6_layers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 200 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 10 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_6_layers_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 4 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_4_layers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 4 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 6 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_4_layers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 4 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 6 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_4_layers_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 4 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 200 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 6 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --resume-weights last_net_1_diffusion_transformer_tuning_1.0.pth --resume-training-state training_state_1_last_diffusion_transformer_tuning_1.0.pth --run_name fix_att_code_merge_into_4_layers_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 3 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 6 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_3_layers_augment_224 Cifar10

# divide 1 layer into 3 sublayers
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_divide_into_3_sublayers Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 1.0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_divide_into_3_sublayers_epsilon_0.33_clip Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_divide_mean_into_3_sublayers_epsilon_0.33 Cifar10


### mix divide and merge layers
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.2 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_2_sublayers_t_[0./12,1./12,..]forinference_[0./12,0.5/12,1./12,...]fortrain_for_epoch_%2!=0_else_merge_2_layers_clipgrad_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.2 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_2_sublayers_t_[0./12,2./12,4./12..]for_celoss_[0./12,0.5/12,1./12,...]for_meanloss_for_epoch_%2!=0_else_merge_2_layers_val_and_test_use_t_[0./12,2./12,4./12..]_clipgrad_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.2 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_expanded_2_sublayers_t_[0./12,0.5/12,1./12,...]for_meanloss_merge_2_layers_celoss_val_and_test_use_t_[0./12,2./12,4./12..]_clipgrad_augment_224 Cifar10


### merge layers, time_index \in (0,1)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.5 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_merge_2_layers_celoss_meanloss_val_and_test_use_t_[0./12,2./12,4./12..]_clipgrad_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.5 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_merge_3_layers_celoss_meanloss_val_and_test_use_t_[0./12,3./12,6./12..]_clipgrad_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_merge_2_layers_celoss_meanloss_val_and_test_use_t_[0./12,2./12,4./12..]_no_clipgrad_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_merge_2_layers_celoss_meanloss_val_and_test_use_t_[0./12,2./12,4./12..]_no_clipgrad_augment_224 Cifar10

# time index 1 2 .. 6
CUDA_VISIBLE_DEVICES=2,4,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_6_layers_augment_224_rerun Cifar10
CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_into_6_layers_augment_224_rerun2 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2layers_into_6_layers_replace_celoss_with_distillation_loss_T=2_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2layers_into_6_layers_add_celoss_to_distillation_loss_T=2_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2layers_into_6_layers_replace_celoss_with_distillation_loss_T=2_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=3_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_augment_224 Cifar10



CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29515 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.2 --lambda_var 0 --lambda_ce 0.8 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29520 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.3 --lambda_var 0 --lambda_ce 0.7 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 32 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 6 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_DiT_6layers_72M_augment_224 Cifar10


### celoss learn from soft logits of pretrained model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.5 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_merge_2_layers_celoss_meanloss_val_and_test_use_t_[0./12,2./12,4./12..]_celoss_learn_from_soft_logits_of_pretrain_model_T=2_clipgrad_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=2,3,4,6 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.5 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_12_layers_merge_2_layers_celoss_meanloss_val_and_test_use_t_[0./12,2./12,4./12..]_celoss_+_distillation_loss_T=2_clipgrad_augment_224 Cifar10


### freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t] (1 forward only for celoss and return means for training meanloss)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10

# change lr
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-4 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 3e-4 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 1e-5 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 3e-5 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 50 --nb-run 1 --lr 5e-5 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 3e-4 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10

# increase model size 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 5 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 6 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10

# use var_range 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 1 --lambda_ce 1 --var_range 1e-3 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_set_var=var_rangexN0,1_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 1 --lambda_ce 1 --var_range 1e-4 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_set_var=var_rangexN0,1_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 1 --lambda_ce 1 --var_range 1e-5 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_set_var=var_rangexN0,1_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 1 --lambda_ce 1 --var_range 1e-2 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_set_var=var_rangexN0,1_augment_224 Cifar10

# use celoss = celoss + distillation_loss
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29505 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.1 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss_+=_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.2 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss_+=_distillation_loss_T=1_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10

# meanloss = mean + std*eps
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 1.0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_meanloss=mean+stdxeps-mean_vit_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_set_var=var_rangexN0,1_augment_224 Cifar10


# don't freeze after epoch 50
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_freeze_embedding_layernorm..classifier_after_epoch_10_train_classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_replace_celoss_with_distillation_loss_T=1_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 5 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss=_distillation_loss_T=1_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 6 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss=_distillation_loss_T=1_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10


# change weight
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 0.5 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss=_distillation_loss_T=1_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 0.1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss=_distillation_loss_T=1_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_augment_224 Cifar10


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0.2 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss=_distillation_loss_T=1_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_use_both_train+val_to_train_augment_224 Cifar10

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_use_both_train+val_to_train_augment_224 Cifar10


# current best 
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29510 main.py --model diffusion --seed 0 --depth 6 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 64 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 0 --accumulation-steps 1 --save-dir ./results/diffusion_finetuned --backbone transformer --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name fix_att_code_merge_2_layers_into_6_layers_celoss_no_freeze_embedding_layernorm..classifier_change_forward_of_meanloss_input=mean_plus_std_not_x[t]_use_both_train+val_to_train_augment_224 Cifar10