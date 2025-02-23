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
python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone mlp --pretrained_dir ./results/vit_out_cat --clip 0.1 --mlp_hdim1 1024 --mlp_hdim2 1024 --mlp_hdim3 1024 --mlp_hdim4 64 --pretrained_seed 3 --mlp_dropout 0.1 --mlp_gamma 1.0 Cifar10 

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

python3 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 16 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/imagenet/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed ImageNet
