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

# python3 main.py \
# --seed 5 \
# --attn-type softmax \
# --batch-size 128 \
# --gpu 0 \
# --nb-epochs 3000 \
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

python3 test.py \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out_cat \
--seed 0 \
Cifar10 \
&python3 test.py \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out_cat \
--seed 1 \
Cifar10 \
&python3 test.py \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out_cat \
--seed 2 \
Cifar10 \
&python3 test.py \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out_cat \
--seed 3 \
Cifar10 \
&python3 test.py \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out_cat \
--seed 4 \
Cifar10

### Train Diffusion
# python3 main.py \
# --model diffusion \
# --seed 0 \
# --depth 7 \
# --attn-type kep_svgp \
# --concate \
# --ksvd-layers 7 \
# --num_heads 12 \
# --hdim 384 \
# --eta-ksvd 10 \
# --batch-size 128 \
# --gpu 0 \
# --nb-epochs 100 \
# --nb-run 1 \
# --lr 1e-3 \
# --weight-decay 5e-5 \
# --save-dir ./results/diffusion \
# --backbone mlp \
# --pretrained_dir ./results/vit_out_cat \
# --ema_decay 0.999 \
# --ema_update_every 1 \
# --clip 0.01 \
# --mlp_hdim 512 \
# --pretrained_seed 0 \
# --mlp_dropout 0.1 \
# Cifar10