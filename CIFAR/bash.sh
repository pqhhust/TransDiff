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
python3 test.py \
--seed 0 \
--attn-type softmax \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out \
Cifar10 \
&python3 test.py \
--seed 1 \
--attn-type softmax \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out \
Cifar10 \
&python3 test.py \
--seed 2 \
--attn-type softmax \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out \
Cifar10 \
&python3 test.py \
--seed 3 \
--attn-type softmax \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out \
Cifar10 \
&python3 test.py \
--seed 4 \
--attn-type softmax \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./results/vit_out \
Cifar10
