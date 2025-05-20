# DIRECTOR
## CIFAR
### Environment Setup
To begin, create a dedicated Conda environment and install the necessary dependencies for the experiments.
```
conda create -n director python=3.8
conda activate director
cd CIFAR
bash requirements.sh
```

### Data preparation
In the ```CIFAR/data``` directory run the following command
```
cd CIFAR/data
bash download_cifar.sh
bash download_cifar10c.sh
```

### Model training
Training the DIRECTOR involves two stages: (1) pre-training a transformer model (either ViT or KEP-SVGP), and (2) training the diffusion model to align with the pre-trained model. Below are the commands for each step.


Pre-train a Vision Transformer (ViT) model on CIFAR-10 using the following command:
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 1 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 2 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 4 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
```

Pre-train a KEP-SVGP model (a variant of ViT with kernel-based attention) using the command below:
```
python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
```

Train the diffusion model to align with the pre-trained ViT using the command below:
```
python3 main.py --model diffusion --seed 0 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 1 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 1 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 2 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 2 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 3 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 3 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 4 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 4 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
```

Train the diffusion model to align with the pre-trained KEP-SVGP using the command below:
```
python3 main.py --model diffusion --seed 0 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 1 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 1 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 2 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 2 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 3 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 4 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 4 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
```

## Cola
## IMDB