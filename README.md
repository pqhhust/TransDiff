# DIRECTOR
To reproduce all the experiments, set your Weights & Biases API key using: 
```
export WANDB_API_KEY=<your_api_key>
```

## CIFAR
### Environment Setup
To begin, create a dedicated Conda environment and install the necessary dependencies for the experiments.
```
conda create -n director python=3.10
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
Training the DIRECTOR involves two stages: (1) pre-training a transformer model (either ViT or KEP-SVGP), and (2) training the diffusion model to align with the pre-trained model. Below are the commands for each step. All commands should be run in the ```CIFAR``` directory.


- Pre-train Vision Transformers (ViT) on CIFAR-10 using the following commands:
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 1 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 2 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 4 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
```

- Pre-train KEP-SVGPs (configuring the `ksvd-layers` to values of 1, 2 and 7) using the commands below:
```
python3 main.py --seed 0 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 1 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 2 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 3 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
python3 main.py --seed 4 --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar10
```

- Train the diffusion model to align with the pre-trained ViT using the commands below:
```
python3 main.py --model diffusion --seed 0 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 1 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 1 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 2 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 2 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 3 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 3 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 4 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 4 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed Cifar10
```

- Train the diffusion model to align with the pre-trained KEP-SVGP (configuring the `ksvd-layers` to values of 1, 2 and 7) using the commands below:
```
python3 main.py --model diffusion --seed 0 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 1 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 1 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 2 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 2 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 3 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 3 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
python3 main.py --model diffusion --seed 4 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 4 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar10
```

### Baselines
Run the following commands to reproduce results for compared baselines.
- Temperature Scaling
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 1 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 2 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 4 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
```

- MC Dropout
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 1 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 2 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 4 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
```

- KFLLA
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 1 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 2 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 4 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
```

- SVDKL
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 50 --nb-run 1 --model temperature_scaling --lr 1e-2 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 1 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 50 --nb-run 1 --model temperature_scaling --lr 1e-2 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 2 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 50 --nb-run 1 --model temperature_scaling --lr 1e-2 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 50 --nb-run 1 --model temperature_scaling --lr 1e-2 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 4 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 50 --nb-run 1 --model temperature_scaling --lr 1e-2 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
```

- SGPA
```
python3 main.py --seed 0 --attn-type sgpa --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 1 --attn-type sgpa --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 2 --attn-type sgpa --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 3 --attn-type sgpa --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
python3 main.py --seed 4 --attn-type sgpa --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar10
```

### Distillation
- Distill from ViT (7 layers) to ViT (3 layers)
```
python3 main.py --depth 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model vit_cifar_distillation --lr 1e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/vit_out_distillation --pretrained_seed 0 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0.0 --lambda_ce 0.2 --temperature 3 --run_name ViT-distillation Cifar10
python3 main.py --depth 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model vit_cifar_distillation --lr 1e-3 --weight-decay 5e-5 --seed 1 --save-dir ./results/vit_out_distillation --pretrained_seed 1 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0.0 --lambda_ce 0.2 --temperature 3 --run_name ViT-distillation Cifar10
python3 main.py --depth 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model vit_cifar_distillation --lr 1e-3 --weight-decay 5e-5 --seed 2 --save-dir ./results/vit_out_distillation --pretrained_seed 2 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0.0 --lambda_ce 0.2 --temperature 3 --run_name ViT-distillation Cifar10
python3 main.py --depth 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model vit_cifar_distillation --lr 1e-3 --weight-decay 5e-5 --seed 3 --save-dir ./results/vit_out_distillation --pretrained_seed 3 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0.0 --lambda_ce 0.2 --temperature 3 --run_name ViT-distillation Cifar10
python3 main.py --depth 3 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model vit_cifar_distillation --lr 1e-3 --weight-decay 5e-5 --seed 4 --save-dir ./results/vit_out_distillation --pretrained_seed 4 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0.0 --lambda_ce 0.2 --temperature 3 --run_name ViT-distillation Cifar10
```

- Distill from ViT (7 layers) to DiT
```
python3 main.py --model diffusion_distillation --seed 0 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --save-dir ./results/diffusion_distillation --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --pretrained_seed 0 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --temperature 3 --run_name DiT-5-seed-distillation Cifar10
python3 main.py --model diffusion_distillation --seed 1 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --save-dir ./results/diffusion_distillation --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --pretrained_seed 1 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --temperature 3 --run_name DiT-5-seed-distillation Cifar10
python3 main.py --model diffusion_distillation --seed 2 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --save-dir ./results/diffusion_distillation --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --pretrained_seed 2 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --temperature 3 --run_name DiT-5-seed-distillation Cifar10
python3 main.py --model diffusion_distillation --seed 3 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --save-dir ./results/diffusion_distillation --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --pretrained_seed 3 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --temperature 3 --run_name DiT-5-seed-distillation Cifar10
python3 main.py --model diffusion_distillation --seed 4 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --save-dir ./results/diffusion_distillation --backbone transformer --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --pretrained_seed 4 --pretrained_dir ./results/vit_out --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --temperature 3 --run_name DiT-5-seed-distillation Cifar10
```

### Test OOD Detection
We evaluate OOD Detection on SVHN, LSUN, CIFAR10, CIFAR100.

For downloading SVHN dataset, run
```
cd CIFAR/data
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
python3 selected_svhn_data.py
```

For downloading LSUN dataset, run
```
cd CIFAR/data
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

#### SVHN
- Vanilla ViT
```
python3 main_ood.py --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/vit_out --ood_data svhn --ood_test_dir ./data Cifar10
```

- KEP-SVGPs
```
python3 main_ood.py --attn-type kep_svgp --concate --ksvd-layers 3 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/vit_out_cat --ood_data svhn --ood_test_dir ./data Cifar10
```

- DiT - ViT
```
python3 main_ood.py --model diffusion --seed 0 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed --ood_data svhn --ood_test_dir ./data Cifar10
```

- DiT - KEP-SVGPs
```
python3 main_ood.py --model diffusion --seed 0 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3 --run_name DiT-5-seed --ood_data svhn --ood_test_dir ./data Cifar10
```

- SV-DKL
```
python3 main_ood.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out --pretrained_dir ./results/vit_out --pretrained_seed 0 --ood_data svhn --ood_test_dir ./data Cifar10
```

- KFLLA
```
python3 main_ood.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out --ood_data svhn --ood_test_dir ./data Cifar10
```

- MC Dropout
```
python3 main_ood.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out --ood_data svhn --ood_test_dir ./data Cifar10
```

- SGPA
```
python3 main_ood.py --attn-type sgpa --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/vit_out --ood_data svhn --ood_test_dir ./data Cifar10
```

We do not run for TS since TS is designed for in-distribution tasks.
#### LSUN
Use the following settings
```
ood_data lsun
ood_test_dir ./data/LSUN
```

## CoLA
### Environment Setup
To begin, create a dedicated Conda environment and install the necessary dependencies for the experiments.
```
conda create -n cola python=3.10
conda activate cola
pip install allennlp
pip install wandb
pip install warmup_scheduler
pip install gpytorch
pip install laplace-torch
```

### Data preparation
In the ```CoLA``` directory, please download the dataset via
```
mkdir data
cd data
wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
unzip cola_public_1.1.zip
```
and use `in_domain_train.tsv`, `in_domain_dev.tsv`, `out_of_domain_dev.tsv` from the `raw/` folder. The structure of the file should be:
```
./data/
  ├── cola_public
    ├── raw
      ├── in_domain_train.tsv
      ├── in_domain_dev.tsv
      └── out_of_domain_dev.tsv
```

### Model training
Training the DIRECTOR involves two stages: (1) pre-training a transformer model (either ViT or KEP-SVGP), and (2) training the diffusion model to align with the pre-trained model. Below are the commands for each step. All commands should be run in the ```CoLA``` directory.

- Pre-train Transformers on CoLA using the following commands:
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 4 --save-dir ./results/vit_out
```

- Pre-train KEP-SVGPs (configuring the `ksvd-layers` to values of 1, 2 and 5) using the commands below:
```
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --weight-decay 5e-5 --seed 0 --save-dir ./results/vit_out_sum
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --weight-decay 5e-5 --seed 1 --save-dir ./results/vit_out_sum
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --weight-decay 5e-5 --seed 2 --save-dir ./results/vit_out_sum
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --weight-decay 5e-5 --seed 3 --save-dir ./results/vit_out_sum
main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --weight-decay 5e-5 --seed 4 --save-dir ./results/vit_out_sum
```

- Train the diffusion model to align with the pre-trained Transformer using the commands below:
```
python3 main.py --model diffusion --seed 0 --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --model diffusion --seed 1 --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 1 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --model diffusion --seed 2 --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 2 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --model diffusion --seed 3 --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 3 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --model diffusion --seed 4 --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 4 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
```

- Train the diffusion model to align with the pre-trained KEP-SVGP (configuring the `ksvd-layers` to values of 1, 2 and 5) using the commands below:
```
python3 main.py --model diffusion --seed 0 --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --lr 5e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_sum --pretrained_seed 0 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --model diffusion --seed 1 --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --lr 5e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_sum --pretrained_seed 1 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --model diffusion --seed 2 --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --lr 5e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_sum --pretrained_seed 2 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --model diffusion --seed 3 --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --lr 5e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_sum --pretrained_seed 3 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --model diffusion --seed 4 --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --lr 5e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_sum --pretrained_seed 4 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
```

### Baselines
Run the following commands to reproduce results for compared baselines.
- Temperature Scaling
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 5e-4 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 5e-4 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 5e-4 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 5e-4 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 5e-4 --seed 4 --save-dir ./results/vit_out
```

- MC Dropout
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 5e-4 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 5e-4 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 5e-4 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 5e-4 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 5e-4 --seed 4 --save-dir ./results/vit_out
```

- KFLLA
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 5e-4 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 5e-4 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 5e-4 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 5e-4 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 5e-4 --seed 4 --save-dir ./results/vit_out
```

- SVDKL
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 4 --save-dir ./results/vit_out
```

- SGPA
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 4 --save-dir ./results/vit_out
```

## IMDB
### Environment Setup
To begin, IMDB task uses the same conda environment as CIFAR

### Data preparation
Move to IMDB directory via `cd IMDB`.

Please download dataset via
```
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvf aclImdb_v1.tar.gz
```

Run the following command to pre-process IMDB dataset
```
python3 preprocessing.py
```

### Model training
Training the DIRECTOR involves two stages: (1) pre-training a transformer model (either ViT or KEP-SVGP), and (2) training the diffusion model to align with the pre-trained model. Below are the commands for each step. All commands should be run in the ```IMDB``` directory.

Pre-train Transformers on IMDB using the following commands:
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 4 --save-dir ./results/vit_out
```

Pre-train KEP-SVGPs (configuring the `ksvd-layers` to values of 1, 2 and 5) using the commands below:
```
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 0 --save-dir ./results/vit_out_sum
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 1 --save-dir ./results/vit_out_sum
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 2 --save-dir ./results/vit_out_sum
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 3 --save-dir ./results/vit_out_sum
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 4 --save-dir ./results/vit_out_sum
```

Train the diffusion model to align with the pre-trained Transformer using the commands below:
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --seed 0 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --seed 1 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 1 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --seed 3 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 3 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --seed 3 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 3 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --seed 4 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 4 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2
```

Train the diffusion model to align with the pre-trained KEP-SVGP (configuring the `ksvd-layers` to values of 1, 2 and 5) using the commands below:
```
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --weight-decay 5e-5 --seed 1 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 1 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --weight-decay 5e-5 --seed 2 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 2 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --weight-decay 5e-5 --seed 3 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 3 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --weight-decay 5e-5 --seed 4 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 4 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3
```

### Baselines
Run the following commands to reproduce results for compared baselines.
- Temperature Scaling
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model temperature_scaling --lr 1e-3 --seed 4 --save-dir ./results/vit_out
```

- MC Dropout
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --seed 4 --save-dir ./results/vit_out
```

- KFLLA
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --seed 4 --save-dir ./results/vit_out
```

- SVDKL
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-2 --seed 4 --save-dir ./results/vit_out
```

- SGPA
```
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 0 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 1 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 2 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 3 --save-dir ./results/vit_out
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 4 --save-dir ./results/vit_out
```

## Acknowledgement
This repository is based on official codes in: [SGPA](https://github.com/chenw20/SGPA/), [KEP-SVGPs](https://github.com/yingyichen-cyy/KEP-SVGP/), [DiT](https://github.com/facebookresearch/DiT), [IMDB Sentiment Analysis](https://www.kaggle.com/code/dhaniyapudina/sentiment-analysis-on-imdb-movie-review-using-bert#4.-Tokenizing-Data-with-BERT-Tokenizer), [TS](https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py), [MCD](https://torch-uncertainty.github.io/_modules/torch_uncertainty/models/wrappers/mc_dropout.html#mc_dropout), [KFLLA](https://aleximmer.com/Laplace/calibration_example/), [SVDKL](https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html)
