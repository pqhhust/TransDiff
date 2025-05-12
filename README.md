# TransDiff
## CIFAR
### Environment Setup
To begin, create a dedicated Conda environment and install the necessary dependencies for the experiments.
```
conda create -n diffomer python=3.8
conda activate diffomer
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

###
OOD data preparation
For downloading SVHN dataset, run
```
cd CIFAR/data
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
```

For downloading LSUN dataset, run
```
cd CIFAR/data
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

### Test OOD Detection
We evaluate OOD Detection on SVHN, LSUN, CIFAR10, CIFAR100

#### SVHN
Vanilla ViT
```
python3 main.py --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/vit_out --ood_data svhn --ood_test_dir ./data Cifar10
```

KEP-SVGPs
```
python3 main.py --attn-type kep_svgp --concate --ksvd-layers 3 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/vit_out_cat --ood_data svhn --ood_test_dir ./data Cifar10
```

DiT - ViT
```
python3 main.py --model diffusion --seed 0 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2 --run_name DiT-5-seed --ood_data svhn --ood_test_dir ./data Cifar10
```

DiT - KEP-SVGPs
```
python3 main.py --model diffusion --seed 0 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 1 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3 --run_name DiT-5-seed --ood_data svhn --ood_test_dir ./data Cifar10
```

SV-DKL
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 50 --nb-run 1 --model svdkl --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out --pretrained_dir ./results/vit_out --pretrained_seed 0 --ood_data svhn --ood_test_dir ./data Cifar10
```

KFLLA
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model kflla --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out --ood_data svhn --ood_test_dir ./data Cifar10
```

MC Dropout
```
python3 main.py --seed 0 --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 0 --nb-run 1 --model mc_dropout --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out --ood_data svhn --ood_test_dir ./data Cifar10
```

We do not run for TS since TS is designed for in-distribution tasks.
#### LSUN
Use the following settings
```
ood_data lsun
ood_test_dir ./data/LSUN
```

## Cola
## IMDB