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

### Model training
Below are the command for parallel training Difformer on ImageNet-1K.
Adjust the values based on your hardware setup
- CUDA_VISIBLE_DEVICES: Specify the GPU IDs (e.g., 0,1,2,3 for 4 GPUs)
- nproc_per_node: Set to the number of GPUs/processes per node
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 128 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 0 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0 --lambda_ce 0.5 --run_name DiT-5-seed ImageNet
```

## Cola
## IMDB
