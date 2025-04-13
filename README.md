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
In the `CIFAR/data` directory, prepare the datasets by running sequentially the following commands.

#### Step 1: Directory Creation

Create `IMAGENET1K` folder
```
mkdir IMAGENET1K
mkdir IMAGENET1K/train
mkdir IMAGENET1K/val
mkdir IMAGENET1K/ILSVRC2012_devkit_t12
```

#### Step 2: Downloading Training Dataset

Commands for downloading the ImageNet training dataset
```
wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
tar -xvf ./ILSVRC2012_img_train.tar -C ./IMAGENET1K/train 
```
#### Step 3: Downloading Val Dataset

Commands for downloading the ImageNet val dataset
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xvf ./ILSVRC2012_img_val.tar -C ./IMAGENET1K/val 
```
#### Step 4: Downloading Development Toolkits

Commands for downloading the ImageNet's development toolkits
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
tar -xvf ./ILSVRC2012_devkit_t12.tar.gz -C ./IMAGENET1K/ILSVRC2012_devkit_t12 --strip-components=1
```
#### Step 5: Checking the folder structure

The `IMAGENET1K` folder must have the structure
```
IMAGENET1K
├── train
├── val
└── ILSVRC2012_devkit_t12
```

#### Step 6: Extraction and Reorganization

Run `bash download_imagenet.sh` to extract all files in train folder.

Run the following command
```
python3 extract_synset.py create_val_labels.py reorganize_val.py
```

The final structure for val and train directory must look the same as
```
train
├── n01440764
├── n01443537
└── n01484850
```

#### Step 7: Cleanup

If everything works, run the following command to remove tar files:
```
rm ./ILSVRC2012_img_train.tar ./ILSVRC2012_img_val.tar ./ILSVRC2012_devkit_t12.tar.gz
```

### Model training
Below are the command for parallel training Difformer on ImageNet-1K.
Adjust the values based on your hardware setup
The effective batch size is `nproc_per_node * batch-size * accumulation-steps`
- nproc_per_node: Set to the number of GPUs/processes per node
```
torchrun --nproc_per_node=4 main.py --model diffusion --seed 0 --depth 12 --attn-type softmax --num_heads 12 --hdim 768 --batch-size 32 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --warmup-epoch 5 --clip-grad-value 1.0 --accumulation-steps 4 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --resume-weights last_net_1_diffusion_transformer.pth --trans_depth 4 --trans_num_heads 12 --trans_mlp_ratio 4 --trans_dropout 0.1 --lambda_mean 1 --lambda_var 0 --lambda_ce 1 --run_name DiT-5-seed ImageNet
```

## Cola
## IMDB
