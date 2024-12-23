# TransDiff
## CIFAR
First, create a conda environment and download packages for experiments
```
conda create -n transdiff python=3.8
conda activate transdiff
pip install -r requirements.txt
```

In the ```CIFAR/data``` directory run the following command
```
bash scripts/download_cifar.sh
bash scripts/download_cifar10c.sh
```

The commands for training and testing are placed in the ```CIFAR/experiments/run_scripts``` directory

## Cola
## IMDB