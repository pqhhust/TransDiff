# TransDiff
## CIFAR

## Cola
### Environment
Since we use [``allennlp``](https://github.com/allenai/allennlp) package, we need to install
```
conda create -n allennlp python=3.7
conda activate allennlp
pip install allennlp
pip install warmup_scheduler
### run if core dumped
# pip install torch==1.13.0
```

### Dataset
Please download the dataset via
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

### Run Tasks
Please train our model according to `run_cola.sh`.

### Acknowledgement
This code is based on the official codes of [huggingface](https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/bert/modeling_bert.py), [SGPA](https://github.com/chenw20/SGPA/).


## IMDB
### Environment
Our model can be learnt on a **single NVIDIA GeForce RTX 2070 SUPER GPU** 
```
conda create -n imdb python=3.8
conda activate imdb
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install scikit-learn==1.0.2
pip install transformers
pip install warmup_scheduler
pip install wandb
```
### Dataset
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
### Run Tasks
Please train our model according to `run_imdb.sh`. 
### Acknowledgement
This code is based on [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/master), [text](https://github.com/pytorch/text).