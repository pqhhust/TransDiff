# -f https://download.pytorch.org/whl/cu111/torch_stable.html
# torch==1.10.1+cu111
# torchvision==0.11.2+cu111
# tensorboard
# setuptools==59.5.0
# six
# scikit-learn
# pandas
# timm
# wandb
# warmup_scheduler
# einops
# ema-pytorch

### install
conda install pytorch==2.4.1 torchvision==0.20.0 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install setuptools
pip install six
pip install scikit-learn
pip install pandas
# pip install --upgrade setuptools
pip install timm
pip install warmup_scheduler
pip install wandb
pip install einops