################ Baseline ViT ################
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 main.py \
--depth 5 \
--attn-type softmax \
--batch-size 128 \
--num_heads 4 \
--hdim 128 \
--gpu 1 \
--nb-epochs 600 \
--nb-run 1 \
--model vit_cifar \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ../experiments/CIFAR10_out/vit_out \
Cifar10

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 test.py \
--depth 5 \
--attn-type softmax \
--hdim 128 \
--num_heads 4 \
--batch-size 128 \
--gpu 1 \
--nb-run 1 \
--model vit_cifar \
--seed 0 \
--save-dir ../experiments/CIFAR10_out/vit_out \
Cifar10

################ KEP-SVGP-Attention ################ 
########## [e(x),r(x)] ##########
# one-layer of KSVD
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 main.py \
--seed 0 \
--wandb-key 6cf7b84d1bd52c9eb1e5eade43f583a8059231f2 \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 1 \
--nb-epochs 300 \
--nb-run 1 \
--model vit_cifar \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ../experiments/CIFAR10_out/vit_out_cat \
Cifar10

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 test.py \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 1 \
--nb-run 1 \
--model vit_cifar \
--save-dir ../experiments/CIFAR10_out/vit_out_cat \
--seed 0 \
Cifar10
# --ood-data cifar100 \
# --ood-train-dir ../data/CIFAR100/train \
# --ood-val-dir ../data/CIFAR100/val \
# --ood-test-dir ../data/CIFAR100/test \
# Cifar10

########## e(x)+r(x) ##########
# one-layer of KSVD
python3 main.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 128 \
--gpu 0 \
--nb-epochs 300 \
--nb-run 1 \
--model vit_cifar \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ./CIFAR10_out/vit_out_sum \
Cifar10

python3 test.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 128 \
--gpu 0 \
--nb-run 1 \
--model vit_cifar \
--save-dir ./CIFAR10_out/vit_out_sum \
Cifar10

################## KEP-SVGP-Diffusion ##################
########## Training Diffusion Model ##########
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 main.py \
--seed 0 \
--wandb-key 6cf7b84d1bd52c9eb1e5eade43f583a8059231f2 \
--depth 7 \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--num_heads 12 \
--hdim 384 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 1 \
--nb-epochs 300 \
--nb-run 1 \
--model diffusion \
--lr 1e-3 \
--weight-decay 5e-5 \
--save-dir ../experiments/CIFAR10_out/diffusion \
--pretrained_dir ../experiments/CIFAR10_out/vit_out_cat \
--backbone mlp \
Cifar10

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 test.py \
--attn-type kep_svgp \
--concate \
--ksvd-layers 7 \
--eta-ksvd 10 \
--batch-size 128 \
--gpu 1 \
--nb-run 1 \
--model diffusion \
--backbone mlp \
--save-dir ../experiments/CIFAR10_out/diffusion \
--pretrained_dir ../experiments/CIFAR10_out/vit_out_cat \
--seed 0 \
Cifar10

# 1 layers
# Branch,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# blur,70.78450000000001,79.599,90.066,56.979499999999994,77.3815,125.978,74.933,21.4765,17.5685,49.1455
# digital,73.7964,81.3976,90.50120000000001,55.654799999999994,74.2076,115.15400000000001,62.4088,19.1588,16.2332,44.0212
# noise,58.36,74.48466666666667,79.102,62.938,82.87866666666667,248.6,128.71733333333333,30.891333333333332,26.970666666666666,69.91133333333333
# weather,70.54199999999999,79.54333333333334,89.50733333333332,56.95666666666667,77.30866666666667,131.11733333333333,76.67800000000001,21.435333333333332,17.304666666666666,49.38133333333334


# cifar10,vit_cifar
# MSP_results,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# ,84.12±nan,87.07±nan,97.30±nan,50.94±nan,65.05±nan,37.18±nan,23.85±nan,11.28±nan,8.52±nan,26.42±nan

# 2 layers
# cifar10,vit_cifar
# MSP_results,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# ,83.45±nan,86.83±nan,97.14±nan,51.03±nan,67.07±nan,39.95±nan,25.43±nan,11.79±nan,9.01±nan,27.71±nan

# Branch,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# blur,70.784,79.649,90.223,56.795,77.6945,124.875,74.2485,21.534,17.6715,49.240500000000004
# digital,72.71,81.2196,89.8348,56.4592,74.8648,123.3512,64.5588,20.1672,16.9376,46.0172
# noise,59.287333333333336,75.86333333333333,81.056,63.721333333333334,81.318,231.14399999999998,118.29266666666666,30.482000000000003,26.801333333333332,68.41933333333333
# weather,69.08533333333334,79.216,88.65866666666668,58.135999999999996,77.574,141.728,81.122,22.468,18.208666666666666,51.654



# 7 layers
# Branch,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# blur,62.99399999999999,77.159,84.4315,61.7555,79.556,193.0625,103.7835,20.490000000000002,15.133000000000001,56.0515
# digital,67.742,79.0364,86.4444,59.5432,76.4508,162.56,85.1384,16.7876,13.8776,48.6
# noise,47.379333333333335,71.01066666666667,68.78533333333333,69.402,84.98066666666666,362.812,165.63866666666667,29.157333333333334,21.422,77.87533333333333
# weather,63.83866666666667,77.43866666666666,84.88466666666666,61.45133333333333,78.73333333333333,187.14600000000002,100.30533333333334,18.612000000000002,14.515333333333333,54.118

# cifar10,vit_cifar
# MSP_results,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# ,82.95±nan,87.18±nan,97.12±nan,54.45±nan,63.17±nan,40.44±nan,25.00±nan,7.46±nan,5.61±nan,25.05±nan

# cifar10,diffusion
# MSP_results,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# ,83.83±nan,86.29±nan,97.07±nan,50.52±nan,64.87±nan,39.68±nan,25.84±nan,11.11±nan,8.54±nan,26.87±nan

# Branch,Acc.,AUROC,AUPR Succ.,AUPR,FPR,AURC,EAURC,ECE,NLL,Brier
# blur,69.431,78.863,88.999,57.391,78.55850000000001,138.0345,81.3015,22.119999999999997,17.7395,51.163
# digital,73.1496,81.1456,89.98360000000001,56.2936,74.3248,120.49199999999999,64.1944,18.938,15.9888,44.42359999999999
# noise,56.14866666666667,73.59266666666667,76.86200000000001,64.13666666666667,83.10533333333333,272.53933333333333,138.3,32.682,28.616,73.65933333333334
# weather,69.73866666666666,79.45333333333333,89.27,57.33266666666667,78.472,135.1906666666667,77.90266666666666,21.634,17.260666666666665,50.374
# Overall,68.0756,78.68786666666668,86.95400000000001,58.36266666666666,78.03933333333333,158.5192,86.31906666666667,23.074533333333328,19.235466666666664,53.258