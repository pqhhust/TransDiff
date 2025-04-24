# Vanilla ViT
python3 main.py --attn-type softmax --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out Cifar100

# KEP-SVGP 1 layers
python3 main.py --attn-type kep_svgp --concate --ksvd-layers 1 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 300 --nb-run 1 --model vit_cifar --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/vit_out_cat Cifar100

# DiT match with vanilla ViT
main.py --model diffusion --seed 0 --depth 7 --attn-type softmax --num_heads 12 --hdim 384 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0 --lambda_ce 0.5 --run_name DiT-5-seed Cifar100

# DiT match with KEP-SVGP
main.py --model diffusion --seed 0 --depth 7 --attn-type kep_svgp --concate --ksvd-layers 7 --num_heads 12 --hdim 384 --eta-ksvd 10 --batch-size 128 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_cat --pretrained_seed 0 --trans_depth 1 --trans_num_heads 12 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.4 --lambda_var 0.2 --lambda_ce 0.4 --run_name DiT-5-seed Cifar100