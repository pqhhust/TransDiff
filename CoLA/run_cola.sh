################ Baseline ViT ################
main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 0 --save-dir ./results/vit_out

# python3 test.py \
# --attn-type softmax \
# --batch-size 32 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cola \
# --save-dir ./results/vit_out

################ KEP-SVGP-Attention ################ 
########## e(x)+r(x) ##########
main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --weight-decay 5e-5 --seed 1 --save-dir ./results/vit_out_sum

# python3 test.py \
# --attn-type kep_svgp \
# --ksvd-layers 1 \
# --eta-ksvd 1 \
# --batch-size 32 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cola \
# --save-dir ./results/vit_out_sum

################ SGPA-Attention ################
python3 main.py --depth 5 --attn-type sgpa --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --model vit_cola --lr 5e-4 --seed 0 --save-dir ./results/vit_out_sgpa

# python3 test.py \
# --attn-type sgpa \
# --batch-size 32 \
# --gpu 0 \
# --nb-run 1 \
# --model vit_cola \
# --save-dir ./results/vit_out_sgpa

### Match ViT with Diffusion ###
python3 main.py --model diffusion --seed 0 --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2

### Match SGPA with Diffusion ###
python3 main.py --model diffusion --seed 0 --depth 5 --attn-type sgpa --batch-size 32 --gpu 0 --nb-epochs 100 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out_sgpa --pretrained_seed 0 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3