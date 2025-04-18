################ Baseline ViT ################
python3 main.py \
--attn-type softmax \
--batch-size 32 \
--gpu 0 \
--nb-epochs 50 \
--nb-run 1 \
--model vit_cola \
--lr 5e-4 \
--seed 2 \
--save-dir ./results/vit_out

python3 test.py \
--attn-type softmax \
--batch-size 32 \
--gpu 0 \
--nb-run 1 \
--model vit_cola \
--save-dir ./results/vit_out

################ KEP-SVGP-Attention ################ 
########## e(x)+r(x) ##########
python3 main.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 32 \
--gpu 0 \
--nb-epochs 50 \
--nb-run 1 \
--model vit_cola \
--lr 5e-4 \
--weight-decay 5e-5 \
--seed 0 \
--save-dir ./results/vit_out_sum

python3 test.py \
--attn-type kep_svgp \
--ksvd-layers 1 \
--eta-ksvd 1 \
--batch-size 32 \
--gpu 0 \
--nb-run 1 \
--model vit_cola \
--save-dir ./results/vit_out_sum

python3 main.py --model diffusion --seed 0 --depth 2 --attn-type kep_svgp --ksvd-layers 2 --num_heads 4 --hdim 256 --eta-ksvd 1 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --lr 1e-3 --weight-decay 5e-5 --save-dir ./cola_out/diffusion --backbone mlp --pretrained_dir ./cola_out/vit_out_sum --clip 0.1 --mlp_hdim1 64 --mlp_hdim2 64 --mlp_hdim3 64 --pretrained_seed 0 --mlp_dropout 0

python3 main.py --model diffusion --seed 0 --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --lr 5e-4 --weight-decay 5e-5 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 4 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 1.0 --lambda_var 0 --lambda_ce 1.0