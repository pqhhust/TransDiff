### Vanilla ViT
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 0 --save-dir ./results/vit_out

### KEP-SVGP 1 layer
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 1 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 50 --nb-run 1 --model transformer_imdb --lr 1e-3 --seed 0 --save-dir ./results/vit_out_sum

### DiT match with Vanilla ViT
python3 main.py --depth 5 --attn-type softmax --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --seed 0 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.8 --lambda_var 0 --lambda_ce 0.2

### DiT match with KEP-SVGP 1 layer
python3 main.py --depth 5 --attn-type kep_svgp --ksvd-layers 5 --eta-ksvd 10 --batch-size 32 --gpu 0 --nb-epochs 20 --nb-run 1 --model diffusion --lr 5e-3 --weight-decay 5e-5 --seed 0 --save-dir ./results/diffusion --backbone transformer --pretrained_dir ./results/vit_out --pretrained_seed 0 --trans_depth 1 --trans_num_heads 8 --trans_mlp_ratio 1 --trans_dropout 0.1 --lambda_mean 0.5 --lambda_var 0.2 --lambda_ce 0.3