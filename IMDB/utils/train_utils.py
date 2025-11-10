import argparse

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Kernel-Eigen Pair Sparse Variational Gaussian Processes',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nb-epochs', default=20, type=int, help='Total number of training epochs ')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--num-classes',type=int,default=2)
    parser.add_argument('--max-len',type=int,default=100)
    parser.add_argument('--dataset', default='imdb', type=str, choices = ['imdb'], help='dataset')
    parser.add_argument('--seed', type=int,default=0)

    # KEP-SVGP-attention
    parser.add_argument('--ksvd-layers', type=int, default=1, help='Number of ksvd layers applied to the transformer')
    parser.add_argument('--attn-type', default='kep_svgp', type=str, choices = ['kep_svgp', 'softmax', 'sgpa'], help='Type of attention')
    parser.add_argument('--eta-ksvd', type=float, default=0.1, help='coefficient of the KSVD regularization')
    parser.add_argument('--eta-kl', type=float, default=1.0, help='coefficient of the KL divergence regularization')
    parser.add_argument('--low-rank', type=int, default=10, help='Number of dimension the low rank method projected to')
    parser.add_argument('--rank-multi', type=int, default=10, help='low rank dimension * rank_multi')

    ## optimizer 
    parser.add_argument('--lr', default=1e-3, type=float, help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument("--min-lr", default=1e-4, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--warmup-epoch", default=5, type=int)

    ## nb of run + print freq
    parser.add_argument('--nb-run', default=1, type=int, help='Run n times, in order to compute std')

    ## dataset setting
    parser.add_argument('--nb-worker', default=4, type=int, help='Nb of workers')
    
    ## Model
    parser.add_argument('--model', default='transformer_imdb', type=str, choices = ['transformer_imdb', 'diffusion', 'svdkl', 'temperature_scaling', 'mc_dropout'], help='Models name to use')
    parser.add_argument('--emb_dim',type=int,default=128)
    parser.add_argument('--depth',type=int,default=5)
    parser.add_argument('--hdim',type=int,default=128)
    parser.add_argument('--num_heads',type=int,default=8)
    
    parser.add_argument('--save-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

    parser.add_argument('--lambda_mean', default=0, type=float, help='weight of mean_loss')
    parser.add_argument('--lambda_var', default=0, type=float, help='weight of var_loss')
    parser.add_argument('--lambda_ce', default=1., type=float, help='weight of ce_loss')
    parser.add_argument('--run_name', default=None, type=str, help='name of wandb run')
    parser.add_argument('--backbone', type=str, default='mlp', choices=['mlp', 'unet1d', 'transformer', 'mlp_mixer'], help='Backbone name')
    parser.add_argument('--pretrained_dir', default=None, type=str, help='Pretrained diffusion model directory')
    parser.add_argument('--use_ema', type=bool, default=True, help='Whether to use EMA')
    parser.add_argument('--ema_decay', default=0.995, type=float, help='Exponential moving average decay')
    parser.add_argument('--update_ema_interval', default=5, type=int, help='Update EMA every n steps')
    parser.add_argument('--start_ema_step', default=50, type=int, help='Start EMA step')
    parser.add_argument('--clip', default=0, type=float, help='std error clipping value')
    parser.add_argument('--mlp_hdim1', default=64, type=int, help='hidden dimension 1 for diffusion mlp')
    parser.add_argument('--mlp_hdim2', default=64, type=int, help='hidden dimension 2 for diffusion mlp')
    parser.add_argument('--mlp_hdim3', default=64, type=int, help='hidden dimension 3 for diffusion mlp')
    parser.add_argument('--mlp_hdim4', default=64, type=int, help='hidden dimension 4 for diffusion mlp')
    parser.add_argument('--pretrained_seed', default=0, type=int, help='seed for pretraining ViT')
    parser.add_argument('--mlp_dropout', default=0.1, type=float, help='dropout rate for diffusion mlp')
    parser.add_argument('--mlp_gamma', default=1., type=float, help='weight of stds_loss')
    
    parser.add_argument('--trans_depth', type=int, help='number of DiTBlock')
    parser.add_argument('--trans_num_heads', type=int, help='number of heads of a DiTBlock')
    parser.add_argument('--trans_mlp_ratio', type=float, help='ratio between mlp hidden dimension of a transformer layer and d_model')
    parser.add_argument('--trans_dropout', type=float, help='dropout rate for transformer backbone')

    return parser.parse_args()
