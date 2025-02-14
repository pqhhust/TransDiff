import argparse

def get_args_parser():
    
    parser = argparse.ArgumentParser(description='Kernel-Eigen Pair Sparse Variational Gaussian Processes',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nb-epochs', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--nb-worker', default=4, type=int, help='Nb of workers')
    parser.add_argument('--seed', default=0, type=int, help='Set seed for reproducibility')
    parser.add_argument('--wandb-key', default=None, type=str, help='Set wandb key for tracking progresses')
    
    ## Model
    parser.add_argument('--model', default='vit_cifar', type=str, choices = ['vit_cifar', 'diffusion'], help='Models name to use')
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--hdim', type=int, default=384)
    parser.add_argument('--num_heads', type=int, default=12)

    # KEP-SVGP-attention
    parser.add_argument('--ksvd-layers', type=int, default=1, help='Number of ksvd layers applied to the transformer')
    parser.add_argument('--attn-type', default='kep_svgp', type=str, choices = ['kep_svgp', 'softmax'], help='Type of attention')
    parser.add_argument('--concate', action='store_true', help='whether to use [e(x),r(x)] instead of (e(x)+r(x))')  
    parser.add_argument('--eta-ksvd', type=float, default=0.1, help='coefficient of the KSVD regularization')
    parser.add_argument('--eta-kl', type=float, default=1.0, help='coefficient of the KL divergence regularization')
    parser.add_argument('--low_rank', type=int, default=10, help='Number of dimension the low rank method projected to')
    parser.add_argument('--rank_multi', type=int, default=10, help='low rank dimension * rank_multi')

    ## optimizer 
    parser.add_argument('--lr', default=1e-3, type=float, help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--warmup-epoch", default=5, type=int)

    ## nb of run 
    parser.add_argument('--nb-run', default=1, type=int, help='Run n times, in order to compute std')
    parser.add_argument('--save-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

    ## diffusion
    # parser.add_argument('--stage', default=1, type=int, help='Stage of the training')

    ## dataset setting
    subparsers = parser.add_subparsers(title="dataset setting", dest="subcommand")
    Cifar10 = subparsers.add_parser("Cifar10",
                                    description='Dataset parser for training on Cifar10',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on Cifar10")
    Cifar10.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
    Cifar10.add_argument("--train-dir", type=str, default='./data/CIFAR10/train', help="Cifar10 train directory")
    Cifar10.add_argument("--val-dir", type=str, default='./data/CIFAR10/val', help="Cifar10 val directory")
    Cifar10.add_argument("--test-dir", type=str, default='./data/CIFAR10/test', help="Cifar10 test directory")
    Cifar10.add_argument("--corruption-dir", type=str, default='./data', help="Cifar10-C directory")
    Cifar10.add_argument("--nb-cls", type=int, default=10, help="number of classes in Cifar10")

    Cifar100 = subparsers.add_parser("Cifar100",
                                     description='Dataset parser for training on Cifar100',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     help="Dataset parser for training on Cifar100")
    Cifar100.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
    Cifar100.add_argument("--train-dir", type=str, default='./data/CIFAR100/train', help="Cifar100 train directory")
    Cifar100.add_argument("--val-dir", type=str, default='./data/CIFAR100/val', help="Cifar100 val directory")
    Cifar100.add_argument("--test-dir", type=str, default='./data/CIFAR100/test', help="Cifar100 test directory")
    Cifar100.add_argument("--nb-cls", type=int, default=100, help="number of classes in Cifar100")

    parser.add_argument('--backbone', type=str, default='mlp', choices=['mlp', 'unet1d', 'transformer', 'mlp_mixer', 'lstm', 'gru'], help='Backbone name')
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
    parser.add_argument('--lambda_mean', default=1., type=float, help='weight of mean_loss')
    parser.add_argument('--lambda_var', default=1., type=float, help='weight of var_loss')
    parser.add_argument('--lambda_ce', default=1., type=float, help='weight of ce_loss')
    parser.add_argument('--run_name', default=None, type=str, help='name of wandb run')
    parser.add_argument('--adversarial_noise', default=0.01, type=float, help='std of adversarial noise')
    parser.add_argument('--adversarial_samples', default=0, type=int, help='number of adversarial samples')

    parser.add_argument('--rnn_hidden', type=int, help='hidden dimension of rnn backbone')
    parser.add_argument('--rnn_num_layers', type=int, help='number of layers of rnn backbone')
    parser.add_argument('--rnn_dropout', type=float, help='dropout rate for rnn backbone')
    parser.add_argument('--rnn_low_dim', type=int, help='low dimension of rnn backbone')

    parser.add_argument('--trans_depth', type=int, help='number of DiTBlock')
    parser.add_argument('--trans_num_heads', type=int, help='number of heads of a DiTBlock')
    parser.add_argument('--trans_mlp_ratio', type=float, help='ratio between mlp hidden dimension of a transformer layer and d_model')
    parser.add_argument('--trans_dropout', type=float, help='dropout rate for transformer backbone')
    # diffusion setting
    # Diffusion = subparsers.add_parser("Diffusion",
    #                                  description='Dataset parser for training on Diffusion',
    #                                  add_help=True,
    #                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    #                                  help="Dataset parser for training on Diffusion")
    # Diffusion.add_argument('--backbone', default='diffusion', type=str, default='mlp', choices=['mlp', 'unet1d', 'transformer'], help='Backbone name')

    return parser.parse_args()