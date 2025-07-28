import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data
import wandb

import os
import json 
import random

import train
import val
import test

import models.get_model
import utils.train_utils
from utils.seed_utils import set_seed

# from transformers import BertTokenizer
# from torchtext.legacy import data
# from torchtext import datasets

# import gpytorch

from data_loader import get_imdb_data

import warmup_scheduler
# wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')#
wandb.login(key='1cfab558732ccb32d573a7276a337d22b7d8b371')#


def main(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.attn_type == 'softmax':
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}")
        group = "Transformer-IMDB"
    elif args.attn_type == 'kep_svgp':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}"
        )
        group = "KEP-SVGP-IMDB"
    elif args.attn_type == 'sgpa':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}"
        )
        group = "SGPA-IMDB"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project='Difformer', 
               group=group,
               name=f"Seed_{args.seed}",
               config=vars(args))

    # Set seed everything
    set_seed(args.seed)

    logger = utils.utils.get_logger(save_path)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main_diffusion(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.attn_type == 'softmax':
        if args.backbone == 'mlp':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}")
        elif args.backbone == 'lstm' or args.backbone == 'gru':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.rnn_hidden}_{args.rnn_num_layers}_{args.rnn_dropout}_{args.rnn_low_dim}_{args.lr}_{args.nb_epochs}")
        elif args.backbone == 'transformer':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.trans_depth}_{args.trans_num_heads}_{args.trans_mlp_ratio}_{args.trans_dropout}_{args.lr}_{args.nb_epochs}")
        pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_transformer_imdb_{args.pretrained_seed}")
        group = "Transformer-DiT-IMDB"
    elif args.attn_type == 'kep_svgp':
        if args.backbone == 'mlp':
            save_path = os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.rnn_low_dim}_{args.lr}_{args.clip}_{args.nb_epochs}"
            )
        elif args.backbone == 'lstm' or args.backbone == 'gru':
            save_path = os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.rnn_hidden}_{args.rnn_num_layers}_{args.rnn_dropout}_{args.rnn_low_dim}_{args.lr}_{args.nb_epochs}"
            )
        elif args.backbone == 'transformer':
            save_path = os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.trans_depth}_{args.trans_num_heads}_{args.trans_mlp_ratio}_{args.trans_dropout}_{args.lr}_{args.nb_epochs}"
            )
        pretrained_path = os.path.join(
            args.pretrained_dir,
            f"{args.dataset}_{args.attn_type}_transformer_imdb_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.pretrained_seed}"
        )
        group = "KEP-SVGP-DiT-IMDB"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project='Difformer', 
               group=group,
               name=f"Change batch-size: Diffusion_seed_{args.seed}_lr_{args.lr}_clip_{args.clip}_pretrained_seed_{args.pretrained_seed}_mlp_dropout_{args.mlp_dropout}_ksvd_layers_{args.ksvd_layers}_gamma_{args.mlp_gamma}",
               config=vars(args))

    # Set seed everything
    set_seed(args.seed)

    logger = utils.utils.get_logger(save_path)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    args = utils.train_utils.get_args_parser()
    if args.model == 'diffusion': 
        main_diffusion(args)
        test.test_diffusion(args)
        wandb.finish()
    # elif args.model == 'diffusion' and args.stage == 2:
    #     main_diffusion_stage2(args)
    # elif args.model == 'svdkl':
    #     main_svdkl(args)
    #     test.test(args)
    #     wandb.finish()
    else:
        main(args)
        test.test(args)
        wandb.finish()
