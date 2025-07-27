import torch
import torch.nn as nn
import torch.backends.cudnn
import wandb

import os
import json 

import train
import val
import test

import models.get_model
import datasets.cifar_loader
import utils.train_utils
# import gpytorch
from utils.seed_utils import set_seed
from utils.ema import EMA
import utils.utils

import warmup_scheduler
wandb.login(key='1cfab558732ccb32d573a7276a337d22b7d8b371')
# wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')

def step_ema(args, ema, net, epoch):
        with_decay = False if epoch < args.start_ema_step else True
        ema.update(net, with_decay=with_decay)
def apply_ema(args, ema, net):
    if args.use_ema:
        ema.apply_shadow(net)
def restore_ema(args, ema, net):
    if args.use_ema:
        ema.restore(net)

def main(args):
    if args.attn_type == 'softmax':
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}")
        group = "VIT-CIFAR"
    elif args.attn_type == 'kep_svgp':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}"
        )
        group = "KEP-SVGP-CIFAR"
    elif args.attn_type == 'sgpa':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}"
        )
        group = "SGPA-CIFAR"
    elif args.attn_type == 'cgpt':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}"
        )
        group = "CGPT-CIFAR"
    elif args.attn_type == 'scgpt':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}"
        )
        group = "SCGPT-CIFAR"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project='Difformer', 
               group=group,
               name=f"Seed_{args.seed}",
               config=vars(args))

    # Set seed everything
    set_seed(42)

    logger = utils.utils.get_logger(save_path)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
                
      
def main_diffusion(args):
    if args.attn_type == 'softmax':
        if args.backbone == 'mlp':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}")
        elif args.backbone == 'lstm' or args.backbone == 'gru':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.rnn_hidden}_{args.rnn_num_layers}_{args.rnn_dropout}_{args.rnn_low_dim}_{args.lr}_{args.nb_epochs}")
        elif args.backbone == 'transformer':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.trans_depth}_{args.trans_num_heads}_{args.trans_mlp_ratio}_{args.trans_dropout}_{args.lr}_{args.nb_epochs}")
        pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_vit_cifar_{args.pretrained_seed}")
        group = "VIT-DiT"
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
            f"{args.dataset}_{args.attn_type}_vit_cifar_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.pretrained_seed}"
        )
        group = "KEP-SVGP-DiT"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project='Difformer', 
               group=group,
               name=f"Diffusion {args.run_name}: seed_{args.seed}_lr_{args.lr}_pretrained_seed_{args.pretrained_seed}_ksvd_layers_{args.ksvd_layers}_lambda_mean_{args.lambda_mean}_var_{args.lambda_var}_ce_{args.lambda_ce}_batchsize_{args.batch_size}_epochs_{args.nb_epochs}",
            #    name=f"Diffusion {args.run_name}: seed_{args.seed}_lr_{args.lr}_clip_{args.clip}_pretrained_seed_{args.pretrained_seed}_mlp_dropout_{args.mlp_dropout}_ksvd_layers_{args.ksvd_layers}_lambda_mean_{args.lambda_mean}_var_{args.lambda_var}_ce_{args.lambda_ce}_batchsize_{args.batch_size}_architecture_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_epochs_{args.nb_epochs}_adversarial_noise_{args.adversarial_noise}_adversarial_samples_{args.adversarial_samples}_rnn_hidden_{args.rnn_hidden}_rnn_num_layers_{args.rnn_num_layers}_rnn_dropout_{args.rnn_dropout}",
               config=vars(args))

    # Set seed everything
    set_seed(42)

    logger = utils.utils.get_logger(save_path)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        
if __name__ == '__main__':
    args = utils.train_utils.get_args_parser()
    if args.model == 'diffusion':
        main_diffusion(args)
        test.test_diffusion(args)
        wandb.finish()
    else:
        main(args)
        test.test(args)
        wandb.finish()
