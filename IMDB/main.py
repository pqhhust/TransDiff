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

import gpytorch

from data_loader import get_imdb_data

import warmup_scheduler

wandb.login()

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

    train_loader, val_loader, test_loader, tokenizer = get_imdb_data('./data', args.batch_size)

    for run in range(args.nb_run):
        prefix = '{:d} / {:d} Running'.format(run + 1, args.nb_run)
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, len(tokenizer.vocab), logger, args)
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.to(device)
        ## define optimizer with warm-up
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.min_lr)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
        
        ## make logger
        best_acc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training
        for epoch in range(args.nb_epochs):
            train.train(train_loader, net, optimizer, epoch, logger, args)
            
            scheduler.step()

            # validation
            net_val = net
            res = val.validation(val_loader, net_val, args) 
            log = [key + ': {:.3f}'.format(res[key]) for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)
            wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)

            if res['Acc.'] > best_acc :
                acc = res['Acc.']
                msg = f'ACC improved from {best_acc:.2f} to {acc:.2f}!!!'
                logger.info(msg)
                best_acc = acc
                torch.save(net_val.state_dict(),os.path.join(save_path, f'best_acc_net_{run+1}.pth'))
            
            if res['AUROC'] > best_auroc :
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                # torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}.pth'))
        
            if res['AURC'] < best_aurc :
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                # torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}.pth'))


def main_svdkl(args):
    if args.attn_type == 'softmax':
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}")
        pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_transformer_imdb_{args.pretrained_seed}")
        group = "SVDKL-Transformer-IMDB"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project='Difformer', 
               group=group,
               name=f"Seed_{args.seed}_svdkl",
               config=vars(args))

    # Set seed everything
    set_seed(args.seed)

    logger = utils.utils.get_logger(save_path)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_loader, val_loader, test_loader, tokenizer = get_imdb_data('./data', args.batch_size)

    for run in range(args.nb_run):
        prefix = f'{run + 1} / {args.nb_run} Running'
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, 2, logger, args)
        net.feature_extractor.load_state_dict(torch.load(os.path.join(pretrained_path, f'best_acc_net_{run + 1}.pth')))
        for params in net.feature_extractor.parameters():
            params.requires_grad = False
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.cuda()
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=args.hdim, num_classes=2).cuda()
        ## define optimizer with warm-up
        args.lr = 0.1
        optimizer = torch.optim.SGD([
            {'params': net.feature_extractor.parameters(), 'weight_decay': 1e-4},
            {'params': net.gp_layer.hyperparameters(), 'lr': args.lr * 0.01},
            {'params': net.gp_layer.variational_parameters()},
            {'params': likelihood.parameters()},
        ], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * args.nb_epochs, 0.75 * args.nb_epochs], gamma=0.1)
        mll = gpytorch.mlls.VariationalELBO(likelihood, net.gp_layer, num_data=len(train_loader.dataset))
        
        ## make logger
        best_acc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training
        for epoch in range(args.nb_epochs):
            train_log = {
                'Tot. Loss': utils.utils.AverageMeter(),
                'LR': utils.utils.AverageMeter(),
            }
            msg = '####### --- Training Epoch {:d} --- #######'.format(epoch)
            logger.info(msg)
            
            with gpytorch.settings.use_toeplitz(False):
                net.train()
                likelihood.train()
                
                for i, (data, target) in enumerate(train_loader):
                    data, target = data.cuda(), target.cuda()
                    optimizer.zero_grad()
                    output = net(data)
                    loss = -mll(output, target)
                    loss.backward()
                    optimizer.step()
            
                    for param_group in optimizer.param_groups:
                        lr = param_group["lr"]
                        break

                    train_log['Tot. Loss'].update(loss.item(), data.size(0))
                    train_log['LR'].update(lr, data.size(0))

                    if i % 100 == 99:
                        log = ['LR : {:.5f}'.format(train_log['LR'].avg)] + [
                            key + ': {:.2f}'.format(train_log[key].avg) for key in train_log if key != 'LR'
                        ]
                        msg = 'Epoch {:d} \t Batch {:d}\t'.format(epoch, i) + '\t'.join(log)
                        logger.info(msg)
                        for key in train_log:
                            train_log[key] = utils.utils.AverageMeter()

                # Replace writer.add_scalar with wandb.log
                wandb.log({f"Train/{key}": train_log[key].avg for key in train_log}, step=epoch)
            
            scheduler.step()

            # validation
            net_val = net
            res = val.validation(val_loader, (net_val, likelihood), args, method='svdkl') 
            log = [f"{key}: {res[key]:.3f}" for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)

            wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)

            if res['Acc.'] > best_acc:
                acc = res['Acc.']
                msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
                logger.info(msg)
                best_acc = acc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_acc_net_{run+1}.pth'))
                torch.save(likelihood.state_dict(), os.path.join(save_path, f'best_acc_likelihood_{run+1}.pth'))
            
            if res['AUROC'] > best_auroc:
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                # torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}.pth'))
        
            if res['AURC'] < best_aurc:
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                # torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}.pth')) 

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

    train_loader, val_loader, test_loader, tokenizer = get_imdb_data('./data', args.batch_size)

    for run in range(args.nb_run):
        prefix = '{:d} / {:d} Running'.format(run + 1, args.nb_run)
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, len(tokenizer.vocab), logger, args)
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.cuda()
        pretrained_ViT = models.get_model.get_model('q_distribution', len(tokenizer.vocab), logger, args)
        pretrained_ViT.load_state_dict(torch.load(os.path.join(pretrained_path, f'best_acc_net_{run + 1}.pth')))
        pretrained_ViT.cuda()
        net.embedding.load_state_dict(pretrained_ViT.embedding.state_dict())
        net.pos_encoder.load_state_dict(pretrained_ViT.pos_encoder.state_dict())
        net.ln.load_state_dict(pretrained_ViT.enc[args.depth - 1].la2.state_dict())
        net.solution_head_1.load_state_dict(pretrained_ViT.enc[args.depth - 1].mlp.state_dict())
        net.solution_head_2.load_state_dict(pretrained_ViT.fc.state_dict())
        ## define optimizer with warm-up
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.min_lr)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
        
        ## make logger
        best_acc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training
        for epoch in range(args.nb_epochs):
            train.train_diffusion(train_loader, net, optimizer, epoch, logger, args, pretrained_ViT)
            
            scheduler.step()

            # validation
            net_val = net
            res = val.validation(val_loader, net_val, args) 
            log = [key + ': {:.3f}'.format(res[key]) for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)

            wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)

            if res['Acc.'] > best_acc :
                acc = res['Acc.']
                msg = f'Acc. improved from {best_acc:.2f} to {acc:.2f}!!!'
                logger.info(msg)
                best_acc = acc
                torch.save(net_val.state_dict(),os.path.join(save_path, f'best_acc_net_{run+1}_{args.lambda_mean}_{args.lambda_var}_{args.lambda_ce}.pth'))
            
            if res['AUROC'] > best_auroc :
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                # torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}.pth'))
        
            if res['AURC'] < best_aurc :
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                # torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}.pth'))



if __name__ == '__main__':
    args = utils.train_utils.get_args_parser()
    if args.model == 'diffusion': 
        main_diffusion(args)
        test.test_diffusion(args)
        wandb.finish()
    elif args.model == 'svdkl':
        main_svdkl(args)
        test.test(args)
        wandb.finish()
    else:
        main(args)
        test.test(args)
        wandb.finish()
