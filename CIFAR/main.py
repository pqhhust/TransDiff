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
from utils.seed_utils import set_seed
from utils.ema import EMA

import warmup_scheduler
wandb.login(key='1cfab558732ccb32d573a7276a337d22b7d8b371')#(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')#

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
        group = "VIT"
    elif args.attn_type == 'kep_svgp':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}"
        )
        group = "KEP-SVGP"

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

    train_loader, val_loader, _, nb_cls = datasets.cifar_loader.get_loader(
        args.dataset, args.train_dir, args.val_dir, args.test_dir, args.batch_size
    )

    for run in range(args.nb_run):
        prefix = f'{run + 1} / {args.nb_run} Running'
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        # print(net)
        # print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.cuda()
        
        ## define optimizer with warm-up
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.nb_epochs, eta_min=args.min_lr
        )
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.,
            total_epoch=args.warmup_epoch,
            after_scheduler=base_scheduler
        )
        
        ## make logger
        best_acc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training
        for epoch in range(args.nb_epochs):
            train.train(train_loader, net, optimizer, epoch, logger, args)
            
            scheduler.step()

            # validation
            net_val = net
            res = val.validation(val_loader, net_val, args) 
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
            
            if res['AUROC'] > best_auroc:
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}.pth'))
        
            if res['AURC'] < best_aurc:
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}.pth'))
                
       
def main_diffusion(args):
    if args.attn_type == 'softmax':
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}")
        pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_vit_cifar_{args.pretrained_seed}")
        group = "VIT"
    elif args.attn_type == 'kep_svgp':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}"
        )
        pretrained_path = os.path.join(
            args.pretrained_dir,
            f"{args.dataset}_{args.attn_type}_vit_cifar_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.pretrained_seed}"
        )
        group = "KEP-SVGP"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project='Difformer', 
               group=group,
               name=f"Add cross-entropy loss; mean+std*noise clip; Tuning lr: Diffusion_seed_{args.seed}_lr_{args.lr}_clip_{args.clip}_pretrained_seed_{args.pretrained_seed}_mlp_dropout_{args.mlp_dropout}_ksvd_layers_{args.ksvd_layers}_lambda_mean_{args.lambda_mean}_var_{args.lambda_var}_ce_{args.lambda_ce}_batchsize_{args.batch_size}_architecture_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_epochs_{args.nb_epochs}",
               config=vars(args))

    # Set seed everything
    set_seed(args.seed)

    logger = utils.utils.get_logger(save_path)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_loader, val_loader, test_loader, nb_cls = datasets.cifar_loader.get_loader(
        args.dataset, args.train_dir, args.val_dir, args.test_dir, args.batch_size
    )

    for run in range(args.nb_run):
        prefix = f'{run + 1} / {args.nb_run} Running'
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        # net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{run + 1}_diffusion.pth')))
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.cuda()
        pretrained_ViT = models.get_model.get_model('q_distribution', nb_cls, logger, args)
        pretrained_ViT.load_state_dict(torch.load(os.path.join(pretrained_path, f'best_acc_net_{run + 1}.pth')))
        pretrained_ViT.cuda()
        net.emb.load_state_dict(pretrained_ViT.emb.state_dict())
        net.pos_emb.data.copy_(pretrained_ViT.pos_emb.data)
        net.ln.load_state_dict(pretrained_ViT.enc[args.depth - 1].la2.state_dict())
        net.solution_head_1.load_state_dict(pretrained_ViT.enc[args.depth - 1].mlp.state_dict())
        net.solution_head_2.load_state_dict(pretrained_ViT.fc.state_dict())
        
        ## define optimizer with warm-up
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.nb_epochs, eta_min=args.min_lr
        )
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.,
            total_epoch=args.warmup_epoch,
            after_scheduler=base_scheduler
        )

        ## Initialize EMA
        # ema = EMA(args.ema_decay)
        # ema.register(net)
        
        ## make logger
        best_acc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training
        for epoch in range(args.nb_epochs):
            train.train_diffusion(train_loader, net, optimizer, epoch, logger, args, pretrained_ViT)

            # if epoch % args.update_ema_interval == 0:
            #     step_ema(args, ema, net, epoch)
            
            #validate

            scheduler.step()

            # validation
            # if epoch % args.update_ema_interval == 0:
            #     apply_ema(args, ema, net)
            net_val = net
            res = val.validation_diffusion(val_loader, net_val, args, pretrained_ViT) 
            test_results = val.validation_diffusion(test_loader, net_val, args, pretrained_ViT)
            # if epoch % args.update_ema_interval == 0:
            #     restore_ema(args, ema, net)
            log = [f"{key}: {res[key]:.3f}" for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)

            wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)
            log = [f"{key}: {test_results[key]:.3f}" for key in test_results]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)
            wandb.log({f"Test/{key}": test_results[key] for key in test_results}, step=epoch)

            if res['Acc.'] > best_acc:
                acc = res['Acc.']
                msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
                logger.info(msg)
                best_acc = acc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_acc_net_{run+1}_diffusion_{args.backbone}.pth'))
                # torch.save(pretrained_ViT.state_dict(), os.path.join(save_path, f'best_acc_net_{run + 1}_vit_fc.pth'))
            
            if res['AUROC'] > best_auroc:
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}_diffusion_{args.backbone}.pth'))
        
            if res['AURC'] < best_aurc:
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}_diffusion_{args.backbone}.pth'))
        
        torch.save(net.state_dict(), os.path.join(save_path, f'last_net_{run+1}_diffusion_{args.backbone}.pth'))


def main_diffusion_stage2(args):
    if args.attn_type == 'softmax':
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}")
        pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_{args.model}")
    elif args.attn_type == 'kep_svgp':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_vit_cifar_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}"
        )
        pretrained_path = os.path.join(
            args.pretrained_dir,
            f"{args.dataset}_{args.attn_type}_vit_cifar_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}"
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project='Diffusion-KEP-SVGP', config=vars(args))

    # Set seed everything
    set_seed(args.seed)

    logger = utils.utils.get_logger(save_path)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_loader, val_loader, _, nb_cls = datasets.cifar_loader.get_loader(
        args.dataset, args.train_dir, args.val_dir, args.test_dir, args.batch_size
    )

    for run in range(args.nb_run):
        prefix = f'{run + 1} / {args.nb_run} Running'
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{run + 1}_diffusion_{args.backbone}.pth')))
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.cuda()
        pretrained_ViT = models.get_model.get_model('q_distribution', nb_cls, logger, args)
        pretrained_ViT.load_state_dict(torch.load(os.path.join(pretrained_path, f'best_acc_net_{run + 1}.pth')))
        pretrained_ViT.cuda()

        ## define optimizer with warm-up
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.nb_epochs, eta_min=args.min_lr
        )
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.,
            total_epoch=args.warmup_epoch,
            after_scheduler=base_scheduler
        )

        ## make logger
        best_acc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training

        for epoch in range(args.nb_epochs):
            train.train_diffusion_stage2(train_loader, net, optimizer, epoch, logger, args, pretrained_ViT)
            scheduler.step()
            # optimizer.step()

            # validation
            net_val = net
            res = val.validation_diffusion(val_loader, net_val, args, pretrained_ViT)
            log = [f"{key}: {res[key]:.3f}" for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)

            wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)

            if res['Acc.'] > best_acc:
                acc = res['Acc.']
                msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
                logger.info(msg)
                best_acc = acc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_acc_net_{run+1}_diffusion_{args.backbone}.pth'))
                # torch.save(pretrained_ViT.state_dict(), os.path.join(save_path, f'best_acc_net_{run + 1}_vit_fc.pth'))

            if res['AUROC'] > best_auroc:
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}_diffusion_{args.backbone}.pth'))

            if res['AURC'] < best_aurc:
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}_diffusion_{args.backbone}.pth'))


if __name__ == '__main__':
    args = utils.train_utils.get_args_parser()
    if args.model == 'diffusion':
        main_diffusion(args)
        test.test_diffusion(args)
        wandb.finish()
    # elif args.model == 'diffusion' and args.stage == 2:
    #     main_diffusion_stage2(args)
    else:
        main(args)
        test.test(args)
        wandb.finish()
