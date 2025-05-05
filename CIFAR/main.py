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
import datasets.imagenet_loader
import utils.train_utils
import utils.seed_utils
from utils.seed_utils import set_seed
from utils.ema import EMA

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

import gc
import warmup_scheduler

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_TIMEOUT"] = "900"
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
    # Determine save path based on attention type and backbone
    if args.attn_type == 'softmax':
        if args.backbone == 'mlp':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}")
        elif args.backbone == 'lstm' or args.backbone == 'gru':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.rnn_hidden}_{args.rnn_num_layers}_{args.rnn_dropout}_{args.rnn_low_dim}_{args.lr}_{args.nb_epochs}")
        elif args.backbone == 'transformer':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.trans_depth}_{args.trans_num_heads}_{args.trans_mlp_ratio}_{args.trans_dropout}_{args.lr}_{args.nb_epochs}")
        # pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_vit_cifar_{args.pretrained_seed}")
        group = "ViT-b-16-cifar"
    elif args.attn_type == 'kep_svgp':
        if args.backbone == 'mlp':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.rnn_low_dim}_{args.lr}_{args.clip}_{args.nb_epochs}")
        elif args.backbone == 'lstm' or args.backbone == 'gru':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.rnn_hidden}_{args.rnn_num_layers}_{args.rnn_dropout}_{args.rnn_low_dim}_{args.lr}_{args.nb_epochs}")
        elif args.backbone == 'transformer':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.trans_depth}_{args.trans_num_heads}_{args.trans_mlp_ratio}_{args.trans_dropout}_{args.lr}_{args.nb_epochs}")
        # pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_vit_cifar_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.pretrained_seed}")
        group = "KEP-SVGP-DiT"

    # Get rank information from environment variables set by torchrun
    local_rank = int(os.environ['LOCAL_RANK'])  # Rank within the local node
    global_rank = int(os.environ['RANK'])       # Global rank across all nodes

    # Initialize distributed process group with NCCL backend for GPU communication
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)  # Set the current device to the local rank's GPU
    
    # Create save directory only on rank 0 to avoid race conditions
    if global_rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Initialize Weights & Biases logging only on rank 0
    if global_rank == 0:
        # wandb.login(key='1cfab558732ccb32d573a7276a337d22b7d8b371')
        # wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')
        wandb.init(project='Difformer', 
                   group=group,
                   name=f"Diffusion {args.run_name}: seed_{args.seed}_lr_{args.lr}_pretrained_seed_{args.pretrained_seed}_ksvd_layers_{args.ksvd_layers}_lambda_mean_{args.lambda_mean}_var_{args.lambda_var}_ce_{args.lambda_ce}_batchsize_{args.batch_size}_epochs_{args.nb_epochs}",
                   config=vars(args))

    # Set random seed for reproducibility
    utils.seed_utils.set_seed(args.seed)

    # Initialize logger only on rank 0
    logger = utils.utils.get_logger(save_path) if global_rank == 0 else None
    if global_rank == 0:
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # Load data with DistributedSampler for distributed training
    if args.dataset == 'imagenet1k':
        train_loader, val_loader, test_loader, nb_cls = datasets.imagenet_loader.get_loader(
            args.dataset, args.train_dir, args.val_dir, args.test_dir, args.batch_size
        )
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=dist.get_world_size(), rank=global_rank, shuffle=True)
        val_sampler = DistributedSampler(val_loader.dataset, num_replicas=dist.get_world_size(), rank=global_rank, shuffle=False)
        train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, drop_last=True)
        if global_rank == 0:
            val_loader = DataLoader(val_loader.dataset, batch_size=args.batch_size, sampler=None, num_workers=8, drop_last=False)
    else:
        train_loader, val_loader, test_loader, nb_cls = datasets.cifar_loader.get_loader(
            args.dataset, args.train_dir, args.val_dir, args.test_dir, args.batch_size
        )
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=dist.get_world_size(), rank=global_rank, shuffle=True)
        # val_sampler = DistributedSampler(val_loader.dataset, num_replicas=dist.get_world_size(), rank=global_rank, shuffle=False)
        train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, drop_last=True)
        if global_rank == 0:
            val_loader = DataLoader(val_loader.dataset, batch_size=args.batch_size, sampler=None, num_workers=8, drop_last=False)
            test_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, sampler=None, num_workers=8, drop_last=False)

    for run in range(args.nb_run):
        if global_rank == 0:
            prefix = f'{run + 1} / {args.nb_run} Running'
            logger.info(100*'#' + '\n' + prefix)

        # Define and initialize the model
        pretrained_ViT = models.get_model.get_model('q_distribution_imagenet', nb_cls, logger, args)
        net = models.get_model.get_model(args.model, nb_cls, logger, (args, pretrained_ViT.config))
        net = net.cuda()
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])  # Wrap model with DDP

        # Print model info on rank 0
        if global_rank == 0:
            print(net)
            print(sum(p.numel() for p in net.parameters() if p.requires_grad))

        pretrained_ViT = nn.parallel.DistributedDataParallel(pretrained_ViT.cuda(), device_ids=[local_rank])

        # Define optimizer and scheduler
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        if args.warmup_epoch > 0:
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.min_lr / args.lr, end_factor=1.0, total_iters=args.warmup_epoch)
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs - args.warmup_epoch, eta_min=args.min_lr)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warmup_epoch])
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.min_lr)
        # scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)

        # Track best metrics
        best_acc, best_auroc, best_aurc = 0, 0, 1e6
        start_epoch = 0
        if args.resume_weights:
            weights_checkpoint = torch.load(os.path.join(save_path, args.resume_weights), map_location='cpu', weights_only=True)
            net.module.load_state_dict(weights_checkpoint)
            
            training_state_checkpoint = torch.load(os.path.join(save_path, args.resume_training_state), map_location='cpu')
            optimizer.load_state_dict(training_state_checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(training_state_checkpoint['lr_scheduler_state_dict'])
            start_epoch = training_state_checkpoint['epoch'] + 1
            # logger.info(f"Resuming training from epoch {start_epoch}...")
        else:
            net.module.embedding.load_state_dict(pretrained_ViT.module.model.vit.embeddings.state_dict())
            net.module.intermediate.load_state_dict(pretrained_ViT.module.model.vit.encoder.layer[-1].intermediate.state_dict())
            net.module.output.load_state_dict(pretrained_ViT.module.model.vit.encoder.layer[-1].output.state_dict())
            net.module.layernorm_after.load_state_dict(pretrained_ViT.module.model.vit.encoder.layer[-1].layernorm_after.state_dict())
            net.module.layernorm.load_state_dict(pretrained_ViT.module.model.vit.layernorm.state_dict())
            net.module.classifier.load_state_dict(pretrained_ViT.module.model.classifier.state_dict())
        # Training loop over epochs
        for epoch in range(start_epoch, args.nb_epochs):
            # Set epoch for sampler to ensure shuffling is consistent across processes
            if dist.get_world_size() > 1:
                train_sampler.set_epoch(epoch)

            # Train the model
            train.train_diffusion(train_loader, net, optimizer, epoch, logger, args, pretrained_ViT)
            lr_scheduler.step()
            # Validate the model
            # res = val.validation_diffusion(val_loader, net, args, pretrained_ViT)
            if global_rank == 0:
                # Save the last model state on rank 0
                torch.save(net.module.state_dict(), os.path.join(save_path, f'last_net_{run+1}_diffusion_{args.backbone}_tuning_{args.lambda_mean}.pth'))
                training_state_checkpoint = {'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'lr_scheduler_state_dict': lr_scheduler.state_dict()}
                torch.save(training_state_checkpoint, os.path.join(save_path, f'training_state_{run+1}_last_diffusion_{args.backbone}_tuning_{args.lambda_mean}.pth'))
                    
                res = val.validation_diffusion(val_loader, net, args, pretrained_ViT)
                log = [f"{key}: {res[key]:.3f}" for key in res]
                msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
                logger.info(msg)
                wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)
                
                test_results = val.validation_diffusion(test_loader, net, args, pretrained_ViT)
                # if epoch % args.update_ema_interval == 0:
                #     restore_ema(args, ema, net)
        
                log = [f"{key}: {test_results[key]:.3f}" for key in test_results]
                msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
                logger.info(msg)
                wandb.log({f"Test/{key}": test_results[key] for key in test_results}, step=epoch)

                # Save best models based on metrics
                if res['Acc.'] > best_acc:
                    acc = res['Acc.']
                    msg = f'Accuracy improved from {best_acc:.2f} to {acc:.2f}!!!'
                    logger.info(msg)
                    best_acc = acc
                    torch.save(net.module.state_dict(), os.path.join(save_path, f'best_acc_net_{run+1}_diffusion_{args.backbone}.pth'))

                if res['AUROC'] > best_auroc:
                    auroc = res['AUROC']
                    msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                    logger.info(msg)
                    best_auroc = auroc
                    # torch.save(net.module.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}_diffusion_{args.backbone}.pth'))

                if res['AURC'] < best_aurc:
                    aurc = res['AURC']
                    msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                    logger.info(msg)
                    best_aurc = aurc
                    # torch.save(net.module.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}_diffusion_{args.backbone}.pth'))
            dist.barrier()    
            # torch.cuda.empty_cache()
            # gc.collect()
    # Clean up distributed process group
    dist.destroy_process_group()

    # Finish wandb logging on rank 0
    if global_rank == 0:
        wandb.finish()


if __name__ == '__main__':
    args = utils.train_utils.get_args_parser()
    if args.model == 'diffusion':
        main_diffusion(args)
    else:
        main(args)
        test.test(args)
        wandb.finish()
