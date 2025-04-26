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
import utils.train_utils
from utils.seed_utils import set_seed
from data_loader import get_data, get_vocab, DataLoader

import warmup_scheduler
# wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')#
wandb.login(key='1cfab558732ccb32d573a7276a337d22b7d8b371')#


def main(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.attn_type == 'softmax':
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}")
        group = "Transformer-CoLA"
    elif args.attn_type == 'kep_svgp':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}"
        )
        group = "KEP-SVGP-CoLA"
    elif args.attn_type == 'sgpa':
        save_path = os.path.join(
            args.save_dir,
            f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}"
        )
        group = "SGPA-CoLA"

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

    data_train,gold_train,data_test,gold_test,data_ood,gold_ood=\
            get_data(['./data/cola_public/raw/in_domain_train.tsv','./data/cola_public/raw/in_domain_dev.tsv'],['./data/cola_public/raw/out_of_domain_dev.tsv'], args.seed)
    word_to_int, _ = get_vocab(data_train, args.min_word_count)
    vocab_size = len(word_to_int)

    train_loader = DataLoader(data_train,gold_train,args.batch_size,word_to_int,device)
    test_loader = DataLoader(data_test,gold_test,args.batch_size,word_to_int,device,shuffle=False)

    for run in range(args.nb_run):
        prefix = '{:d} / {:d} Running'.format(run + 1, args.nb_run)
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, vocab_size, logger, args)
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.to(device)
        ## define optimizer with warm-up
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.min_lr)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
        
        ## make logger
        best_mcc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training
        for epoch in range(args.nb_epochs):
            train.train(train_loader, net, optimizer, epoch, logger, args)
            
            scheduler.step()

            # validation
            net_val = net
            res = val.validation(test_loader, net_val, args) 
            log = [key + ': {:.3f}'.format(res[key]) for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)

            wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)

            if res['MCC'] > best_mcc :
                mcc = res['MCC']
                msg = f'MCC improved from {best_mcc:.2f} to {mcc:.2f}!!!'
                logger.info(msg)
                best_mcc = mcc
                torch.save(net_val.state_dict(),os.path.join(save_path, f'best_mcc_net_{run+1}.pth'))
            
            if res['AUROC'] > best_auroc :
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}.pth'))
        
            if res['AURC'] < best_aurc :
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}.pth'))

def main_diffusion(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    if args.attn_type == 'softmax':
        if args.backbone == 'mlp':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}")
        elif args.backbone == 'lstm' or args.backbone == 'gru':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.rnn_hidden}_{args.rnn_num_layers}_{args.rnn_dropout}_{args.rnn_low_dim}_{args.lr}_{args.nb_epochs}")
        elif args.backbone == 'transformer':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.trans_depth}_{args.trans_num_heads}_{args.trans_mlp_ratio}_{args.trans_dropout}_{args.lr}_{args.nb_epochs}")
        pretrained_path = os.path.join(args.pretrained_dir, f"{args.dataset}_{args.attn_type}_vit_cola_{args.pretrained_seed}")
        group = "Transformer-DiT-CoLA"
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
            f"{args.dataset}_{args.attn_type}_vit_cola_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.pretrained_seed}"
        )
        group = "KEP-SVGP-DiT-CoLA"

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

    data_train,gold_train,data_test,gold_test,data_ood,gold_ood=\
            get_data(['./data/cola_public/raw/in_domain_train.tsv','./data/cola_public/raw/in_domain_dev.tsv'],['./data/cola_public/raw/out_of_domain_dev.tsv'], args.seed)
    word_to_int, _ = get_vocab(data_train, args.min_word_count)
    vocab_size = len(word_to_int)

    train_loader = DataLoader(data_train,gold_train,args.batch_size,word_to_int,device)
    test_loader = DataLoader(data_test,gold_test,args.batch_size,word_to_int,device,shuffle=False)

    for run in range(args.nb_run):
        prefix = '{:d} / {:d} Running'.format(run + 1, args.nb_run)
        logger.info(100*'#' + '\n' + prefix)

        ## define model
        net = models.get_model.get_model(args.model, vocab_size, logger, args)
        print(net)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))
        net.cuda()
        pretrained_ViT = models.get_model.get_model('q_distribution', vocab_size, logger, args)
        pretrained_ViT.load_state_dict(torch.load(os.path.join(pretrained_path, f'best_mcc_net_{run + 1}.pth')))
        pretrained_ViT.cuda()
        net.embedding.load_state_dict(pretrained_ViT.embedding.state_dict())
        net.ln.load_state_dict(pretrained_ViT.enc[args.depth - 1].la2.state_dict())
        net.solution_head_1.load_state_dict(pretrained_ViT.enc[args.depth - 1].mlp.state_dict())
        net.solution_head_2.load_state_dict(pretrained_ViT.fc.state_dict())
        ## define optimizer with warm-up
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.min_lr)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)
        
        ## make logger
        best_mcc, best_auroc, best_aurc = 0, 0, 1e6

        ## start training
        for epoch in range(args.nb_epochs):
            train.train_diffusion(train_loader, net, optimizer, epoch, logger, args, pretrained_ViT)
            
            scheduler.step()

            # validation
            net_val = net
            res = val.validation(test_loader, net_val, args) 
            log = [key + ': {:.3f}'.format(res[key]) for key in res]
            msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            logger.info(msg)

            wandb.log({f"Val/{key}": res[key] for key in res}, step=epoch)

            # test_results = val.validation(test_loader, net_val, args)
            # log = [f"{key}: {test_results[key]:.3f}" for key in test_results]
            # msg = '################## \n ---> Validation Epoch {:d}\t'.format(epoch) + '\t'.join(log)
            # logger.info(msg)
            # wandb.log({f"Test/{key}": test_results[key] for key in test_results}, step=epoch)

            if res['MCC'] > best_mcc :
                mcc = res['MCC']
                msg = f'MCC improved from {best_mcc:.2f} to {mcc:.2f}!!!'
                logger.info(msg)
                best_mcc = mcc
                torch.save(net_val.state_dict(),os.path.join(save_path, f'best_mcc_net_{run+1}.pth'))
            
            if res['AUROC'] > best_auroc :
                auroc = res['AUROC']
                msg = f'AUROC improved from {best_auroc:.2f} to {auroc:.2f}!!!'
                logger.info(msg)
                best_auroc = auroc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_auroc_net_{run+1}.pth'))
        
            if res['AURC'] < best_aurc :
                aurc = res['AURC']
                msg = f'AURC decreased from {best_aurc:.2f} to {aurc:.2f}!!!'
                logger.info(msg)
                best_aurc = aurc
                torch.save(net_val.state_dict(), os.path.join(save_path, f'best_aurc_net_{run+1}.pth'))



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
