import torch
import val_ood
import os
import utils.test_utils
import datasets.cifar_loader
import datasets.CIFARC
import utils.utils
from utils.seed_utils import set_seed
import models.get_model
import csv
from torch.utils.data import DataLoader
import torchvision.transforms
import wandb
import gpytorch
from laplace import Laplace

def process_results_ood(args, loader, ood_loader, model, metrics, logger, method_name, results_storage):
    res = val_ood.validation_ood(loader, ood_loader, model, args)
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.3f}" for key in res]
    logger.info(f'################## \n ---> Test {method_name} results：\t' + '\t'.join(log))

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['softmax/AUROC', 'softmax/AUPR', 'softmax/FPR95', 'entropy/AUROC', 'entropy/AUPR', 'entropy/FPR95', 'AUROC', 'AUPR', 'FPR95']
    results_storage = {metric: [] for metric in metrics}

    if args.attn_type == 'sgpa':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_' + str(args.seed)
    if args.attn_type == 'softmax':
        args_model = 'vit_cifar' if args.model == 'temperature_scaling' or args.model == 'mc_dropout' or args.model == 'kflla' else args.model
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args_model + '_' + str(args.seed)
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl) + '_' + str(args.seed)
    logger = utils.utils.get_logger(save_path)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        train_loader, valid_loader, test_loader, nb_cls = datasets.cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        ood_test_loader = datasets.cifar_loader.get_ood_loader(args.ood_data,
                                                            args.ood_test_dir,
                                                            args.batch_size)                                                               
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}.pth')))
        net = net.cuda()
        if args.model == 'svdkl':
            # pass
            likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=args.hdim, num_classes=args.nb_cls).cuda()
            likelihood.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_likelihood_{r + 1}.pth')))
            net = (net, likelihood) 
        if args.model == "kflla":
            net.train()
            la = Laplace(net, 'classification', subset_of_weights='last_layer', hessian_structure='kron')
            with torch.enable_grad():
                la.fit(train_loader)
                la.optimize_prior_precision(method='marglik')
            net.eval()
            net = la
        process_results_ood(args, test_loader, ood_test_loader, net, metrics, logger, "MSP", results_storage)

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    wandb.log({f"Test_final/{metric}": results[metric]['mean'] for metric in results})
    test_results_path = os.path.join(save_path, 'test_results.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)

def test_diffusion(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['softmax/AUROC', 'softmax/AUPR', 'softmax/FPR95', 'entropy/AUROC', 'entropy/AUPR', 'entropy/FPR95', 'AUROC', 'AUPR', 'FPR95']
    results_storage = {metric: [] for metric in metrics}
    cor_results_all_models = {}

    if args.attn_type == 'softmax':
        if args.backbone == 'mlp':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}")
        elif args.backbone == 'lstm' or args.backbone == 'gru':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.rnn_hidden}_{args.rnn_num_layers}_{args.rnn_dropout}_{args.rnn_low_dim}_{args.lr}_{args.nb_epochs}")
        elif args.backbone == 'transformer':
            save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.attn_type}_{args.model}_{args.seed}_{args.backbone}_{args.trans_depth}_{args.trans_num_heads}_{args.trans_mlp_ratio}_{args.trans_dropout}_{args.lr}_{args.nb_epochs}")

    elif args.attn_type == 'kep_svgp':
        if args.backbone == 'mlp':
            save_path = os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.attn_type}_{args.model}_ksvdlayer{args.ksvd_layers}_ksvd{args.eta_ksvd}_kl{args.eta_kl}_{args.seed}_{args.backbone}_{args.mlp_hdim1}_{args.mlp_hdim2}_{args.mlp_hdim3}_{args.mlp_dropout}_{args.lr}_{args.clip}_{args.nb_epochs}"
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

    logger = utils.utils.get_logger(save_path)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        _, valid_loader, test_loader, nb_cls = datasets.cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        ood_test_loader = datasets.cifar_loader.get_ood_loader(args.ood_data,
                                                            args.ood_test_dir,
                                                            args.batch_size)
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        pretrained_ViT = None
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}_diffusion_{args.backbone}.pth')))
        net = net.cuda()
        process_results_ood(args, test_loader, ood_test_loader, net, metrics, logger, "MSP", results_storage)

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    wandb.log({f"Test_final/{metric}": results[metric]['mean'] for metric in results})
    test_results_path = os.path.join(save_path, 'test_results_diffusion.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
