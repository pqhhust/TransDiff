import torch
import val
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

def process_results(args, loader, model, metrics, logger, method_name, results_storage):
    res = val.validation(loader, model, args, method_name)
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.3f}" for key in res]
    logger.info(f'################## \n ---> Test {method_name} results：\t' + '\t'.join(log))

def process_results_ood(args, loader, ood_loader, model, metrics, logger, method_name, results_storage):
    res = val.validation_ood(loader, ood_loader, model, args)
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.3f}" for key in res]
    logger.info(f'################## \n ---> Test {method_name} results：\t' + '\t'.join(log))

def process_results_diffusion(args, loader, model, metrics, logger, method_name, results_storage, vit_model):
    res = val.validation_diffusion(loader, model, args, vit_model)
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.3f}" for key in res]
    logger.info(f'################## \n ---> Test {method_name} results：\t' + '\t'.join(log))


def test_cifar_c_corruptions(dataset, model, corruption_dir, transform_test, batch_size, metrics, logger, args):
    if dataset == "cifar10":
        cor_results_storage = {corruption: {severity: {metric: [] for metric in metrics} for severity in range(1, 6)} for
                           corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets}
        for corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets:
            for severity in range(1, 6):
                logger.info(f"Testing on corruption: {corruption}, severity: {severity}")
                corrupted_test_dataset = datasets.CIFARC.CIFAR10C(root=corruption_dir, transform=transform_test, subset=corruption,
                                                            severity=severity, download=True)
                corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=4, drop_last=False)
                res = val.validation(corrupted_test_loader, model, args)
                for metric in metrics:
                    cor_results_storage[corruption][severity][metric].append(res[metric])

    return cor_results_storage

def test_cifar_c_corruptions_diffusion(dataset, model, corruption_dir, transform_test, batch_size, metrics, logger, vit_model, args):
    if dataset == "cifar10":
        cor_results_storage = {corruption: {severity: {metric: [] for metric in metrics} for severity in range(1, 6)} for
                           corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets}
        for corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets:
            for severity in range(1, 6):
                logger.info(f"Testing on corruption: {corruption}, severity: {severity}")
                corrupted_test_dataset = datasets.CIFARC.CIFAR10C(root=corruption_dir, transform=transform_test, subset=corruption,
                                                            severity=severity, download=True)
                corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=4, drop_last=False)
                res = val.validation_diffusion(corrupted_test_loader, model, args, vit_model)
                for metric in metrics:
                    cor_results_storage[corruption][severity][metric].append(res[metric])

    return cor_results_storage

def ood_test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['AUROC', 'AUPR', 'FPR95']
    results_storage = {metric: [] for metric in metrics}
    
    if args.attn_type == 'softmax':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_' + str(args.seed)
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl) + '_' + str(args.seed)

    logger = utils.utils.get_logger(save_path)

    for r in range(args.nb_run):
        logger.info(f'Testing model {r + 1} ...')

        _, _, test_loader, nb_cls = datasets.cifar_loader.get_loader(args.dataset, 
                                                            args.train_dir, 
                                                            args.val_dir, 
                                                            args.test_dir, 
                                                            args.batch_size)
        print(nb_cls)
        _, _, ood_test_loader, _ = datasets.cifar_loader.get_loader(args.ood_data,
                                                                args.ood_train_dir,
                                                                args.ood_val_dir,
                                                                args.ood_test_dir,
                                                                args.batch_size)
        
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}.pth')))
        net = net.cuda()

        process_results_ood(args, test_loader, ood_test_loader, net, metrics, logger, "MSP", results_storage)


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}
    # cor_results_all_models = {}

    if args.attn_type == 'sgpa':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_' + str(args.seed)
    if args.attn_type == 'softmax':
        args_model = 'vit_cifar' if args.model == 'temperature_scaling' or args.model == 'mc_dropout' else args.model
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args_model + '_' + str(args.seed)
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl) + '_' + str(args.seed)
    logger = utils.utils.get_logger(save_path)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        _, valid_loader, test_loader, nb_cls = datasets.cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        print(nb_cls)
        net = models.get_model.get_model('q_distribution', nb_cls, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}.pth')))
        net = net.cuda()
        process_results(args, test_loader, net, metrics, logger, "MSP", results_storage)

        # if args.dataset == 'cifar10':
        #     transform_test = torchvision.transforms.Compose([
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #     ])

        #     cor_results_storage = test_cifar_c_corruptions(args.dataset, net, args.corruption_dir, transform_test, args.batch_size,
        #                                                     metrics, logger, args)
        #     cor_results = {corruption: {
        #         severity: {metric: cor_results_storage[corruption][severity][metric][0] for metric in metrics} for severity
        #         in range(1, 6)} for corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets}
        #     cor_results_all_models[f"model_{r + 1}"] = cor_results

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    wandb.log({f"Test_final/{metric}": results[metric]['mean'] for metric in results})
    test_results_path = os.path.join(save_path, 'test_results.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    # if args.dataset == 'cifar10':
    #     utils.utils.save_cifar_c_results_to_csv(args.dataset, args.attn_type, save_path, metrics, cor_results_all_models)

def test_diffusion(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}
    # cor_results_all_models = {}

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
        print(nb_cls)
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        pretrained_ViT = None
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}_diffusion_{args.backbone}.pth')))
        net = net.cuda()
        process_results_diffusion(args, test_loader, net, metrics, logger, "MSP", results_storage, pretrained_ViT)

        # if args.dataset == 'cifar10':
        #     transform_test = torchvision.transforms.Compose([
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #     ])

        #     cor_results_storage = test_cifar_c_corruptions_diffusion(args.dataset, net, args.corruption_dir, transform_test, args.batch_size,
        #                                                     metrics, logger, pretrained_ViT, args)
        #     cor_results = {corruption: {
        #         severity: {metric: cor_results_storage[corruption][severity][metric][0] for metric in metrics} for severity
        #         in range(1, 6)} for corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets}
        #     cor_results_all_models[f"model_{r + 1}"] = cor_results

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    wandb.log({f"Test_final/{metric}": results[metric]['mean'] for metric in results})
    test_results_path = os.path.join(save_path, 'test_results_diffusion.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    # if args.dataset == 'cifar10':
    #     utils.utils.save_cifar_c_results_to_csv(args.dataset, args.attn_type, save_path, metrics, cor_results_all_models)

def test_distillation(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
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
        print(nb_cls)
        if args.model == 'diffusion_distillation':
            net = models.get_model.get_model('diffusion', nb_cls, logger, args)
        elif args.model == 'vit_cifar_distillation':
            net = models.get_model.get_model('vit_cifar', nb_cls, logger, args)
            
        pretrained_ViT = None
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}_{args.temperature}_{args.lambda_mean}_{args.lambda_var}_{args.lambda_ce}.pth')))
        net = net.cuda()
        process_results_diffusion(args, test_loader, net, metrics, logger, "MSP", results_storage, pretrained_ViT)

        if args.dataset == 'cifar10':
            transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            cor_results_storage = test_cifar_c_corruptions_diffusion(args.dataset, net, args.corruption_dir, transform_test, args.batch_size,
                                                            metrics, logger, pretrained_ViT, args)
            cor_results = {corruption: {
                severity: {metric: cor_results_storage[corruption][severity][metric][0] for metric in metrics} for severity
                in range(1, 6)} for corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets}
            cor_results_all_models[f"model_{r + 1}"] = cor_results

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    wandb.log({f"Test_final/{metric}": results[metric]['mean'] for metric in results})
    test_results_path = os.path.join(save_path, 'test_results_diffusion.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    if args.dataset == 'cifar10':
        utils.utils.save_cifar_c_results_to_csv(args.dataset, args.attn_type, save_path, metrics, cor_results_all_models)

if __name__ == '__main__':
    wandb.login(key='1cfab558732ccb32d573a7276a337d22b7d8b371')
    args = utils.test_utils.get_args_parser()
    if args.attn_type == 'kep_svgp':
        group = 'KEP-SVGP'
    else:
        group = 'VIT'
    wandb.init(project='Difformer',     
               group=group,
               name=f"Seed_{args.seed}",
               config=vars(args))
    print(args)
    set_seed(args.seed)
    if args.ood_data is None and args.model == 'diffusion':
        test_diffusion(args)
    elif args.ood_data is None and args.model == 'vit_cifar':
        test(args)
    else:
        ood_test(args)