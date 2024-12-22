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

def process_results(args, loader, model, metrics, logger, method_name, results_storage):
    res = val.validation(loader, model, args)
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


def test_cifar_c_corruptions(dataset, model, corruption_dir, transform_test, batch_size, metrics, logger):
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

def ood_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['AUROC', 'AUPR', 'FPR95']
    results_storage = {metric: [] for metric in metrics}
    
    if args.attn_type == 'softmax':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl)

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


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}
    cor_results_all_models = {}

    if args.attn_type == 'softmax':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl)
    logger = utils.utils.get_logger(save_path)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        _, valid_loader, test_loader, nb_cls = datasets.cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        print(nb_cls)
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}.pth')))
        net = net.cuda()
        process_results(args, test_loader, net, metrics, logger, "MSP", results_storage)

        if args.dataset == 'cifar10':
            transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            cor_results_storage = test_cifar_c_corruptions(args.dataset, net, args.corruption_dir, transform_test, args.batch_size,
                                                            metrics, logger)
            cor_results = {corruption: {
                severity: {metric: cor_results_storage[corruption][severity][metric][0] for metric in metrics} for severity
                in range(1, 6)} for corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets}
            cor_results_all_models[f"model_{r + 1}"] = cor_results

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    test_results_path = os.path.join(save_path, 'test_results.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    if args.dataset == 'cifar10':
        utils.utils.save_cifar_c_results_to_csv(args.dataset, args.attn_type, save_path, metrics, cor_results_all_models)

def test_diffusion():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}
    cor_results_all_models = {}

    if args.attn_type == 'softmax':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model
        pretrained_path = args.pretrained_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + 'vit_cifar' + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl)
        pretrained_path = args.pretrained_dir + '/' + args.dataset + '_' + args.attn_type + '_' + 'vit_cifar' + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl)
    logger = utils.utils.get_logger(save_path)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        _, valid_loader, test_loader, nb_cls = datasets.cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        print(nb_cls)
        net = models.get_model.get_model(args.model, nb_cls, logger, args)
        pretrained_ViT = models.get_model.get_model('q_distribution', nb_cls, logger, args)
        pretrained_ViT.load_state_dict(torch.load(os.path.join(pretrained_path, f'best_acc_net_{r + 1}.pth')))
        pretrained_ViT.cuda()
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_acc_net_{r + 1}_diffusion_{args.backbone}.pth')))
        net = net.cuda()
        process_results(args, test_loader, net, metrics, logger, "MSP", results_storage, pretrained_ViT)

        if args.dataset == 'cifar10':
            transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            cor_results_storage = test_cifar_c_corruptions(args.dataset, net, args.corruption_dir, transform_test, args.batch_size,
                                                            metrics, logger, pretrained_ViT)
            cor_results = {corruption: {
                severity: {metric: cor_results_storage[corruption][severity][metric][0] for metric in metrics} for severity
                in range(1, 6)} for corruption in datasets.CIFARC.CIFAR10C.cifarc_subsets}
            cor_results_all_models[f"model_{r + 1}"] = cor_results

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    test_results_path = os.path.join(save_path, 'test_results_diffusion.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    if args.dataset == 'cifar10':
        utils.utils.save_cifar_c_results_to_csv(args.dataset, args.attn_type, save_path, metrics, cor_results_all_models)


if __name__ == '__main__':
    args = utils.test_option.get_args_parser()
    set_seed(args.seed)
    if args.ood_data is None and args.model == 'diffusion':
        test_diffusion()
    elif args.ood_data is None and args.model == 'vit_cifar':
        test()
    else:
        ood_test()