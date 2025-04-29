import torch
import val
import os
import utils.test_utils
import utils.utils
from utils.seed_utils import set_seed
import models.get_model
import csv
from torch.utils.data import DataLoader
import wandb
# import gpytorch

from data_loader import get_data, get_vocab, DataLoader


def process_results(args, loader, model, metrics, logger, method_name, results_storage):
    res = val.validation(loader, model, args)
    for metric in metrics:
        results_storage[metric].append(res[metric])
    log = [f"{key}: {res[key]:.3f}" for key in res]
    logger.info(f'################## \n ---> Test {method_name} results：\t' + '\t'.join(log))

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['MCC', 'Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}
    results_storage_ood = {metric: [] for metric in metrics}
    cor_results_all_models = {}

    if args.attn_type == 'sgpa':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_' + str(args.seed)
    if args.attn_type == 'softmax':
        args_model = 'vit_cola' if args.model == 'temperature_scaling' or args.model == 'mc_dropout' else args.model
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args_model + '_' + str(args.seed)
    elif args.attn_type == 'kep_svgp':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_ksvdlayer{}'.format(args.ksvd_layers) + '_ksvd{}'.format(args.eta_ksvd) + '_kl{}'.format(args.eta_kl) + '_' + str(args.seed)
    logger = utils.utils.get_logger(save_path)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_train, gold_train, data_test, gold_test, data_ood, gold_ood=\
            get_data(['./data/cola_public/raw/in_domain_train.tsv','./data/cola_public/raw/in_domain_dev.tsv'],['./data/cola_public/raw/out_of_domain_dev.tsv'], args.seed)
    word_to_int, _ = get_vocab(data_train, args.min_word_count)
    vocab_size = len(word_to_int)

    test_loader = DataLoader(data_test,gold_test,args.batch_size,word_to_int,device,shuffle=False)
    ood_loader = DataLoader(data_ood,gold_ood,args.batch_size,word_to_int,device,shuffle=False)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        
        net = models.get_model.get_model(args.model, vocab_size, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_mcc_net_{r + 1}.pth')))
        net = net.cuda()
        if args.model == 'svdkl':
            pass
            # likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=args.hdim, num_classes=2).cuda()
            # likelihood.load_state_dict(torch.load(os.path.join(save_path, f'best_mcc_likelihood_{r + 1}.pth')))
            # net = (net, likelihood) 
        process_results(args, test_loader, net, metrics, logger, "Test Evaluation", results_storage)
        process_results(args, ood_loader, net, metrics, logger, "OOD Robustness", results_storage_ood)

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    results_ood = {metric: utils.utils.compute_statistics(results_storage_ood[metric]) for metric in metrics}
    wandb.log({f"Test/{metric}": results[metric]['mean'] for metric in results})
    wandb.log({f"Test_ood/{metric}": results_ood[metric]['mean'] for metric in results_ood})
    test_results_path = os.path.join(save_path, 'test_results.csv')
    test_results_path_ood = os.path.join(save_path, 'test_results_ood.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    utils.utils.csv_writter(test_results_path_ood, args.dataset, args.model, metrics, results_ood)

def test_diffusion(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = ['MCC', 'Acc.', 'AUROC', 'AUPR Succ.', 'AUPR', 'FPR', 'AURC', 'EAURC', 'ECE', 'NLL', 'Brier']
    results_storage = {metric: [] for metric in metrics}
    results_storage_ood = {metric: [] for metric in metrics}
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

    logger = utils.utils.get_logger(save_path)
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_train, gold_train, data_test, gold_test, data_ood, gold_ood=\
            get_data(['./data/cola_public/raw/in_domain_train.tsv','./data/cola_public/raw/in_domain_dev.tsv'],['./data/cola_public/raw/out_of_domain_dev.tsv'], args.seed)
    word_to_int, _ = get_vocab(data_train, args.min_word_count)
    vocab_size = len(word_to_int)

    test_loader = DataLoader(data_test,gold_test,args.batch_size,word_to_int,device,shuffle=False)
    ood_loader = DataLoader(data_ood,gold_ood,args.batch_size,word_to_int,device,shuffle=False)

    for r in range(args.nb_run):
        logger.info(f'Testing model_{r + 1} ...')
        
        net = models.get_model.get_model(args.model, vocab_size, logger, args)
        net.load_state_dict(torch.load(os.path.join(save_path, f'best_mcc_net_{r + 1}_{args.lambda_mean}_{args.lambda_var}_{args.lambda_ce}.pth')))
        net = net.cuda()
        process_results(args, test_loader, net, metrics, logger, "Test Evaluation", results_storage)
        process_results(args, ood_loader, net, metrics, logger, "OOD Robustness", results_storage_ood)

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    results_ood = {metric: utils.utils.compute_statistics(results_storage_ood[metric]) for metric in metrics}
    wandb.log({f"Test/{metric}": results[metric]['mean'] for metric in results})
    wandb.log({f"Test_ood/{metric}": results_ood[metric]['mean'] for metric in results_ood})
    test_results_path = os.path.join(save_path, 'test_results.csv')
    test_results_path_ood = os.path.join(save_path, 'test_results_ood.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    utils.utils.csv_writter(test_results_path_ood, args.dataset, args.model, metrics, results_ood)
    
    
if __name__ == '__main__':
    # wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')
    args = utils.test_utils.get_args_parser()
    if args.attn_type == 'kep_svgp':
        group = 'KEP-SVGP-CoLA'
    else:
        group = 'Transformer-CoLA'
    wandb.init(project='Difformer',     
               group=group,
               name=f"Seed_{args.seed}",
               config=vars(args))
    print(args)
    set_seed(args.seed)
    test(args)
