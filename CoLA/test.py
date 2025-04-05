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
    cor_results_all_models = {}

    if args.attn_type == 'softmax':
        save_path = args.save_dir + '/' + args.dataset + '_' + args.attn_type + '_' + args.model + '_' + str(args.seed)
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
        process_results(args, test_loader, net, metrics, logger, "Test Evaluation", results_storage)
        process_results(args, ood_loader, net, metrics, logger, "OOD Robustness", results_storage)

    results = {metric: utils.utils.compute_statistics(results_storage[metric]) for metric in metrics}
    wandb.log({f"Test/{metric}": results[metric]['mean'] for metric in results})
    test_results_path = os.path.join(save_path, 'test_results.csv')
    utils.utils.csv_writter(test_results_path, args.dataset, args.model, metrics, results)
    
if __name__ == '__main__':
    wandb.login(key='6cf7b84d1bd52c9eb1e5eade43f583a8059231f2')
    args = utils.test_utils.get_args_parser()
    if args.attn_type == 'kep_svgp':
        group = 'KEP-SVGP'
    else:
        group = 'Transformer'
    wandb.init(project='Difformer',     
               group=group,
               name=f"Seed_{args.seed}",
               config=vars(args))
    print(args)
    set_seed(args.seed)
    test(args)
