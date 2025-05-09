import torch
import torch.nn.functional as F
import utils.metrics
import numpy as np 
from sklearn.metrics import matthews_corrcoef
from utils.temperature_scaling import ModelWithTemperature
from utils.mc_dropout import mc_dropout
from data_loader import get_imdb_data
# from data_loader import get_data, get_vocab, DataLoader
# import gpytorch

@torch.no_grad()
def validation(loader, net, args, method=None):
    # if args.model == 'svdkl':
    #     method = 'svdkl'
    if args.model == "temperature_scaling":
        train_loader, val_loader, test_loader, tokenizer = get_imdb_data('./data', args.batch_size)
        net = ModelWithTemperature(net)
        net.set_temperature(val_loader)
    elif args.model == "mc_dropout":
        net = mc_dropout(net, num_estimators=10, last_layer=True, on_batch=False)
    elif args.model == 'svdkl':
        net, likelihood = net
        likelihood.eval()

    if args.model != 'kflla':
        net.eval()
    
    # mcc_list = []
    val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[]}

    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.model == 'svdkl':
            pass
            # with gpytorch.settings.num_likelihood_samples(10):
            #     gp_output = net(inputs)
            #     output_dist = likelihood(gp_output)
            #     softmax = output_dist.probs.mean(0)
            #     output = torch.zeros_like(softmax)
        elif args.model == 'mc_dropout' or args.model == 'kflla':
            softmax = net(inputs)
            output = torch.zeros_like(softmax)
        else:
            if args.model == 'diffusion':
                output = net(inputs, train=False)
            elif args.attn_type == "softmax":
                output = net(inputs)
                
            elif args.attn_type == "kep_svgp":
                results = []
                for _ in range(10):
                    results.append(net(inputs)[0])
                outputs = torch.stack(results)
                output = torch.mean(outputs, 0)
            
            elif args.attn_type == "sgpa":
                results = []
                for _ in range(10):
                    results.append(net(inputs)[0])
                outputs = torch.stack(results)
                output = torch.mean(outputs, 0)
                
            softmax = F.softmax(output, dim=1)
        _, pred_cls = softmax.max(1)

        val_log['correct'].append(pred_cls.cpu().eq(targets.cpu().data.view_as(pred_cls)).numpy())
        val_log['softmax'].append(softmax.cpu().data.numpy())
        val_log['logit'].append(output.cpu().data.numpy())
        val_log['target'].append(targets.cpu().data.numpy())

        # mcc_list.append(matthews_corrcoef(answers.cpu().numpy(), pred_cls.detach().cpu().numpy()))
        
    for key in val_log : 
        val_log[key] = np.concatenate(val_log[key])
    ## acc
    acc = 100. * val_log['correct'].mean()
    ## mcc
    # mcc = 100. * np.array(mcc_list).mean()

    # aurc, eaurc
    aurc, eaurc = utils.metrics.calc_aurc_eaurc(val_log['softmax'], val_log['correct'])
    # fpr, aupr
    auroc, aupr_success, aupr, fpr = utils.metrics.calc_fpr_aupr(val_log['softmax'], val_log['correct'])
    # calibration measure ece , mce, rmsce
    ece = utils.metrics.calc_ece(val_log['softmax'], val_log['target'], bins=15)
    # brier, nll
    if args.model == 'svdkl' or args.model == 'mc_dropout':
        softmax = val_log['softmax'].astype(np.float32)
        targets = val_log['target'].astype(np.int64)
        log_probs = np.log(softmax[range(len(targets)), targets] + 1e-10)
        nll = -log_probs.mean()
        one_hot = np.zeros_like(softmax)
        one_hot[range(len(targets)), targets] = 1
        brier = np.mean(np.sum((softmax - one_hot) ** 2, axis=1))
    else:
        nll, brier = utils.metrics.calc_nll_brier(val_log['softmax'], val_log['logit'], val_log['target'])

    # log
    res = {
        'Acc.': acc,
        'FPR' : fpr*100,
        'AUROC': auroc*100,
        'AUPR': aupr*100,
        'AURC': aurc*1000,
        'EAURC': eaurc*1000,
        'AUPR Succ.': aupr_success*100,
        'ECE' : ece*100,
        'NLL' : nll*10,
        'Brier' : brier*100
    }

    return res