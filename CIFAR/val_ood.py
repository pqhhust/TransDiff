import torch
import torch.nn.functional as F
import gpytorch
import utils.metrics
import numpy as np  
import sklearn.metrics as skm
import datasets.cifar_loader as cifar_loader
from utils.temperature_scaling import ModelWithTemperature
from utils.mc_dropout import mc_dropout
# from laplace import Laplace

@torch.no_grad()
def validation_ood(loader, ood_loader, net, args):
    val_log = {'softmax': [], 'entropy': []}

    if args.model == "temperature_scaling":
        _, valid_loader, _, _ = cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        net = ModelWithTemperature(net)
        net.set_temperature(valid_loader)
    elif args.model == "mc_dropout":
        net = mc_dropout(net, num_estimators=10, last_layer=True, on_batch=False)
    elif args.model == "svdkl":
        net, likelihood = net
        likelihood.eval()
    if args.model != 'kflla':
        net.eval()
    

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.model == 'diffusion':
            softmax_list = []
            for _ in range(10):
                output = net(inputs)
                softmax = F.softmax(output, dim=1)
                softmax_list.append(softmax)
            softmax = torch.mean(torch.stack(softmax_list), 0)
        elif args.model == 'svdkl':
            # pass
            with gpytorch.settings.num_likelihood_samples(10):
                gp_output = net(inputs)
                output_dist = likelihood(gp_output)
                softmax = output_dist.probs.mean(0)
                output = torch.zeros_like(softmax)
        elif args.model == 'kflla':
            softmax = net(inputs)
            output = torch.zeros_like(softmax)
        elif args.model == 'mc_dropout':
            softmax = net(inputs)
            output = torch.zeros_like(softmax)
        else:  
            if args.attn_type == "softmax":
                if args.model == "mc_dropout":
                    output = net(inputs)
                    B, C = inputs.size(0), output.size(1)
                    output = output.view(B, 10, C).mean(1)
                else:
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
            
        pred_prob = softmax.max(1)[0]  
        val_log['softmax'].append(pred_prob.cpu().numpy())

        neg_entropy = torch.sum(softmax * torch.log(softmax + 1e-10), dim=1)
        val_log['entropy'].append(neg_entropy.cpu().numpy())


    # Out-of-distribution data
    for batch_idx, (inputs, targets) in enumerate(ood_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.model == 'diffusion':
            softmax_list = []
            for _ in range(10):
                output = net(inputs)
                softmax = F.softmax(output, dim=1)
                softmax_list.append(softmax)
            softmax = torch.mean(torch.stack(softmax_list), 0)
        elif args.model == 'svdkl':
            # pass
            with gpytorch.settings.num_likelihood_samples(10):
                gp_output = net(inputs)
                output_dist = likelihood(gp_output)
                softmax = output_dist.probs.mean(0)
                output = torch.zeros_like(softmax)
        elif args.model == 'kflla':
            softmax = net(inputs)
            output = torch.zeros_like(softmax)
        elif args.model == 'mc_dropout':
            softmax = net(inputs)
            output = torch.zeros_like(softmax)
        else:  
            if args.attn_type == "softmax":
                if args.model == "mc_dropout":
                    output = net(inputs)
                    B, C = inputs.size(0), output.size(1)
                    output = output.view(B, 10, C).mean(1)
                else:
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
        
        pred_prob = softmax.max(1)[0] 
        val_log['softmax'].append(pred_prob.cpu().numpy())

        neg_entropy = torch.sum(softmax * torch.log(softmax + 1e-10), dim=1)
        val_log['entropy'].append(neg_entropy.cpu().numpy())

    # Concatenate all predictions
    for key in val_log:
        print(key)
        val_log[key] = np.concatenate(val_log[key])

    # Binary target: 1 for in-distribution, 0 for out-of-distribution
    val_log['target'] = np.array([1] * len(loader.dataset) + [0] * len(ood_loader.dataset))

    # Ensure `softmax` aligns with `target`
    assert len(val_log['softmax']) == len(val_log['target']), \
        f"Inconsistent lengths: {len(val_log['softmax'])} vs {len(val_log['target'])}"

    # AUROC, AUPR, FPR95 calculation
    results = {}
    for method in ['softmax', 'entropy']:
        auroc = skm.roc_auc_score(val_log['target'], val_log[method])
        aupr = skm.average_precision_score(val_log['target'], val_log[method])
        fpr, tpr, _ = skm.roc_curve(val_log['target'], val_log[method])
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
        results[method] = {'AUROC': auroc, 'AUPR': aupr, 'FPR95': fpr95}

    final_results = {
        'softmax/AUROC': results['softmax']['AUROC']*100,
        'softmax/AUPR': results['softmax']['AUPR']*100,
        'softmax/FPR95': results['softmax']['FPR95']*100,
        'entropy/AUROC': results['entropy']['AUROC']*100,
        'entropy/AUPR': results['entropy']['AUPR']*100,
        'entropy/FPR95': results['entropy']['FPR95']*100,
        'AUROC': np.mean([results[m]['AUROC'] for m in ['softmax', 'entropy']])*100,
        'AUPR': np.mean([results[m]['AUPR'] for m in ['softmax', 'entropy']])*100,
        'FPR95': np.mean([results[m]['FPR95'] for m in ['softmax', 'entropy']])*100
    }
    return final_results