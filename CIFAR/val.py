import torch
import torch.nn.functional as F
import utils.metrics
import numpy as np  
import sklearn.metrics as skm
import datasets.cifar_loader as cifar_loader
from utils.temperature_scaling import ModelWithTemperature
from utils.mc_dropout import mc_dropout

@torch.no_grad()
def validation(loader, net, args, method=None):
    if method == "temperature_scaling":
        _, valid_loader, _, _ = cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        net = ModelWithTemperature(net)
        net.set_temperature(valid_loader)
    elif method == "mc_dropout":
        net = mc_dropout(net, num_estimators=10, last_layer=False, on_batch=False)

    net.eval()
    
    val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[]}

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.attn_type == "softmax":
            if method == "mc_dropout":
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
            
        softmax = F.softmax(output, dim=1)
        _, pred_cls = softmax.max(1)

        val_log['correct'].append(pred_cls.cpu().eq(targets.cpu().data.view_as(pred_cls)).numpy())
        val_log['softmax'].append(softmax.cpu().data.numpy())
        val_log['logit'].append(output.cpu().data.numpy())
        val_log['target'].append(targets.cpu().data.numpy())
        
    for key in val_log : 
        val_log[key] = np.concatenate(val_log[key])
        
    ## acc
    acc = 100. * val_log['correct'].mean()
    
    # aurc, eaurc
    aurc, eaurc = utils.metrics.calc_aurc_eaurc(val_log['softmax'], val_log['correct'])
    # fpr, aupr
    auroc, aupr_success, aupr, fpr = utils.metrics.calc_fpr_aupr(val_log['softmax'], val_log['correct'])
    # calibration measure ece , mce, rmsce
    ece = utils.metrics.calc_ece(val_log['softmax'], val_log['target'], bins=15)
    # brier, nll
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

@torch.no_grad()
def validation_ood(loader, ood_loader, net, args):
    net.eval()
    
    val_log = {'softmax': [], 'logit': []}

    # In-distribution data
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.cuda()
        if args.attn_type == "softmax":
            output = net(inputs)
        elif args.attn_type == "kep_svgp":
            results = [net(inputs)[0] for _ in range(10)]
            output = torch.mean(torch.stack(results), dim=0)
        
        softmax = F.softmax(output, dim=1)
        pred_prob = softmax.max(1)[0]  # Get probabilities of the predicted class
        val_log['softmax'].append(pred_prob.cpu().numpy())
        # val_log['in_softmax'].append(pred_prob.cpu().numpy())
        val_log['logit'].append(output.cpu().numpy())

    # Out-of-distribution data
    for batch_idx, (inputs, _) in enumerate(ood_loader):
        inputs = inputs.cuda()
        if args.attn_type == "softmax":
            output = net(inputs)
        elif args.attn_type == "kep_svgp":
            results = [net(inputs)[0] for _ in range(10)]
            output = torch.mean(torch.stack(results), dim=0)
        
        softmax = F.softmax(output, dim=1)
        pred_prob = softmax.max(1)[0]  # Get probabilities of the predicted class
        val_log['softmax'].append(pred_prob.cpu().numpy())
        # val_log['out_softmax'].append(pred_prob.cpu().numpy())
        val_log['logit'].append(output.cpu().numpy())

    # Concatenate all predictions
    for key in val_log:
        val_log[key] = np.concatenate(val_log[key])

    # Binary target: 1 for in-distribution, 0 for out-of-distribution
    val_log['target'] = np.array([1] * len(loader.dataset) + [0] * len(ood_loader.dataset))

    # Ensure `softmax` aligns with `target`
    assert len(val_log['softmax']) == len(val_log['target']), \
        f"Inconsistent lengths: {len(val_log['softmax'])} vs {len(val_log['target'])}"

    # AUROC and AUPR calculation
    auroc = skm.roc_auc_score(val_log['target'], val_log['softmax'])
    aupr = skm.average_precision_score(val_log['target'], val_log['softmax'])
    fpr, tpr, thresholds = skm.roc_curve(val_log['target'], val_log['softmax'])
    fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
    print(np.where(tpr >= 0.95))

    res = {
        'AUROC': auroc,
        'AUPR': aupr,
        'FPR95': fpr95
    }
    return res

@torch.no_grad()
def validation_diffusion(loader, net, args, pretrained_vit):
    net.eval()
    # pretrained_vit.eval()
    val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[]}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            # output = pretrained_vit._to_words(inputs)
            # output = pretrained_vit.emb(output)
            # output = output + pretrained_vit.pos_emb
            output = net(inputs)
            # h = pretrained_vit.enc[args.depth - 1].la2(output)
            # h = pretrained_vit.enc[args.depth - 1].mlp(h)
            # output = output + h
            # output = pretrained_vit.fc(output.mean(1))

            # if args.attn_type == "softmax":
            #     output = net(inputs)
                
            # elif args.attn_type == "kep_svgp":
            #     results = []
            #     for _ in range(10):
            #         results.append(net(inputs)[0])
            #     outputs = torch.stack(results)
            #     output = torch.mean(outputs, 0)
                
            softmax = F.softmax(output, dim=1)
            _, pred_cls = softmax.max(1)

            val_log['correct'].append(pred_cls.cpu().eq(targets.cpu().data.view_as(pred_cls)).numpy())
            val_log['softmax'].append(softmax.cpu().data.numpy())
            val_log['logit'].append(output.cpu().data.numpy())
            val_log['target'].append(targets.cpu().data.numpy())
            
    for key in val_log : 
        val_log[key] = np.concatenate(val_log[key])
        
    ## acc
    acc = 100. * val_log['correct'].mean()
    
    # aurc, eaurc
    aurc, eaurc = utils.metrics.calc_aurc_eaurc(val_log['softmax'], val_log['correct'])
    # fpr, aupr
    auroc, aupr_success, aupr, fpr = utils.metrics.calc_fpr_aupr(val_log['softmax'], val_log['correct'])
    # calibration measure ece , mce, rmsce
    ece = utils.metrics.calc_ece(val_log['softmax'], val_log['target'], bins=15)
    # brier, nll
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


    
    