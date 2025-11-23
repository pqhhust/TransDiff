import torch
import torch.nn.functional as F
# import gpytorch
import utils.metrics
import numpy as np  
import sklearn.metrics as skm
import datasets.cifar_loader as cifar_loader
from utils.temperature_scaling import ModelWithTemperature
from utils.mc_dropout import mc_dropout
import wandb
# from laplace import Laplace

@torch.no_grad()
def validation(loader, net, args, method=None):
    if args.model == 'svdkl':
        method = 'svdkl'
    if args.model == "temperature_scaling":
        _, valid_loader, _, _ = cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
                                                                       args.test_dir, args.batch_size)
        net = ModelWithTemperature(net)
        net.set_temperature(valid_loader)
    elif args.model == "mc_dropout":
        net = mc_dropout(net, num_estimators=10, last_layer=False, on_batch=False)
    elif method == "svdkl":
        net, likelihood = net
        likelihood.eval()
    # if method == "kflla":
    #     net.train()
    #     la = Laplace(net, 'classification', subset_of_weights='last_layer', hessian_structure='kron')
    #     train_loader, _, _, _ = cifar_loader.get_loader(args.dataset, args.train_dir, args.val_dir,
    #                                                                    args.test_dir, args.batch_size)
    #     with torch.enable_grad():
    #         la.fit(train_loader)
    #         la.optimize_prior_precision(method='marglik')
    net.eval()
    
    val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[], 'rv_coff': []}

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        if method == 'svdkl':
            pass
            # with gpytorch.settings.num_likelihood_samples(10):
            #     gp_output = net(inputs)
            #     output_dist = likelihood(gp_output)
            #     softmax = output_dist.probs.mean(0)
            #     output = torch.zeros_like(softmax)
        if method == 'kflla':
            pass
            # softmax = la(inputs)
            # output = torch.zeros_like(softmax)
        elif args.model == 'mc_dropout':
            softmax = net(inputs)
            output = torch.zeros_like(softmax)
        else:  
            if args.attn_type == "softmax":
                if method == "mc_dropout":
                    output = net(inputs)
                    B, C = inputs.size(0), output.size(1)
                    output = output.view(B, 10, C).mean(1)
                else:
                    output = net(inputs)    
            elif args.attn_type == "kep_svgp":
                results = []
                x1s = []
                x2s = []
                # frobenius_norm = 0
                for _ in range(10):
                    out, x_t, _, _ = net(inputs)
                    results.append(out)
                    # print(x_t[0][0].shape)
                    ## Shape of x_t: list of num_layers + 1 vectors, each of shape (B, num_tokens, dim)
                    x1s.append(x_t[args.index1].reshape(x_t[args.index1].size(0), 1, -1)) ## Shape (B, num_tokens, dim)
                    x2s.append(x_t[args.index2].reshape(x_t[args.index2].size(0), 1, -1)) ## Shape (B, num_tokens, dim)
                # print(x1s[0].shape)
                x1s = torch.stack(x1s) 
                x1s = x1s.reshape(x1s.size(0), x1s.size(1), x1s.size(2), x1s.size(3), 1).permute(1, 2, 0, 3, 4) ## (B, num_tokens, num_samples, dim, 1)
                x2s = torch.stack(x2s) 
                x2s = x2s.reshape(x2s.size(0), x2s.size(1), x2s.size(2), x2s.size(3), 1).permute(1, 2, 0, 3, 4) ## (B, num_tokens, num_samples, dim, 1)
                ## Compute cross-layer covariance of x1s and x2s
                ## Compute E[X1X2^T] - E[X1]E[X2]^T
                ex1 = torch.mean(x1s, dim=2, keepdim=True)
                ex2 = torch.mean(x2s, dim=2, keepdim=True)
                x1_minus_ex1 = x1s - ex1
                x2_minus_ex2 = x2s - ex2
                # ex1x2t = torch.mean(torch.matmul(x1s, x2s), dim=0)
                # vx1 = torch.var(x1s, dim=2, unbiased=True, keepdim=True)
                # trvx1square = (vx1 ** 2).sum((-2, -1))
                # vx2 = torch.var(x2s, dim=2, unbiased=True, keepdim=True)
                # trvx2square = (vx2 ** 2).sum((-2, -1))
                var_x1 = torch.matmul(x1_minus_ex1, x1_minus_ex1.transpose(-2, -1)).mean(2)  ## (B, num_tokens, dim, dim)
                var_x2 = torch.matmul(x2_minus_ex2, x2_minus_ex2.transpose(-2, -1)).mean(2)  ## (B, num_tokens, dim, dim)
                cov_x1x2 = torch.matmul(x1_minus_ex1, x2_minus_ex2.transpose(-2, -1)).mean(2)
                # frobenius_norm = torch.norm(cov_x1x2, dim=(-2, -1)).squeeze(0).mean(1)
                tr_cov_x1x2square = (cov_x1x2 ** 2).sum((-2, -1))  ## (B, num_tokens)
                trvx1square = (var_x1 ** 2).sum((-2, -1))  ## (B, num_tokens)
                trvx2square = (var_x2 ** 2).sum((-2, -1))  ## (B, num_tokens)
                rv_coff = tr_cov_x1x2square / (trvx1square.sqrt() * trvx2square.sqrt() + 1e-10) ## (1, B, num_tokens)
                val_log['rv_coff'].append(rv_coff.cpu().data.numpy())
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
    if args.model == 'svdkl' or args.model == 'kflla' or args.model == 'mc_dropout':
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
    wandb.log({'RV_Coff': np.mean(val_log['rv_coff'])})
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
        'Brier' : brier*100,
        'RV Coff': np.mean(val_log['rv_coff'])
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
    val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[], 'rv_coff': []}

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # output = pretrained_vit._to_words(inputs)
        # output = pretrained_vit.emb(output)
        # output = output + pretrained_vit.pos_emb
        results = []
        x1s = []
        x2s = []
        # frobenius_norm = 0
        for _ in range(10):
            out, x_t = net(inputs)
            results.append(out)
            # print(x_t[0][0].shape)
            ## Shape of x_t: list of num_layers + 1 vectors, each of shape (B, num_tokens, dim)
            x1s.append(x_t[args.index1].permute(0, 2, 1)) ## Shape (B, num_tokens, dim)
            x2s.append(x_t[args.index2].permute(0, 2, 1)) ## Shape (B, num_tokens, dim)
        # print(x1s[0].shape)
        x1s = torch.stack(x1s) 
        x1s = x1s.reshape(x1s.size(0), x1s.size(1), x1s.size(2), x1s.size(3), 1).permute(1, 2, 0, 3, 4) ## (B, num_tokens, num_samples, dim, 1)
        x2s = torch.stack(x2s) 
        x2s = x2s.reshape(x2s.size(0), x2s.size(1), x2s.size(2), x2s.size(3), 1).permute(1, 2, 0, 3, 4) ## (B, num_tokens, num_samples, dim, 1)
        ## Compute cross-layer covariance of x1s and x2s
        ## Compute E[X1X2^T] - E[X1]E[X2]^T
        ex1 = torch.mean(x1s, dim=2, keepdim=True)
        ex2 = torch.mean(x2s, dim=2, keepdim=True)
        x1_minus_ex1 = x1s - ex1
        x2_minus_ex2 = x2s - ex2
        # ex1x2t = torch.mean(torch.matmul(x1s, x2s), dim=0)
        # vx1 = torch.var(x1s, dim=2, unbiased=True, keepdim=True)
        # trvx1square = (vx1 ** 2).sum((-2, -1))
        # vx2 = torch.var(x2s, dim=2, unbiased=True, keepdim=True)
        # trvx2square = (vx2 ** 2).sum((-2, -1))
        var_x1 = torch.matmul(x1_minus_ex1, x1_minus_ex1.transpose(-2, -1)).mean(2)  ## (B, num_tokens, dim, dim)
        var_x2 = torch.matmul(x2_minus_ex2, x2_minus_ex2.transpose(-2, -1)).mean(2)  ## (B, num_tokens, dim, dim)
        cov_x1x2 = torch.matmul(x1_minus_ex1, x2_minus_ex2.transpose(-2, -1)).mean(2)
        # frobenius_norm = torch.norm(cov_x1x2, dim=(-2, -1)).squeeze(0).mean(1)
        tr_cov_x1x2square = (cov_x1x2 ** 2).sum((-2, -1))  ## (B, num_tokens)
        trvx1square = (var_x1 ** 2).sum((-2, -1))  ## (B, num_tokens)
        trvx2square = (var_x2 ** 2).sum((-2, -1))  ## (B, num_tokens)
        rv_coff = tr_cov_x1x2square / (trvx1square.sqrt() * trvx2square.sqrt() + 1e-10) ## (1, B, num_tokens)
        val_log['rv_coff'].append(rv_coff.cpu().data.numpy())
        outputs = torch.stack(results)
        output = torch.mean(outputs, 0)
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
    wandb.log({'RV_Coff': np.mean(val_log['rv_coff'])})
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
        'Brier' : brier*100,
        'RV Coff': np.mean(val_log['rv_coff'])
    }

    return res


    
    
