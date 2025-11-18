import torch
import torch.nn.functional as F
import utils.metrics
import numpy as np  
import sklearn.metrics as skm
import loaders.cifar_loader as cifar_loader
from utils.temperature_scaling import ModelWithTemperature
from utils.mc_dropout import mc_dropout
import torch.distributed as dist
from sklearn.metrics import matthews_corrcoef

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

    rank = dist.get_rank() if dist.is_initialized() else 0

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

    # Added: Synchronize validation logs across all processes in distributed mode
    # if dist.is_initialized():
    #     world_size = dist.get_world_size()
    #     for key in val_log:
    #         # Convert to tensor for all_gather
    #         val_tensor = torch.tensor(val_log[key], dtype=torch.float32 if key != 'target' else torch.int64).cuda()
    #         # Gather arrays from all processes
    #         gathered_tensors = [torch.zeros_like(val_tensor) for _ in range(world_size)]
    #         dist.all_gather(gathered_tensors, val_tensor)
    #         # Concatenate into a single array
    #         val_log[key] = torch.cat(gathered_tensors).cpu().numpy()

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
def validation_qwen(loader, net, tokenizer, args):
    """
    Validation function for Qwen models on text classification tasks.
    
    Args:
        loader: DataLoader with batches containing 'input_ids', 'attention_mask', 'labels'
        net: Qwen model for sequence classification
        tokenizer: Qwen tokenizer
        args: Arguments object
    
    Returns:
        Dictionary with evaluation metrics
    """
    net.eval()
    
    mcc_list = []
    val_log = {'softmax': [], 'correct': [], 'logit': [], 'target': []}

    for batch in loader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        # Forward pass
        outputs = net(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        softmax = F.softmax(logits, dim=1)
        _, pred_cls = softmax.max(1)

        val_log['correct'].append(pred_cls.cpu().eq(labels.cpu().data.view_as(pred_cls)).numpy())
        val_log['softmax'].append(softmax.cpu().data.numpy())
        val_log['logit'].append(logits.cpu().data.numpy())
        val_log['target'].append(labels.cpu().data.numpy())

        # Calculate Matthews Correlation Coefficient for each batch
        mcc_list.append(matthews_corrcoef(labels.cpu().numpy(), pred_cls.cpu().numpy()))

    # Concatenate all batches
    for key in val_log:
        val_log[key] = np.concatenate(val_log[key])
    
    ## Accuracy
    acc = 100. * val_log['correct'].mean()
    
    ## Matthews Correlation Coefficient
    mcc = 100. * np.array(mcc_list).mean()

    # AURC, EAURC (Area Under Risk-Coverage curve)
    aurc, eaurc = utils.metrics.calc_aurc_eaurc(val_log['softmax'], val_log['correct'])
    
    # FPR, AUPR, AUROC
    auroc, aupr_success, aupr, fpr = utils.metrics.calc_fpr_aupr(val_log['softmax'], val_log['correct'])
    
    # Calibration measure: Expected Calibration Error
    ece = utils.metrics.calc_ece(val_log['softmax'], val_log['target'], bins=15)
    
    # NLL and Brier Score
    softmax = val_log['softmax'].astype(np.float32)
    targets = val_log['target'].astype(np.int64)
    log_probs = np.log(softmax[range(len(targets)), targets] + 1e-10)
    nll = -log_probs.mean()
    one_hot = np.zeros_like(softmax)
    one_hot[range(len(targets)), targets] = 1
    brier = np.mean(np.sum((softmax - one_hot) ** 2, axis=1))

    # Results dictionary
    res = {
        'Acc.': acc,
        'MCC': mcc,
        'FPR': fpr * 100,
        'AUROC': auroc * 100,
        'AUPR': aupr * 100,
        'AURC': aurc * 1000,
        'EAURC': eaurc * 1000,
        'AUPR Succ.': aupr_success * 100,
        'ECE': ece * 100,
        'NLL': nll * 10,
        'Brier': brier * 100
    }

    return res

@torch.no_grad()
def validation_text(loader, net, args, time_index=None):
    net.eval()
    
    mcc_list = []
    val_log = {'softmax' : [], 'correct' : [], 'logit' : [], 'target':[]}

    rank = dist.get_rank() if dist.is_initialized() else 0

    for batch in loader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['label'].cuda()

        if args.model == 'diffusion_text':
            output = net(input_ids=input_ids, attention_mask=attention_mask, time_index=time_index)[0]
        elif args.attn_type == "softmax":
            output = net(x=input_ids, attention_mask=attention_mask)['logits']

        softmax = F.softmax(output, dim=1)
        _, pred_cls = softmax.max(1)

        val_log['correct'].append(pred_cls.cpu().eq(labels.cpu().data.view_as(pred_cls)).numpy())
        val_log['softmax'].append(softmax.cpu().data.numpy())
        val_log['logit'].append(output.cpu().data.numpy())
        val_log['target'].append(labels.cpu().data.numpy())

        mcc_list.append(matthews_corrcoef(labels.cpu().numpy(), pred_cls.detach().cpu().numpy()))

    for key in val_log:
        val_log[key] = np.concatenate(val_log[key])
    ## acc
    acc = 100. * val_log['correct'].mean()
    ## mcc
    mcc = 100. * np.array(mcc_list).mean()

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
        'MCC': mcc,
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


if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    from transformers import Qwen2Tokenizer, Qwen2ForSequenceClassification, DataCollatorWithPadding
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description="Validate Qwen model from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name")
    parser.add_argument("--dataset", type=str, default="cola", help="Dataset name (cola, sst2, etc.)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to validate on")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print(f"Loading model from checkpoint: {args.checkpoint}")
    
    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = Qwen2ForSequenceClassification.from_pretrained(args.checkpoint)
    model.cuda()
    model.eval()
    
    print(f"Loading {args.dataset} dataset...")
    # Load dataset
    if args.dataset == "cola":
        dataset = load_dataset("glue", "cola")
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")
    else:
        dataset = load_dataset("glue", args.dataset)
    
    # Preprocess function
    def preprocess_function(examples):
        if "sentence" in examples:
            text = examples["sentence"]
        elif "sentence1" in examples:
            text = examples["sentence1"]
        else:
            raise ValueError("Unknown text field in dataset")
        
        tokenized = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=args.max_length
        )
        tokenized["labels"] = examples["label"]
        return tokenized
    
    # Tokenize dataset
    print(f"Tokenizing {args.split} split...")
    tokenized_dataset = dataset[args.split].map(
        preprocess_function,
        batched=True,
        remove_columns=[col for col in dataset[args.split].column_names if col != "label"]
    )
    
    # Create data collator and loader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    print(f"Running validation on {len(tokenized_dataset)} samples...")
    
    # Run validation
    results = validation_qwen(loader, model, tokenizer, args)
    
    # Print results
    print("\n" + "="*50)
    print(f"Validation Results on {args.dataset.upper()} ({args.split} split)")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric:15s}: {value:8.4f}")
    print("="*50)




