import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import utils.utils
import wandb  
import torch.distributed as dist
import gc
import math

def compute_loss_diffusion(args, mse_criterion, means_from_diffusion, means_x_minus, stds_from_diffusion, covariances_x_minus):
    """
    Compute the total loss as the sum of MSE losses between Diffusion and ViT outputs.
    
    Parameters:
        mse_criterion (nn.Module): MSE loss function.
        diffusion_layer_outputs (list of tensors): Sampled outputs from Diffusion model layers.
        vit_layer_outputs (list of tensors): Outputs from ViT model layers.
    
    Returns:
        total_loss (Tensor): Sum of MSE losses across all layers.
        layer_losses (dict): Dictionary of individual layer MSE losses.
    """
    means_mse = 0
    stds_mse = 0

    for layer_idx, (mean_diff_out, mean_vit_out) in enumerate(zip(means_from_diffusion, means_x_minus)):
        # Compute MSE loss between Diffusion output and ViT output
        mean_loss = mse_criterion(mean_diff_out, mean_vit_out)
        means_mse += mean_loss / len(means_from_diffusion)
    
    for layer_idx, (std_diff_out, cov_vit_out) in enumerate(zip(stds_from_diffusion, covariances_x_minus)):
        # Compute MSE loss between Diffusion output and ViT output
        # if args.attn_type == 'softmax':
        #     break
        # else:
        #     if args.depth == args.ksvd_layers:
        #         std_loss = mse_criterion(std_diff_out, cov_vit_out)
        #         stds_mse += std_loss #/ len(stds_from_diffusion)
        #     else: 
        #         if layer_idx < (args.depth - args.ksvd_layers):
        #             continue
        #         else:
        #             std_loss = mse_criterion(std_diff_out, cov_vit_out)
        #             stds_mse += std_loss
        std_loss = mse_criterion(std_diff_out, cov_vit_out)
        stds_mse += std_loss / len(stds_from_diffusion)
    
    return means_mse, stds_mse

def train_diffusion(train_loader, diffusion_model, optimizer, epoch, logger, args, vit_model):
    """
    Train the Diffusion model by aligning its layers with the ViT model's layers using MSE loss.
    
    Parameters:
        train_loader (DataLoader): Training data loader with DistributedSampler for DDP.
        diffusion_model (nn.Module): Diffusion model wrapped in DDP.
        optimizer (Optimizer): Optimizer for the Diffusion model.
        epoch (int): Current epoch number.
        logger (Logger or None): Logger for logging (None if not rank 0).
        args (Namespace): Command-line arguments.
        vit_model (nn.Module): Pre-trained ViT model wrapped in DDP for layer alignment.
    """
    diffusion_model.train()
    vit_model.eval()

    for param in vit_model.parameters():
        param.requires_grad = False

    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    # Get rank and check if running in distributed mode
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_distributed = dist.is_initialized()
    
    # Initialize training logs
    train_log = {
        'CE Loss': utils.utils.AverageMeter(),
        'Mean Loss': utils.utils.AverageMeter(),
        'Var Loss': utils.utils.AverageMeter(),
        'Tot. Loss': utils.utils.AverageMeter(),
        'LR': utils.utils.AverageMeter(),
    }

    # Log epoch start only on rank 0
    if rank == 0 and logger is not None:
        msg = '####### --- Training Epoch {:d} --- #######'.format(epoch)
        logger.info(msg)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # optimizer.zero_grad()
        output = diffusion_model(inputs)

        ce_loss = ce_criterion(output, targets)

        with torch.no_grad():
            _, x_t_from_ViT, means_x_minus, covariances_x_minus = vit_model(inputs)

        means_from_diffusion, stds_from_diffusion = diffusion_model(x_t_from_ViT, train=True)

        means_loss, stds_loss = compute_loss_diffusion(args, mse_criterion, means_from_diffusion, means_x_minus, stds_from_diffusion, covariances_x_minus)

        loss = args.lambda_mean * means_loss + args.lambda_var * stds_loss + args.lambda_ce * ce_loss
        loss /= args.accumulation_steps
        loss.backward() 
        if (i + 1) % args.accumulation_steps == 0: 
            nn.utils.clip_grad_value_(diffusion_model.parameters(), args.clip_grad_value)
            optimizer.step()
            optimizer.zero_grad()

        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break
        
        train_log['CE Loss'].update(ce_loss.item(), inputs.size(0))
        train_log['Mean Loss'].update(means_loss.item(), inputs.size(0))
        if args.attn_type == "softmax":
            train_log['Var Loss'].update(stds_loss.item(), inputs.size(0))
        else: 
            train_log['Var Loss'].update(stds_loss.item(), inputs.size(0))
        train_log['Tot. Loss'].update(loss.item(), inputs.size(0))
        train_log['LR'].update(lr, inputs.size(0))

        # Log every 100 batches only on rank 0
        if i % 100 == 99 and rank == 0 and logger is not None:
            log = ['LR : {:.5f}'.format(train_log['LR'].avg)] + [
                key + ': {:.2f}'.format(train_log[key].avg) for key in train_log if key != 'LR'
            ]
            msg = 'Epoch {:d} \t Batch {:d}\t'.format(epoch, i) + '\t'.join(log)
            logger.info(msg)
            for key in train_log:
                train_log[key] = utils.utils.AverageMeter()

        if i % 100 == 99:
            torch.cuda.empty_cache()
            import gc; gc.collect()
            dist.barrier()
    # Synchronize logs across all processes in distributed mode
    # if is_distributed:
    #     for key in train_log:
    #         avg_val = torch.tensor(train_log[key].avg, device='cuda')
    #         dist.all_reduce(avg_val)
    #         train_log[key].avg = (avg_val / dist.get_world_size()).item()
    
    # Log to wandb only on rank 0
    # if rank == 0:
    #     wandb.log({f"Train/{key}": train_log[key].avg for key in train_log}, step=epoch)


def negative_log_likelihood(xs, means, covariances):
    nll = 0
    for x, mean, std in zip(xs, means, covariances):
        ## compute log density of a Gaussian with mean mean and std std at x, std is a diagonal matrix
        B, S, D = mean.shape
        N = S * D

        log_var = torch.log(std ** 2)                        # [B, S, D]
        log_det_term = log_var.sum(dim=(1, 2))              # [B]
        quad_term = (((x - mean) / std) ** 2).sum(dim=(1, 2))  # [B]

        nll += 0.5 * (N * math.log(2 * math.pi) + log_det_term + quad_term)  # [B]
    return nll

def train_diffusion_text(train_loader, diffusion_model, optimizer, epoch, logger, args, qwen2_model):
    diffusion_model.train()
    qwen2_model.eval()

    for p in qwen2_model.parameters():
        p.requires_grad = False

    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_distributed = dist.is_initialized()

    train_log = {
        'CE Loss': utils.utils.AverageMeter(),
        # 'Mean Loss': utils.utils.AverageMeter(),
        # 'Var Loss': utils.utils.AverageMeter(),
        'NLL': utils.utils.AverageMeter(),
        'Tot. Loss': utils.utils.AverageMeter(),
        'LR': utils.utils.AverageMeter(),
    }

    if rank == 0 and logger is not None:
        logger.info('####### --- Training (Text) Epoch {:d} --- #######'.format(epoch))

    for i, batch in enumerate(train_loader):
        # print(batch)
        if isinstance(batch, dict):
            input_ids = batch.get('input_ids').cuda(non_blocking=True)
            attention_mask = batch.get('attention_mask', None)
            token_type_ids = batch.get('token_type_ids', None)
            labels = batch.get('label') if 'label' in batch else batch.get('targets')
            attention_mask = attention_mask.cuda(non_blocking=True) if attention_mask is not None else None
            token_type_ids = token_type_ids.cuda(non_blocking=True) if token_type_ids is not None else None
            targets = labels.cuda(non_blocking=True)
        else:
            # Expect (input_ids, attention_mask, labels) or (input_ids, labels)
            if len(batch) == 3:
                input_ids, attention_mask, targets = batch
                token_type_ids = None
            elif len(batch) == 2:
                input_ids, targets = batch
                attention_mask = None
                token_type_ids = None
            else:
                raise ValueError('Unsupported batch format for text training')
            input_ids = input_ids.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True) if attention_mask is not None else None

        with torch.no_grad():
            _, _, x_t_from_qwen2, means_x_minus = qwen2_model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            covariances_x_minus = [torch.zeros_like(mean) for mean in means_x_minus]
            means_x_minus = means_x_minus[-args.last_layers:]
            covariances_x_minus = covariances_x_minus[-args.last_layers:]
        output, means_from_diffusion, stds_from_diffusion = diffusion_model(input_ids, attention_mask)  # logits
        ce_loss = ce_criterion(output, targets)

        # Diffusion alignment path
        # means_from_diffusion, stds_from_diffusion = diffusion_model(x_t_from_qwen2, train=True)

        selected_indices_x = [i for i in range(0, args.last_layers + 1)]
        selected_indices_mean = [i for i in range(1, args.last_layers)]
        #     ## merge 3
        # selected_indices_x = [0, 3, 6, 9, 12]
        # selected_indices_mean = [2, 5, 8, 11]
        #     ## merge 4
        #     # selected_indices_x = [0, 4, 8, 12]
        #     # selected_indices_mean = [3, 7, 11]
        # time_index = [selected_indices_x[i]*1.0/24 for i in range(len(selected_indices_x)-1)]
            
        # subset_x = [x_t_from_qwen2[i] for i in selected_indices_x]
        subset_mean = means_x_minus
        subset_cov = covariances_x_minus

        nll = negative_log_likelihood(x_t_from_qwen2, means_from_diffusion, stds_from_diffusion)

        # means_loss, stds_loss = compute_loss_diffusion(args, mse_criterion, means_from_diffusion, subset_mean, stds_from_diffusion, subset_cov)

        loss = args.lambda_mean * nll + args.lambda_ce * ce_loss
        loss /= args.accumulation_steps
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            if args.clip_grad_value != 0:
                nn.utils.clip_grad_value_(diffusion_model.parameters(), args.clip_grad_value)
            optimizer.step()
            optimizer.zero_grad()

        # LR for logging
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
            break

        train_log['CE Loss'].update(ce_loss.item(), input_ids.size(0))
        train_log['NLL'].update(nll.item(), input_ids.size(0))
        # train_log['Mean Loss'].update(means_loss.item(), input_ids.size(0))
        # train_log['Var Loss'].update(stds_loss.item(), input_ids.size(0))
        train_log['Tot. Loss'].update(loss.item(), input_ids.size(0))
        train_log['LR'].update(lr, input_ids.size(0))

        if i % 100 == 99 and rank == 0 and logger is not None:
            log = ['LR : {:.5f}'.format(train_log['LR'].avg)] + [
                key + ': {:.2f}'.format(train_log[key].avg) for key in train_log if key != 'LR'
            ]
            msg = 'Epoch {:d} \t Batch {:d}\t'.format(epoch, i) + '\t'.join(log)
            logger.info(msg)
            for key in train_log:
                train_log[key] = utils.utils.AverageMeter()

        if i % 100 == 99:
            torch.cuda.empty_cache()
            import gc; gc.collect()
            if is_distributed:
                dist.barrier()
