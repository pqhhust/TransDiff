import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import utils.utils
import wandb  
import torch.distributed as dist
import gc

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

def interpolate_list(values, k):
    """
    Expand a list with linear interpolation.
    
    Args:
        values (list of floats/ints): [x0, x1, ..., xn]
        k (int): number of subdivisions between each pair of points

    Returns:
        list: expanded list with interpolated values
    """
    expanded = []
    for i in range(len(values) - 1):
        x0, x1 = values[i], values[i+1]
        for j in range(k):
            # Linear interpolation: weighted average
            interp_val = (1 - j/k) * x0 + (j/k) * x1
            expanded.append(interp_val)
    expanded.append(values[-1])  # add the last element
    return expanded

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
        # if epoch > args.warmup_epoch:
        # if epoch % 2 == 0:
        #     selected_indices_x = [0, 2, 4, 6, 8, 10, 12]
        #     time_index = [selected_indices_x[i]*1.0/12 for i in range(len(selected_indices_x)-1)]
        # else:
        #     time_index = [i*1.0/12 for i in range(12)]
        
        selected_indices_x = [0, 2, 4, 6, 8, 10, 12]
        # selected_indices_x = [0, 3, 6, 9, 12]
        time_index = [selected_indices_x[i]*1.0/12 for i in range(len(selected_indices_x)-1)]
        
        output, means_from_diffusion, stds_from_diffusion = diffusion_model(inputs, time_index=time_index)

        ce_loss = ce_criterion(output, targets)

        with torch.no_grad():
            # _, x_t_from_ViT, means_x_minus, covariances_x_minus = vit_model(inputs)
            soft_logits, x_t_from_ViT, means_x_minus, covariances_x_minus = vit_model(inputs)

        #Soften the student logits by applying softmax first and log() second
        T = 1.0
        soft_targets = nn.functional.softmax(soft_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(output / T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        # ce_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

        # # if epoch > args.warmup_epoch:
        # if epoch % 2 == 0:
        #     ### merge layers
        #     ## merge 2 
        selected_indices_x = [0, 2, 4, 6, 8, 10, 12]
        selected_indices_mean = [1, 3, 5, 7, 9, 11]
        #     ## merge 3
        # selected_indices_x = [0, 3, 6, 9, 12]
        # selected_indices_mean = [2, 5, 8, 11]
        #     ## merge 4
        #     # selected_indices_x = [0, 4, 8, 12]
        #     # selected_indices_mean = [3, 7, 11]
        time_index = [selected_indices_x[i]*1.0/12 for i in range(len(selected_indices_x)-1)]
            
        subset_x = [x_t_from_ViT[i] for i in selected_indices_x]
        subset_mean = [means_x_minus[i] for i in selected_indices_mean]
        subset_cov = [covariances_x_minus[i] for i in selected_indices_mean]
        # subset_cov = [torch.randn_like(x)*args.var_range for x in subset_cov]
        # else:
        ### divide sublayers
        # time_index = interpolate_list([i*1.0 for i in range(13)], k=2)
        # time_index = [i/12.0 for i in time_index][:-1]

        # subset_x = interpolate_list(x_t_from_ViT, k=2)
        # subset_mean = subset_x[1:]
        # subset_cov = [torch.zeros_like(x) for x in subset_mean]
        # del x_t_from_ViT, means_x_minus, covariances_x_minus

        # means_from_diffusion, stds_from_diffusion = diffusion_model(subset_x, train=True, time_index=time_index)
        ### residual
        # residual_mean = [x_t_from_ViT[i+1]-x_t_from_ViT[i] for i in range(len(x_t_from_ViT)-1)]

        means_loss, stds_loss = compute_loss_diffusion(args, mse_criterion, means_from_diffusion, subset_mean, stds_from_diffusion, subset_cov)

        # if epoch >= 5:
        #     loss = 0.5 * means_loss + args.lambda_var * stds_loss + 1 * ce_loss
        # else: 
        loss = args.lambda_mean * means_loss + args.lambda_var * stds_loss + args.lambda_ce * ce_loss
        
        loss /= args.accumulation_steps
        loss.backward() 
        if (i + 1) % args.accumulation_steps == 0: 
            if args.clip_grad_value != 0:
                nn.utils.clip_grad_value_(diffusion_model.parameters(), args.clip_grad_value)
            optimizer.step()
            optimizer.zero_grad()

            # diffusion_model.module.embedding.load_state_dict(vit_model.module.model.vit.embeddings.state_dict())
            # diffusion_model.module.intermediate.load_state_dict(vit_model.module.model.vit.encoder.layer[-1].intermediate.state_dict())
            # diffusion_model.module.output.load_state_dict(vit_model.module.model.vit.encoder.layer[-1].output.state_dict())
            # diffusion_model.module.layernorm_after.load_state_dict(vit_model.module.model.vit.encoder.layer[-1].layernorm_after.state_dict())
            # diffusion_model.module.layernorm.load_state_dict(vit_model.module.model.vit.layernorm.state_dict())
            # if epoch < 20:
            #     diffusion_model.module.classifier.load_state_dict(vit_model.module.model.classifier.state_dict())

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
