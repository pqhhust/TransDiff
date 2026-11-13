"""Equivalent split backward with an exactly replayed stochastic trajectory."""
from contextlib import nullcontext
import torch


def backward_losses(net, model, images, labels, scaler, amp_context, divisor=1,
                    lambda_nll=1., lambda_ce=1., split=True, sync=True, ce_mean=False):
    """Accumulate gradients of (lambda_nll*NLL + lambda_ce*CE) / divisor.

    Prefix/targets are no-grad and computed once. Split mode releases the NLL
    graph before recomputing CE. Restoring CPU/CUDA RNG for CE reuses dropout
    and reparameterization noise; fork_rng leaves the caller's RNG exactly
    where a single forward would. DDP synchronizes only the final backward.
    """
    device = images.device
    with amp_context():
        entry, targets = model.encode(images, targets=lambda_nll != 0)
    if ce_mean and lambda_nll and lambda_ce:
        # CE on the noise-free mean chain (the single inference pass); NLL on the
        # sampled chain. CE goes first under no_sync because it leaves the var head
        # unused; the final NLL backward touches every trainable parameter.
        with net.no_sync() if hasattr(net, 'no_sync') else nullcontext():
            with amp_context():
                logits, _ = net(entry=entry, sample=False)
                ce = torch.nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(ce):
                raise FloatingPointError('Nonfinite cross entropy loss')
            scaler.scale(lambda_ce * ce / divisor).backward()
        ce_value = ce.detach()
        del logits, ce
        context = net.no_sync() if hasattr(net, 'no_sync') and not sync else nullcontext()
        with context:
            with amp_context():
                _, nll = net(entry=entry, targets=targets, with_targets=True, classify=False)
            if not torch.isfinite(nll):
                raise FloatingPointError('Nonfinite likelihood loss')
            scaler.scale(lambda_nll * nll / divisor).backward()
        nll_value = nll.detach()
        return torch.stack([lambda_nll*nll_value + lambda_ce*ce_value, nll_value, ce_value])
    if not split or not lambda_nll or not lambda_ce:
        context = net.no_sync() if hasattr(net, 'no_sync') and not sync else nullcontext()
        with context:
            with amp_context():
                logits, nll = net(entry=entry, targets=targets, with_targets=lambda_nll != 0,
                                  classify=lambda_ce != 0)
                ce = (torch.nn.functional.cross_entropy(logits.float(), labels)
                      if lambda_ce else entry.new_zeros(()))
                loss = lambda_nll*nll + lambda_ce*ce
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss / divisor).backward()
        return torch.stack([loss.detach(), nll.detach(), ce.detach()])

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    context = net.no_sync() if hasattr(net, 'no_sync') else nullcontext()
    with context:
        with amp_context():
            _, nll = net(entry=entry, targets=targets, with_targets=True, classify=False)
        if not torch.isfinite(nll):
            raise FloatingPointError('Nonfinite likelihood loss')
        scaler.scale(lambda_nll * nll / divisor).backward()
    nll_value = nll.detach()
    del nll, targets  # Release teacher targets and likelihood graph before CE.
    devices = [device.index] if device.type == 'cuda' else []
    context = net.no_sync() if hasattr(net, 'no_sync') and not sync else nullcontext()
    with context, torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        with amp_context():
            logits, _ = net(entry=entry)
            ce = torch.nn.functional.cross_entropy(logits.float(), labels)
        if not torch.isfinite(ce):
            raise FloatingPointError('Nonfinite cross entropy loss')
        scaler.scale(lambda_ce * ce / divisor).backward()
    ce_value = ce.detach()
    return torch.stack([lambda_nll*nll_value + lambda_ce*ce_value, nll_value, ce_value])
