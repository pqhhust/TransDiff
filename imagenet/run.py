"""python -m imagenet.run --config imagenet/configs/vit_large_last3.json"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification

from .data import ImageNetStream, row_count, shard_files
from .metrics import Metrics
from .model import ViTTransDiff
from .backward import backward_losses


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--checkpoint', default=str(Path.home() / 'models/google/vit-large-patch16-224'))
    p.add_argument('--data', default=os.path.join(os.getenv('HF_PUBLIC', '/projects/public/huggingface'),
                                                'datasets/ILSVRC/imagenet-1k'))
    p.add_argument('--output', default='results/imagenet-vit-large-last3')
    p.add_argument('--from-layer', type=int, default=21)
    p.add_argument('--to-layer', type=int, default=24)
    p.add_argument('--trans-depth', type=int, default=3)
    p.add_argument('--mlp-ratio', type=float, default=4.)
    p.add_argument('--mlp', choices=['gelu', 'swiglu'], default='gelu', help='DiT block MLP; swiglu uses 2/3 width (parameter-matched)')
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--var-head', choices=['qwen', 'linear'], default='qwen')
    p.add_argument('--head-init', choices=['default', 'zero'], default='default')
    p.add_argument('--log-std-init', type=float, default=-2.)
    p.add_argument('--lambda-nll', type=float, default=1.)
    p.add_argument('--lambda-ce', type=float, default=1.)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--min-lr', type=float, default=1e-7)
    p.add_argument('--weight-decay', type=float, default=5e-7)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--warmup-steps', type=int, default=0, help='Linear warm-up in optimizer updates; overrides --warmup-epochs when > 0')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--effective-batch-size', type=int, default=512)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--steps-per-epoch', type=int, default=0, help='Optimizer updates; 0 covers about one full pass')
    p.add_argument('--val-limit', type=int, default=0, help='0 evaluates all 50,000 validation images')
    p.add_argument('--mc-samples', type=int, default=1, help='1 matches Qwen; >1 averages probabilities')
    p.add_argument('--ce-mean', action=argparse.BooleanOptionalAction, default=False,
                   help='Train CE on the noise-free mean chain; NLL stays on the sampled chain')
    p.add_argument('--eval-mean', action=argparse.BooleanOptionalAction, default=False,
                   help='Validate with one noise-free pass through the transition means')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'bf16'], default='auto')
    p.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--split-backward', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--clip-grad-value', type=float, default=0.)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False,
                   help='torch.compile frozen prefix blocks and shared DiT')
    p.add_argument('--resume', help='Trusted checkpoint produced by this runner')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--baseline', action='store_true', help='Evaluate original pretrained ViT, without training')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False,
                   help='Log to Weights & Biases (key from ~/.netrc or WANDB_API_KEY)')
    p.add_argument('--wandb-project', default='TransDiff-ImageNet')
    p.add_argument('--wandb-entity', default=os.getenv('WANDB_ENTITY'))
    p.add_argument('--run-name')
    preliminary, _ = p.parse_known_args()
    if preliminary.config:
        config = json.loads(Path(preliminary.config).read_text())
        unknown = set(config) - {a.dest for a in p._actions}
        if unknown:
            p.error(f'Unknown configuration fields: {unknown}')
        p.set_defaults(**config)
    a = p.parse_args()
    for key in ['epochs', 'batch_size', 'effective_batch_size', 'mc_samples', 'threads', 'log_every', 'trans_depth']:
        if getattr(a, key) < 1:
            p.error(f'{key} must be positive')
    if min(a.workers, a.steps_per_epoch, a.val_limit, a.warmup_epochs, a.warmup_steps) < 0:
        p.error('Workers, limits and warmup must be nonnegative')
    if a.lambda_nll < 0 or a.lambda_ce < 0 or a.lambda_nll + a.lambda_ce == 0:
        p.error('Loss weights must be nonnegative and not both zero')
    if a.evaluate and not a.baseline and not a.resume:
        p.error('Student evaluation requires --resume (or use --baseline)')
    return a


def start_wandb(a, manifest, rank):
    if not a.wandb or rank != 0:
        return None
    import wandb
    kind = 'baseline' if a.baseline else 'eval' if a.evaluate else 'train'
    return wandb.init(project=a.wandb_project, entity=a.wandb_entity,
                      name=a.run_name or f"{kind}-{Path(a.output).name}",
                      config=manifest, job_type=kind, dir=a.output)


def amp_context(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype else nullcontext()


def loader(a, processor, split, rank, world, epoch=0):
    training = split == 'train'
    stream = ImageNetStream(a.data, split, processor, rank, world, a.seed,
                           epoch, limit=0 if training else a.val_limit, repeat=training)
    return DataLoader(stream, batch_size=a.batch_size, num_workers=a.workers,
                      pin_memory=torch.cuda.is_available(), drop_last=training,
                      **({'prefetch_factor': 2} if a.workers else {}))


@torch.no_grad()
def evaluate(model, batches, device, dtype, samples=1, baseline=False, mean=False):
    model.eval()
    stats = Metrics(device)
    for images, labels in batches:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with amp_context(device, dtype):
            if baseline:
                probs = model.teacher(images).logits.float().softmax(-1)
            else:
                entry, _ = model.encode(images)  # Prefix reused across all MC samples.
                probs = None
                for _ in range(1 if mean else samples):
                    logits, _, _ = model.from_entry(entry, sample=not mean)
                    p = logits.float().softmax(-1)
                    probs = p if probs is None else probs + p
                probs /= 1 if mean else samples
        stats.update(probs, labels)
    return stats.compute()


def lr_at(update, total, warmup, peak, minimum):
    if update < warmup:
        return peak * (update + 1) / max(1, warmup)
    fraction = (update - warmup) / max(1, total - warmup - 1)
    return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * min(1, fraction)))


def atomic_save(state, path):
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    a = arguments()
    torch.set_num_threads(a.threads)
    world, rank, local_rank = (int(os.getenv(k, default)) for k, default in
                                [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')])
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    if a.effective_batch_size % (a.batch_size * world):
        raise ValueError('effective_batch_size must be divisible by batch_size * WORLD_SIZE')
    accum = a.effective_batch_size // (a.batch_size * world)
    precision = a.precision
    if precision == 'auto':
        precision = ('bf16' if torch.cuda.get_device_capability()[0] >= 8 else 'fp16') if device.type == 'cuda' else 'fp32'
    if device.type == 'cpu' and precision == 'fp16':
        raise ValueError('Use fp32 or bf16 on CPU')
    dtype = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    processor = AutoImageProcessor.from_pretrained(a.checkpoint, local_files_only=True, use_fast=False)
    teacher = ViTForImageClassification.from_pretrained(a.checkpoint, local_files_only=True,
                                                       attn_implementation='sdpa')
    if teacher.config.num_labels != 1000:
        raise ValueError('An ImageNet-1k classifier with 1000 output classes is required')
    model = ViTTransDiff(teacher, a.from_layer, a.to_layer, a.trans_depth,
                        a.mlp_ratio, a.dropout, a.activation_checkpointing,
                        a.var_head, a.head_init, a.log_std_init, a.mlp).to(device)
    if a.compile:
        model.compile_modules()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.weight_decay, fused=device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
    out = Path(a.output)
    start, best = 0, -1.
    checkpoint = None
    provenance_path = Path(a.checkpoint)/'provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if a.resume:
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('teacher_provenance') != provenance:
            raise ValueError('Teacher checkpoint provenance differs from the saved run')
        legacy = {'var_head': 'qwen', 'mlp': 'gelu'}
        for key in ['from_layer', 'to_layer', 'trans_depth', 'mlp_ratio', 'dropout', 'checkpoint', 'var_head', 'mlp']:
            if checkpoint['args'].get(key, legacy.get(key)) != getattr(a, key):
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_trainable_state(checkpoint['student'])
        if not (a.evaluate or a.baseline):
            for key in ['epochs', 'lr', 'min_lr', 'warmup_epochs', 'warmup_steps', 'weight_decay', 'effective_batch_size',
                        'steps_per_epoch', 'seed', 'lambda_nll', 'lambda_ce', 'batch_size', 'workers', 'ce_mean']:
                default = {'warmup_steps': 0, 'ce_mean': False}.get(key)
                if checkpoint['args'].get(key, default) != getattr(a, key):
                    raise ValueError(f'Training resume configuration differs: {key}')
            if checkpoint['world_size'] != world:
                raise ValueError('Resume training with the same WORLD_SIZE')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            start, best = checkpoint['epoch'] + 1, checkpoint['best_top1']
    manifest = dict(args=vars(a), precision=precision, world_size=world, accumulation_steps=accum,
                    teacher_parameters=sum(p.numel() for p in teacher.parameters()),
                    trainable_parameters=sum(p.numel() for p in trainable),
                    torch=str(torch.__version__), teacher_provenance=provenance)
    if (out / 'manifest.json').exists() and not a.resume and not (a.evaluate or a.baseline):
        raise FileExistsError('Output already contains a run; use --resume or a new --output')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        try:
            manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        print(json.dumps(manifest), flush=True)
        if not (a.evaluate or a.baseline):
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    run = start_wandb(a, manifest, rank)
    if a.evaluate or a.baseline:
        result = evaluate(model, loader(a, processor, 'validation', rank, world), device, dtype,
                          a.mc_samples, a.baseline, a.eval_mean)
        if rank == 0:
            record = dict(kind='baseline' if a.baseline else 'student', metrics=result, configuration=manifest)
            (out / ('baseline.json' if a.baseline else 'evaluation.json')).write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record), flush=True)
            if run:
                prefix = 'Baseline' if a.baseline else 'Eval'
                run.summary.update({f'{prefix}/{k}': v for k, v in result.items() if v is not None})
                run.finish()
        if world > 1:
            dist.destroy_process_group()
        return
    net = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None,
              broadcast_buffers=False) if world > 1 else model
    steps = a.steps_per_epoch or math.ceil(row_count(shard_files(a.data, 'train')) / a.effective_batch_size)
    total = steps*a.epochs
    warmup = min(a.warmup_steps, total) if a.warmup_steps else steps*min(a.warmup_epochs, a.epochs)
    for epoch in range(start, a.epochs):
        # Epoch-derived RNG makes epoch-boundary resume repeatable on the same topology.
        torch.manual_seed(a.seed + epoch * 100003 + rank)
        net.train()
        batches = iter(loader(a, processor, 'train', rank, world, epoch))
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        started = time.monotonic()
        skipped_updates = 0
        for update in range(steps):
            lr = lr_at(epoch*steps+update, total, warmup, a.lr, a.min_lr)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            for micro in range(accum):
                images, labels = next(batches)
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                values = backward_losses(net, model, images, labels, scaler,
                    lambda: amp_context(device, dtype), divisor=accum,
                    lambda_nll=a.lambda_nll, lambda_ce=a.lambda_ce,
                    split=a.split_backward, sync=micro == accum-1, ce_mean=a.ce_mean)
                sums += values.double()
            if a.clip_grad_value:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(trainable, a.clip_grad_value)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_updates += int(scaler.get_scale() < old_scale)
            if rank == 0 and ((update+1) % a.log_every == 0 or update+1 == steps):
                avg = (sums / ((update+1)*accum)).tolist()
                elapsed = time.monotonic()-started
                print(json.dumps(dict(epoch=epoch, update=update+1, updates=steps, lr=lr,
                    loss=avg, seconds=elapsed)), flush=True)
                if run:
                    run.log({'Train/loss': avg[0], 'Train/nll': avg[1], 'Train/ce': avg[2], 'Train/lr': lr,
                             'Train/images_per_sec': (update+1)*a.effective_batch_size/elapsed,
                             'Train/fp16_scale': scaler.get_scale(), 'epoch': epoch + (update+1)/steps},
                            step=epoch*steps + update + 1)
        del batches
        if world > 1:
            dist.all_reduce(sums)
        # Independent validation shards, no DDP-forward synchronization or padded duplicates.
        result = evaluate(model, loader(a, processor, 'validation', rank, world), device, dtype, a.mc_samples, mean=a.eval_mean)
        improved = result['top1'] > best
        best = max(best, result['top1'])
        if rank == 0:
            record = dict(epoch=epoch, train_loss=(sums/(steps*accum*world)).tolist(), validation=result,
                          seconds=time.monotonic()-started, skipped_fp16_updates_rank0=skipped_updates)
            if device.type == 'cuda':
                record['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated(device)/2**30
                record['peak_gpu_reserved_gib'] = torch.cuda.max_memory_reserved(device)/2**30
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            state = dict(student=model.trainable_state(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, best_top1=best,
                         args=vars(a), world_size=world, teacher_provenance=provenance)
            atomic_save(state, out/'last.pt')
            if improved:
                atomic_save(state, out/'best.pt')
            print(json.dumps(record), flush=True)
            if run:
                log = {f'Val/{k}': v for k, v in result.items() if v is not None}
                log.update({'Val/best_top1': best, 'epoch': epoch + 1, 'Train/epoch_seconds': record['seconds']})
                run.log(log, step=(epoch+1)*steps)
        if world > 1:
            dist.barrier()
    if run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
