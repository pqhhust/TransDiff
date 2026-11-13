"""Compare equivalent execution strategies on the real ViT-L and synthetic images.

Run on an allocated GPU: python -m imagenet.benchmark --batch-size 4
Measures model + optimizer memory/time, excluding image decoding and filesystem I/O.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers import ViTForImageClassification
from .backward import backward_losses
from .model import ViTTransDiff
from .run import amp_context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(Path.home()/'models/google/vit-large-patch16-224'))
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--output', default='results/vit-large-benchmark.json')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run this benchmark on an allocated CUDA GPU')
    torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    teacher = ViTForImageClassification.from_pretrained(a.checkpoint, local_files_only=True,
                                                       attn_implementation='sdpa').to(device).eval()
    images = torch.randn(a.batch_size, 3, 224, 224, device=device)
    labels = torch.arange(a.batch_size, device=device) % 1000
    results = []
    for split, cp in [(False, False), (False, True), (True, False), (True, True)]:
        torch.manual_seed(42)
        model = ViTTransDiff(teacher, 21, 24, trans_depth=3,
                            activation_checkpointing=cp).to(device).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=5e-5, weight_decay=5e-7, fused=True)
        scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
        result = dict(split_backward=split, activation_checkpointing=cp)
        try:
            def update():
                optimizer.zero_grad(set_to_none=True)
                values = backward_losses(model, model, images, labels, scaler,
                    lambda: amp_context(device, dtype), split=split)
                scaler.step(optimizer)
                scaler.update()
                return values
            update()  # Warm kernels and allocate optimizer state before timing.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(a.iterations):
                values = update()
            torch.cuda.synchronize()
            result.update(seconds_per_batch=(time.perf_counter()-start)/a.iterations,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          final_losses=values.tolist())
        except torch.cuda.OutOfMemoryError:
            result['error'] = 'CUDA out of memory'
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(gpu=torch.cuda.get_device_name(), batch_size=a.batch_size,
                  precision=str(dtype), iterations=a.iterations, results=results)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
