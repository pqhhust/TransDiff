"""Step-time sweep: checkpointing x microbatch x compile, real ViT-L, synthetic images.
Also measures CPU data-loader throughput.  python -m imagenet.bench_speed"""
import argparse, json, time, os, gc
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, AutoImageProcessor
from .backward import backward_losses
from .model import ViTTransDiff
from .run import amp_context
from .data import ImageNetStream

CK = str(Path.home()/'models/google/vit-large-patch16-224')

def gpu(bs, ckpt, comp, iters=6):
    dev = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    t = ViTForImageClassification.from_pretrained(CK, attn_implementation='sdpa')
    m = ViTTransDiff(t, 21, 24, activation_checkpointing=ckpt, var_head='linear', head_init='zero').to(dev).train()
    if comp: m.compile_modules()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5, fused=True)
    sc = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    x = torch.randn(bs, 3, 224, 224, device=dev); y = torch.arange(bs, device=dev) % 1000
    torch.cuda.reset_peak_memory_stats()
    def step():
        backward_losses(m, m, x, y, sc, lambda: amp_context(dev, dtype), split=False)
        sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
    for _ in range(3): step()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters): step()
    torch.cuda.synchronize(); dt = (time.perf_counter() - s) / iters
    r = dict(bs=bs, ckpt=ckpt, compile=comp, img_per_s=bs/dt, s_per_512=512*dt/bs,
             peak_gib=torch.cuda.max_memory_allocated()/2**30)
    del m, t, opt; gc.collect(); torch.cuda.empty_cache(); torch._dynamo.reset()
    return r

def data(workers, n=3000):
    proc = AutoImageProcessor.from_pretrained(CK, use_fast=False)
    ds = ImageNetStream('/projects/public/huggingface/datasets/ILSVRC/imagenet-1k', 'train', proc, repeat=True)
    dl = DataLoader(ds, batch_size=64, num_workers=workers, prefetch_factor=4)
    it = iter(dl); next(it); s = time.perf_counter(); k = 0
    while k < n: k += len(next(it)[1])
    return dict(workers=workers, img_per_s=k/(time.perf_counter()-s))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--part', default='all'); a = p.parse_args()
    out = []
    if a.part in ('all', 'data'):
        for w in [4, 8]:
            r = data(w); print(json.dumps(r), flush=True); out.append(r)
    if a.part in ('all', 'gpu'):
        for bs, ck, cp in [(32, True, False), (32, False, False), (64, False, False), (128, False, False),
                           (64, False, True), (128, False, True)]:
            try: r = gpu(bs, ck, cp)
            except torch.OutOfMemoryError: r = dict(bs=bs, ckpt=ck, compile=cp, oom=True); torch.cuda.empty_cache()
            print(json.dumps(r), flush=True); out.append(r)
