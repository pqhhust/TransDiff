"""CE of the original pretrained ViT-L on train (training augmentation and clean) and validation.
python -m imagenet.teacher_loss --limit 20000"""
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification
from .data import ImageNetStream

def run(model, ds, workers, dev):
    tot = n = correct = 0.
    for x, y in DataLoader(ds, batch_size=128, num_workers=workers):
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            lg = model(x).logits.float()
        tot += torch.nn.functional.cross_entropy(lg, y, reduction='sum').item()
        correct += (lg.argmax(1) == y).sum().item(); n += len(y)
    return dict(examples=int(n), ce=tot / n, top1=100 * correct / n)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=20000); p.add_argument('--workers', type=int, default=7)
    p.add_argument('--checkpoint', default=str(Path.home()/'models/google/vit-large-patch16-224'))
    p.add_argument('--data', default='/projects/public/huggingface/datasets/ILSVRC/imagenet-1k')
    a = p.parse_args(); dev = torch.device('cuda')
    proc = AutoImageProcessor.from_pretrained(a.checkpoint, use_fast=False)
    m = ViTForImageClassification.from_pretrained(a.checkpoint, attn_implementation='sdpa').to(dev).eval()
    out = {}
    torch.manual_seed(0)
    aug = ImageNetStream(a.data, 'train', proc, limit=a.limit)              # same transform as student training
    out['train_augmented'] = run(m, aug, a.workers, dev); print(out, flush=True)
    clean = ImageNetStream(a.data, 'train', proc, limit=a.limit); clean.transform = None  # processor eval transform
    out['train_clean'] = run(m, clean, a.workers, dev); print(out, flush=True)
    out['validation'] = run(m, ImageNetStream(a.data, 'validation', proc, limit=a.limit), a.workers, dev)
    print('RESULT', json.dumps(out), flush=True)
