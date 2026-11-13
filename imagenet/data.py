"""Read shared Parquet directly: no extraction or HF Arrow cache in home."""
import io
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms as T


def shard_files(root, split):
    files = sorted((Path(root) / 'data').glob(f'{split}-*.parquet'))
    if not files:
        raise FileNotFoundError(f"No {split} Parquet shards in {root}/data")
    return files


def row_count(files):
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


class ImageNetStream(IterableDataset):
    def __init__(self, root, split, processor, rank=0, world_size=1, seed=0,
                 epoch=0, limit=0, repeat=False):
        self.files = shard_files(root, split)
        self.rank, self.world_size = rank, world_size
        self.seed, self.epoch, self.limit, self.repeat = seed, epoch, limit, repeat
        self.processor = processor
        size = processor.size.get('height', processor.size.get('shortest_edge', 224))
        self.transform = T.Compose([
            T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(), T.ToTensor(),
            T.Normalize(processor.image_mean, processor.image_std),
        ]) if split == 'train' else None

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        torch.manual_seed(self.seed + self.epoch * 100003 + self.rank * 1009 + wid)
        rng = random.Random(self.seed + self.epoch)
        files = list(self.files)
        if self.transform:
            rng.shuffle(files)
        # Rank first, then worker: every validation example is visited once.
        files = files[self.rank::self.world_size][wid::workers]
        if not files:
            if self.repeat:
                raise ValueError('More distributed workers than training shards')
            return
        quota = None
        if self.limit:
            rank_quota = self.limit // self.world_size + int(self.rank < self.limit % self.world_size)
            quota = rank_quota // workers + int(wid < rank_quota % workers)
        emitted = 0
        while True:
            for path in files:
                pf = pq.ParquetFile(path)
                try:
                    for batch in pf.iter_batches(batch_size=32, columns=['image', 'label'], use_threads=False):
                        rows = batch.to_pylist()
                        if self.transform:
                            rng.shuffle(rows)
                        for row in rows:
                            if quota is not None and emitted >= quota:
                                return
                            label = int(row['label'])
                            if not 0 <= label < 1000:
                                raise ValueError(f'Invalid ImageNet label {label}; do not use unlabeled test split')
                            raw = row['image']
                            if not raw.get('bytes'):
                                raise ValueError(f'Missing embedded image in {path}')
                            with Image.open(io.BytesIO(raw['bytes'])) as im:
                                im = im.convert('RGB')
                                x = (self.transform(im) if self.transform else
                                     self.processor(images=im, return_tensors='pt')['pixel_values'][0])
                            yield x, label
                            emitted += 1
                finally:
                    pf.close()
            if not self.repeat:
                return
            rng.shuffle(files)
