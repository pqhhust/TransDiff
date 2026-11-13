"""Streaming classification metrics; no N x 1000 prediction buffer."""
import torch
import torch.distributed as dist
import numpy as np
from CIFAR.utils.metrics import calc_aurc_eaurc, calc_fpr_aupr


class Metrics:
    def __init__(self, device, bins=15):
        self.bins = bins
        self.sums = torch.zeros(6 + 3 * bins, dtype=torch.float64, device=device)
        self.confidence_correct = []  # Two scalars per image, not 1000 probabilities.

    def update(self, probs, labels):
        p = probs.float()
        n = len(labels)
        chosen = p.gather(1, labels[:, None]).squeeze(1)
        conf, pred = p.max(1)
        correct = pred.eq(labels)
        self.confidence_correct.append(torch.stack([conf, correct.float()], 1).cpu())
        top5 = p.topk(min(5, p.shape[1]), dim=1).indices.eq(labels[:, None]).any(1)
        brier = p.square().sum(1) - 2 * chosen + 1
        vals = [n, correct.sum(), top5.sum(), -chosen.clamp_min(1e-30).log().sum(),
                brier.sum(), -(p * p.clamp_min(1e-30).log()).sum()]
        self.sums[:6] += torch.as_tensor(vals, device=p.device, dtype=torch.float64)
        bucket = (torch.ceil(conf * self.bins).long() - 1).clamp(min=0, max=self.bins - 1)
        for j, weights in enumerate([torch.ones_like(conf), conf, correct.float()]):
            self.sums[6 + j*self.bins:6 + (j+1)*self.bins].scatter_add_(0, bucket, weights.double())

    def compute(self, distributed=True):
        s = self.sums.clone()
        if distributed and dist.is_initialized():
            dist.all_reduce(s)
        n = s[0].item()
        if not n:
            raise ValueError('No evaluation examples')
        counts, confidence, correct = s[6:].reshape(3, self.bins)
        ece = (confidence - correct).abs().sum() / n
        result = dict(examples=int(n), top1=100*s[1].item()/n, top5=100*s[2].item()/n,
                    nll=s[3].item()/n, brier=s[4].item()/n,
                    entropy=s[5].item()/n, ece=100*ece.item())
        rows = torch.cat(self.confidence_correct).numpy() if self.confidence_correct else np.empty((0, 2))
        if distributed and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, rows)
            rows = np.concatenate(gathered)
        conf, good = rows[:, :1], rows[:, 1].astype(bool)
        if good.any():
            aurc, eaurc = calc_aurc_eaurc(conf, good)
        else:
            aurc, eaurc = 1., 0.
        result.update(aurc=float(aurc), eaurc=float(eaurc))
        if good.any() and not good.all():
            auroc, aupr_success, aupr_error, fpr95 = calc_fpr_aupr(conf, good)
            result.update(auroc=float(auroc), aupr_success=float(aupr_success),
                          aupr_error=float(aupr_error), fpr95=float(fpr95))
        else:
            result.update(auroc=None, aupr_success=None, aupr_error=None, fpr95=None)
        return result
