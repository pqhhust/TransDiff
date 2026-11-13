import copy
import io
from contextlib import nullcontext

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from imagenet.data import ImageNetStream
from imagenet.metrics import Metrics
from imagenet.model import ViTTransDiff, gaussian_nll, hidden, vit_layers
from imagenet.run import lr_at
from imagenet.backward import backward_losses


@pytest.fixture
def tiny():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    c = ViTConfig(hidden_size=24, intermediate_size=48, num_hidden_layers=4,
                  num_attention_heads=3, image_size=16, patch_size=8, num_labels=7)
    teacher = ViTForImageClassification(c).eval()
    return teacher, torch.randn(2, 3, 16, 16)


@pytest.mark.parametrize('start,end', [(0, 3), (1, 4), (1, 3)])
def test_teacher_boundary_and_target_alignment(tiny, start, end):
    teacher, images = tiny
    m = ViTTransDiff(teacher, start, end, trans_depth=1).eval()
    entry, targets = m.encode(images, targets=True)
    with torch.no_grad():
        x = teacher.vit.embeddings(images)
        states = [x]
        for layer in vit_layers(teacher):
            x = hidden(layer(x))
            states.append(x)
        torch.testing.assert_close(entry, states[start])
        assert len(targets) == end-start
        for i, target in enumerate(targets[:-1]):
            torch.testing.assert_close(target, states[start+i+1])
        # Last target is pre-FFN; its retained FFN + suffix must equal original ViT.
        torch.testing.assert_close(m.finish(targets[-1]), teacher(images).logits, atol=1e-6, rtol=1e-5)


def test_likelihood_matches_qwen_math():
    targets = [torch.randn(2, 5, 8) for _ in range(3)]
    means = [torch.randn_like(x, requires_grad=True) for x in targets]
    logs = [torch.randn_like(x).clamp(-2, 2).requires_grad_() for x in targets]
    expected = sum(.5*((s.exp().square()).log() + ((x-m)/s.exp()).square()).mean()
                   for x, m, s in zip(targets, means, logs))/3
    actual = gaussian_nll(targets, means, logs)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in means+logs)
    with pytest.raises(ValueError):
        gaussian_nll(targets[1:], means, logs)


def test_checkpointing_preserves_output_and_gradients(tiny):
    teacher, images = tiny
    a = ViTTransDiff(teacher, 1, 4, trans_depth=1, dropout=0.).train()
    b = copy.deepcopy(a)
    b.activation_checkpointing = True
    for m in [a, b]:
        torch.manual_seed(123)
        logits, nll = m(images, with_targets=True)
        loss = nll + torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))
        loss.backward()
        assert all(p.grad is None for p in m.teacher.parameters())
        assert not m.teacher.training
    torch.testing.assert_close(a.mean_model[1].weight.grad, b.mean_model[1].weight.grad)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad)
    before = b.trainable_state()
    assert not any(k.startswith('teacher.') for k in before)
    b.load_trainable_state(a.trainable_state())


def test_stream_read_only_shards_and_rank_partition(tmp_path):
    root = tmp_path/'data'
    root.mkdir()
    buf = io.BytesIO()
    Image.new('RGB', (20, 20), 'red').save(buf, format='PNG')
    for i in range(4):
        rows = [{'image': {'bytes': buf.getvalue(), 'path': None}, 'label': i*2+j} for j in range(2)]
        pq.write_table(pa.Table.from_pylist(rows), root/f'validation-{i:05d}.parquet')
    processor = ViTImageProcessor(size={'height': 16, 'width': 16})
    sets = []
    for rank in range(2):
        items = list(ImageNetStream(tmp_path, 'validation', processor, rank=rank, world_size=2))
        assert all(x.shape == (3, 16, 16) for x, _ in items)
        sets.append({label for _, label in items})
    assert sets[0].isdisjoint(sets[1])
    assert sets[0] | sets[1] == set(range(8))
    limited = sum((list(ImageNetStream(tmp_path, 'validation', processor, rank=r,
                                      world_size=2, limit=3)) for r in range(2)), [])
    assert len(limited) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ['data']


def test_metrics_probability_nll_and_ece():
    m = Metrics('cpu')
    p = torch.tensor([[.8, .2], [.4, .6]])
    y = torch.tensor([0, 0])
    m.update(p, y)
    result = m.compute()
    assert result['examples'] == 2
    assert result['top1'] == 50
    assert result['nll'] == pytest.approx(-torch.tensor([.8, .4]).log().mean().item())
    assert result['brier'] == pytest.approx(.4)
    assert result['ece'] == pytest.approx(40.)


def test_warmup_cosine():
    assert lr_at(0, 20, 5, 1., .01) == .2
    assert lr_at(4, 20, 5, 1., .01) == 1.
    assert lr_at(19, 20, 5, 1., .01) == .01


@pytest.mark.parametrize('checkpointing', [False, True])
def test_split_backward_matches_joint_with_dropout_and_noise(tiny, checkpointing):
    teacher, images = tiny
    joint = ViTTransDiff(teacher, 1, 4, trans_depth=2, dropout=.2,
                         activation_checkpointing=checkpointing).train()
    split = copy.deepcopy(joint)
    outputs, states = [], []
    for model, use_split in [(joint, False), (split, True)]:
        torch.manual_seed(918)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        values = backward_losses(model, model, images, torch.tensor([1, 2]), scaler,
                                 nullcontext, divisor=3, lambda_nll=.7, lambda_ce=1.2,
                                 split=use_split)
        outputs.append(values)
        states.append(torch.get_rng_state())
    torch.testing.assert_close(outputs[0], outputs[1])
    assert torch.equal(states[0], states[1])
    for (name, p), (_, q) in zip(joint.named_parameters(), split.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4, msg=name)


def test_ce_mean_backward_matches_manual_loss(tiny):
    teacher, images = tiny
    labels = torch.tensor([1, 2])
    a = ViTTransDiff(teacher, 1, 4, trans_depth=2, dropout=.2).train()
    b = copy.deepcopy(a)
    torch.manual_seed(5)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    values = backward_losses(a, a, images, labels, scaler, nullcontext, divisor=3,
                             lambda_nll=.7, lambda_ce=1.2, ce_mean=True)
    torch.manual_seed(5)
    entry, targets = b.encode(images, targets=True)
    logits, _ = b(entry=entry, sample=False)
    ce = torch.nn.functional.cross_entropy(logits, labels)
    _, nll = b(entry=entry, targets=targets, with_targets=True, classify=False)
    ((1.2 * ce + .7 * nll) / 3).backward()
    torch.testing.assert_close(values, torch.stack([.7 * nll + 1.2 * ce, nll, ce]).detach())
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4, msg=name)
    assert a.var_model[1].weight.grad.abs().sum() > 0  # var head still trained by NLL
