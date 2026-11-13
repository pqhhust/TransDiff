"""Frozen ViT + Qwen-style shared stochastic transition and Gaussian NLL."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from CIFAR.models.DiT import DiT


def hidden(output):
    # Transformers 4 returns tuples; Transformers 5 returns tensors.
    return output[0] if isinstance(output, tuple) else output


def vit_layers(teacher):
    vit = teacher.vit
    return vit.layers if hasattr(vit, 'layers') else vit.encoder.layer


def gaussian_nll(targets, means, log_stds):
    """Qwen's element/layer-averaged negative log likelihood (constant omitted).

    Accept log standard deviation to avoid log(exp(s)**2) under AMP.
    Negative values are valid for a continuous density.
    """
    if not targets or not (len(targets) == len(means) == len(log_stds)):
        raise ValueError("Exactly one teacher target per stochastic transition is required")
    terms = []
    for target, mean, log_std in zip(targets, means, log_stds):
        if target.shape != mean.shape or mean.shape != log_std.shape:
            raise ValueError("Teacher/student shapes must match without broadcasting")
        s = log_std.float()
        terms.append((s + 0.5 * ((target.float() - mean.float()) * (-s).exp()).square()).mean())
    return torch.stack(terms).mean()


class ViTTransDiff(nn.Module):
    def __init__(self, teacher, from_layer=9, to_layer=12, trans_depth=3,
                 mlp_ratio=4.0, dropout=0.1, activation_checkpointing=False,
                 var_head='qwen', head_init='default', log_std_init=-2.0, mlp='gelu'):
        super().__init__()
        n = teacher.config.num_hidden_layers
        if not 0 <= from_layer < to_layer <= n:
            raise ValueError(f"Require 0 <= from_layer < to_layer <= {n}")
        self.teacher = teacher.requires_grad_(False).eval()
        self.from_layer, self.to_layer = from_layer, to_layer
        self.activation_checkpointing = activation_checkpointing
        d = teacher.config.hidden_size
        self.share_params = DiT(hidden_size=d, depth=trans_depth,
                                num_heads=teacher.config.num_attention_heads, mlp_ratio=mlp_ratio,
                                mlp_type=mlp)
        def head():
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))
        self.mean_model = head()
        if var_head == 'qwen':
            # Reference: exp(GELU(.)) bounds std below by exp(-0.17) ~= 0.84.
            self.var_model = head()
        elif var_head == 'linear':
            # Unbounded log-std; needed when per-layer updates are smaller than 0.84.
            self.var_model = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        else:
            raise ValueError(f'Unknown var_head {var_head}')
        if head_init == 'zero':
            # Start at mean = DiT(x) = x (DiT blocks are adaLN-zero identities).
            nn.init.zeros_(self.mean_model[1].weight)
            nn.init.zeros_(self.mean_model[1].bias)
            if var_head == 'linear':
                nn.init.zeros_(self.var_model[1].weight)
                nn.init.constant_(self.var_model[1].bias, log_std_init)
        elif head_init != 'default':
            raise ValueError(f'Unknown head_init {head_init}')

    def compile_modules(self, **kw):
        """torch.compile the frozen prefix blocks and the shared DiT (same math)."""
        layers = vit_layers(self.teacher)
        for layer in layers[:self.from_layer]:
            layer.compile(**kw)
        self.share_params.compile(**kw)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # No dropout drift in frozen prefix, tail or targets.
        return self

    @torch.no_grad()
    def encode(self, images, targets=False):
        """Compute prefix once; retain only selected targets, not all 12 states.

        As in Qwen, retain the final replaced block's FFN after the stochastic
        chain. Thus intermediate targets are block outputs; the final target
        is the pre-FFN attention residual. Applying the retained FFN once then
        recovers the exact teacher boundary (tested with teacher substitution).
        """
        x = self.teacher.vit.embeddings(images)
        layers = vit_layers(self.teacher)
        for layer in layers[:self.from_layer]:
            x = hidden(layer(x))
        entry = x
        if not targets:
            return entry, []
        states = []
        for i in range(self.from_layer, self.to_layer):
            if i == self.to_layer - 1:
                # Capture the real HF attention residual without reimplementing attention.
                handle = layers[i].layernorm_after.register_forward_pre_hook(
                    lambda module, inputs: states.append(inputs[0]))
                try:
                    x = hidden(layers[i](x))
                finally:
                    handle.remove()
            else:
                x = hidden(layers[i](x))
                states.append(x)
        return entry, states

    def finish(self, x):
        # Frozen weights still participate in autograd w.r.t. the sampled states.
        layer = vit_layers(self.teacher)[self.to_layer - 1]
        z = layer.layernorm_after(x)
        if hasattr(layer, 'mlp'):  # Transformers 5
            x = x + layer.dropout(layer.mlp(z))
        else:  # Transformers 4
            x = layer.output(layer.intermediate(z), x)
        for layer in vit_layers(self.teacher)[self.to_layer:]:
            x = hidden(layer(x))
        return self.teacher.classifier(self.teacher.vit.layernorm(x)[:, 0])

    def step(self, x, t, sample=True):
        if self.activation_checkpointing and self.training:
            z = checkpoint(self.share_params, x, t, use_reentrant=False)
        else:
            z = self.share_params(x, t)
        mean = self.mean_model(z) + z
        # Same Qwen head and exp(log_std), with bounds solely for numerical safety.
        log_std = self.var_model(z).float().clamp(-10, 5)
        out = mean + (log_std.exp() * torch.randn_like(mean)).to(mean.dtype) if sample else mean
        return out, mean, log_std

    def from_entry(self, entry, sample=True, classify=True):
        x, means, logs = entry, [], []
        for t in range(self.to_layer - self.from_layer):
            times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            x, mean, log_std = self.step(x, times, sample)
            means.append(mean)
            logs.append(log_std)
        return self.finish(x) if classify else None, means, logs

    def forward(self, images=None, with_targets=False, sample=True, entry=None,
                targets=None, classify=True):
        if entry is None:
            entry, targets = self.encode(images, targets=with_targets)
        logits, means, logs = self.from_entry(entry, sample=sample, classify=classify)
        nll = gaussian_nll(targets, means, logs) if with_targets else entry.new_zeros(())
        return logits, nll

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('teacher.')}

    def load_trainable_state(self, state):
        expected = set(self.trainable_state())
        if set(state) != expected:
            raise ValueError(f"Student state mismatch: {set(state) ^ expected}")
        self.load_state_dict(state, strict=False)
