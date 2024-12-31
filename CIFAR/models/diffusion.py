import torch
import torch.nn as nn
from models.layers import TransformerEncoder  # Adjust if necessary
from models.DiT import DiT
from models.UNet1D import Unet1D

import math

class Diffusion_Transformer(DiT):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        class_dropout_prob=0.1,
        num_classes=10,
        learn_sigma=False,
        ViT_depth=7,
    ):
        super().__init__(input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        )
        self.ViT_depth = ViT_depth
        
    def forward(self, x, train=False):
        if not train:
            for t in range(self.ViT_depth):
                t = self.t_embedder(torch.tensor([t], device=x.device))
                for block in self.blocks:
                    x = block(x, t)
            # x = self.classification_head(x.mean(1))
            return x
        else:
            assert len(x) - 1 == self.ViT_depth
            outputs = []
            for t in range(self.ViT_depth):
                out = x[t]
                t = self.t_embedder(torch.tensor([t], device=out.device))
                for block in self.blocks:
                    out = block(out, t)
                outputs.append(out)
            return outputs

class Diffusion_UNet1D(Unet1D):
    def __init__(
        self,
        dim = 32,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2),
        channels = 384,
        dropout = 0.,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        ViT_depth = 7,
    ):
        super().__init__(dim,
            init_dim = init_dim,
            out_dim = out_dim,
            dim_mults=dim_mults,
            channels = channels,
            dropout = dropout,
            self_condition = self_condition,
            learned_variance = learned_variance,
            learned_sinusoidal_cond = learned_sinusoidal_cond,
            random_fourier_features = random_fourier_features,
            learned_sinusoidal_dim = learned_sinusoidal_dim,
            sinusoidal_pos_emb_theta = sinusoidal_pos_emb_theta,
            attn_dim_head = attn_dim_head,
            attn_heads = attn_heads
        )
        self.ViT_depth = ViT_depth
    
    def forward(self, x, train=False):
        if not train:
            x = x.transpose(1, 2)
            for t in range(self.ViT_depth):
                x = super().forward(x=x, time=t*torch.ones((x.shape[0],), device=x.device), x_self_cond=None)
            return x.transpose(1, 2)
        else:
            assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, f"Expected input list length {self.ViT_depth + 1}, got {len(x)}"
            outputs = []
            # print(f'shape of x[0] {x[0].shape}')
            for t in range(self.ViT_depth):
                out = super().forward(x=x[t].transpose(1, 2), time=t*torch.ones((x[t].shape[0],), device=x[t].device), x_self_cond=None)
                outputs.append(out.transpose(1, 2))
            return outputs
        

class Diffusion_MLP(nn.Module):
    def __init__(self, d_model=384, hdim1=128*4, hdim2=128*8, hdim3=128*4, dropout=0.1, ViT_depth=7):
        super().__init__()
        self.d_model = d_model
        self.hdim1 = hdim1
        self.hdim2 = hdim2
        self.hdim3 = hdim3
        self.dropout = dropout
        self.ViT_depth = ViT_depth
        self.ln = nn.LayerNorm(d_model)
        # Main MLP - processes concatenated input and time embedding
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hdim1),  # d_model for x, d_model for time
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim1, hdim2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim2, hdim3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim3, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.sigma = nn.Linear(d_model, 1)

    def get_timestep_embedding(self, timesteps, dim=None):
        """
        Create sinusoidal timestep embeddings.
        
        :param timesteps: tensor of shape [N] with integer timesteps
        :param dim: embedding dimension (defaults to self.d_model)
        :return: tensor of shape [N, dim]
        """
        if dim is None:
            dim = self.d_model
            
        half_dim = dim // 2
        # Create log-spaced frequencies
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=timesteps.device)
        
        # Create timestep embeddings
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

    def forward_step(self, x, t):
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape
        
        # Create sinusoidal time embedding and expand to match input dimensions
        t_emb = self.get_timestep_embedding(t)  # [batch_size, d_model]
        std = torch.clamp(torch.exp(self.sigma(t_emb)), 0, 1e-2)
        t_emb = t_emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
        
        # Now both x and t_emb have shape [batch_size, seq_len, d_model]
        x_t = x + t_emb
        
        mean_x_t = self.mlp(x_t) + x
        # Process through MLP and add residual connection
        return mean_x_t, std.view(batch_size, 1, 1).expand(batch_size, seq_len, self.d_model), mean_x_t + std.view(batch_size, 1, 1) * torch.randn_like(mean_x_t)

    def forward(self, x, train=False):
        if not train:
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
                x = self.forward_step(x, t_tensor)[-1]
            return x
        else:
            assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, \
                f"Expected input list length {self.ViT_depth + 1}, got {len(x)}"
            
            means = []
            stds = []
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x[t].device).expand(x[t].shape[0])
                mean, std, _ = self.forward_step(x[t], t_tensor)
                means.append(mean)
                stds.append(std)
            return means, stds

# class Diffusion_MLP(nn.Module):
#     def __init__(self, d_model=384, hdim1=384*2, hdim2=384*4, hdim3=384*2, dropout=0.1, ViT_depth=7):
#         super().__init__()
#         self.d_model = d_model
#         self.hdim1 = hdim1
#         self.hdim2 = hdim2
#         self.hdim3 = hdim3
#         self.dropout = dropout
#         self.ViT_depth = ViT_depth
#         self.ln = nn.LayerNorm(d_model)
#         # Main MLP - processes concatenated input and time embedding
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, hdim1),  # d_model for x, d_model for time
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hdim1, hdim2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hdim2, hdim3),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hdim3, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
#         self.mlp_mean = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
#         self.mlp_std = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )

#     def get_timestep_embedding(self, timesteps, dim=None):
#         """
#         Create sinusoidal timestep embeddings.
        
#         :param timesteps: tensor of shape [N] with integer timesteps
#         :param dim: embedding dimension (defaults to self.d_model)
#         :return: tensor of shape [N, dim]
#         """
#         if dim is None:
#             dim = self.d_model
            
#         half_dim = dim // 2
#         # Create log-spaced frequencies
#         freqs = torch.exp(
#             -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
#         ).to(device=timesteps.device)
        
#         # Create timestep embeddings
#         args = timesteps[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
#         # Handle odd dimensions
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
#         return embedding

#     def forward_step(self, x, t):
#         # Get batch size and sequence length
#         batch_size, seq_len, _ = x.shape
        
#         # Create sinusoidal time embedding and expand to match input dimensions
#         t_emb = self.get_timestep_embedding(t)  # [batch_size, d_model]
#         t_emb = t_emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
        
#         # Now both x and t_emb have shape [batch_size, seq_len, d_model]
#         x_t = x + t_emb
        
#         # Process through MLP and add residual connection
#         h = self.mlp(x_t) + x
#         mean = self.mlp_mean(h)
#         std = torch.clamp(torch.exp(self.mlp_std(h)), 0, 1e-3)
#         return mean + std * torch.randn_like(h), mean, std

#     def forward(self, x, train=False):
#         if not train:
#             for t in range(self.ViT_depth):
#                 t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
#                 x = self.forward_step(x, t_tensor)[0]
#             return x
#         else:
#             assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, \
#                 f"Expected input list length {self.ViT_depth + 1}, got {len(x)}"
            
#             means, stds = [], []
#             for t in range(self.ViT_depth):
#                 t_tensor = torch.tensor([t], device=x[t].device).expand(x[t].shape[0])
#                 _, mean, std = self.forward_step(x[t], t_tensor)
#                 means.append(mean)
#                 stds.append(std)
#             return means, stds

class MLPMixerLayer(nn.Module):
    def __init__(self, seq_len, d_model, token_mixing_dim, channel_mixing_dim, dropout=0.1):
        super().__init__()
        self.ln_token_mixing = nn.LayerNorm(d_model)
        self.token_mixing = nn.Sequential(
            nn.Linear(seq_len, token_mixing_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mixing_dim, seq_len),
            nn.Dropout(dropout),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, channel_mixing_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mixing_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # Permute for token mixing
        x = x + self.token_mixing(self.ln_token_mixing(x).permute(0, 2, 1)).permute(0, 2, 1)
        # Channel mixing
        x = x + self.channel_mixing(x)
        return x


class Diffusion_MLPMixer(nn.Module):
    def __init__(self, seq_len=64, d_model=384, token_mixing_dim=192, channel_mixing_dim=768, dropout=0.1, depth=1, ViT_depth=7):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.depth = depth
        self.ViT_depth = ViT_depth

        # Stack of MLP-Mixer layers
        self.mixer_layers = nn.ModuleList(
            [MLPMixerLayer(seq_len, d_model, token_mixing_dim, channel_mixing_dim, dropout) for _ in range(depth)]
        )

        # Layer normalization for input
        self.ln = nn.LayerNorm(d_model)

    def get_timestep_embedding(self, timesteps, dim=None):
        """
        Create sinusoidal timestep embeddings.
        
        :param timesteps: tensor of shape [N] with integer timesteps
        :param dim: embedding dimension (defaults to self.d_model)
        :return: tensor of shape [N, dim]
        """
        if dim is None:
            dim = self.d_model

        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(timesteps.device)

        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:  # Handle odd dimensions
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward_step(self, x, t):
        batch_size, seq_len, _ = x.shape

        # Sinusoidal time embedding
        t_emb = self.get_timestep_embedding(t)  # [batch_size, d_model]
        t_emb = t_emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)

        # Add time embedding to input
        x = x + t_emb

        # Pass through mixer layers
        for mixer in self.mixer_layers:
            x = mixer(x)

        return x

    def forward(self, x, train=False):
        if not train:
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
                x = self.forward_step(x, t_tensor)
            return x
        else:
            assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, \
                f"Expected input list length {self.depth + 1}, got {len(x)}"

            outputs = []
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x[t].device).expand(x[t].shape[0])
                out = self.forward_step(x[t], t_tensor)
                outputs.append(out)
            return outputs

class ViT(nn.Module):
    def __init__(self, args, attn_type, ksvd_layers=1, low_rank=10, rank_multi=10, num_classes=10, img_size=32, channels=3, \
                patch=4, dropout=0., num_layers=7, hidden=384, mlp_hidden=384, head=8, is_cls_token=False):
        super(ViT, self).__init__()
        self.attn_type = attn_type
        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*channels # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)
        self.num_layers = num_layers
        self.ksvd_layers = ksvd_layers

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(args=args, attn_type="softmax", low_rank=low_rank, rank_multi=rank_multi, embed_len=num_tokens, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        if self.attn_type == "kep_svgp":
            for i in range(self.ksvd_layers):
                enc_list[-(i+1)] = TransformerEncoder(args=args, attn_type="kep_svgp", low_rank=low_rank, rank_multi=rank_multi, embed_len=num_tokens, \
                    feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

    def forward(self, x):
        x_t = []
        score_list = []
        Lambda_inv_list = []
        kl_list = []

        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        x_t.append(out)
        for enc in self.enc:
            if enc.attn_type == "softmax":
                out = enc(out)
                x_t.append(out)
            elif enc.attn_type == "kep_svgp":
                out, scores, Lambda_inv, kl = enc(out)
                score_list.append(scores)
                Lambda_inv_list.append(Lambda_inv)
                kl_list.append(kl)
                x_t.append(out)
        
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)

        if self.attn_type == "softmax":
            return out
        elif self.attn_type == "kep_svgp":
            return out, score_list, Lambda_inv_list, kl_list, x_t

def vit_cifar(args, attn_type, num_classes, ksvd_layers, low_rank, rank_multi):
    return ViT(args=args, attn_type=attn_type, ksvd_layers=ksvd_layers, num_classes=num_classes, low_rank=low_rank, rank_multi=rank_multi, \
                img_size=32, patch=8, dropout=0.1, num_layers=args.depth, hidden=args.hdim, head=args.num_heads, mlp_hidden=args.hdim, is_cls_token=False) 