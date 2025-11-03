import torch
import torch.nn as nn
import numpy.random as npr
from models.layers import TransformerEncoder  # Adjust if necessary
from models.DiT import DiT
from models.UNet1D import Unet1D

import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class Patch_embedding(torch.nn.Module):
    def __init__(self, patch_size, in_channels, hdim, max_len, drop_rate):
        super(Patch_embedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.idim = patch_size * patch_size * in_channels
        self.hdim = hdim
        self.max_len = max_len
        self.pos_emb = nn.Parameter(1e-1 * torch.tensor(npr.randn(max_len, hdim), dtype=torch.float32))  

        self.linear_proj = nn.Sequential(
            nn.Conv2d(in_channels, hdim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.ln = nn.LayerNorm(hdim)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):  
        input_emb = self.linear_proj(x)
        patch_emb = input_emb + self.pos_emb
        patch_emb = self.dropout(patch_emb)
        return self.ln(patch_emb), patch_emb
        
class Embeddings(torch.nn.Module):
    def __init__(self,vocab_size,max_len,emb_size,h_size, drop_rate):
        super(Embeddings,self).__init__()
        
        self.token_embeds=nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.pos_embeds=nn.Embedding(max_len,emb_size)
        self.layer_norm=nn.LayerNorm(h_size)
            
        self.project=nn.Linear(emb_size,h_size)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self,input_data,pos):
        rep=self.token_embeds(input_data)
        pos=self.pos_embeds(pos)
      
        output=rep+pos
        output=self.project(output)
        output = self.dropout(output)
        
        return self.layer_norm(output), output


class FC(torch.nn.Module):
    def __init__(self, hdim, drop_rate=0.):
        super(FC, self).__init__()
        self.hdim = hdim
        self.act = torch.nn.GELU() 
        self.fc = nn.Sequential(nn.Linear(hdim, hdim), nn.Dropout(drop_rate), self.act, nn.Linear(hdim,hdim), nn.Dropout(drop_rate))
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x):  
        res = self.fc(x)
        return res
    

class ClassficationHead_vit(torch.nn.Module):
    def __init__(self, hdim, num_class):
        super().__init__()
        self.seqpool = nn.Linear(hdim, 1)
        self.ln = nn.LayerNorm(hdim)
        self.fc = nn.Linear(hdim, num_class)

    def forward(self, x):                  # x: [B, L, D]
        w = self.seqpool(x).transpose(1, 2)    # [B, 1, L]
        w = torch.softmax(w, dim=-1)           # [B, 1, L]
        pooled = w @ x                         # [B, 1, D]
        pooled = pooled.squeeze(1)             # [B, D]
        return self.fc(self.ln(pooled))        # [B, num_class]

class ClassficationHead(torch.nn.Module):
    def __init__(self, hdim, num_class, drop_rate=0.):
        super(ClassficationHead, self).__init__()
        self.hdim = hdim
        self.num_class = num_class
        self.fc = nn.Sequential(nn.Linear(hdim, num_class), nn.Dropout(drop_rate))
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x, input_mask):
        input_mask = input_mask.unsqueeze(-1).unsqueeze(1)
        res = x* input_mask
        res = torch.mean(res, 2)
        res = self.ln(res)
        res = self.fc(res)
        return res

def kernel_ard(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1)
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    factor1 = torch.sum(X1.pow(2), -1)
    factor2 = torch.sum(X2.pow(2), -1)
    return torch.exp(log_sf).unsqueeze(1) * \
        torch.exp(-0.5* (factor1.unsqueeze(3) + factor2.unsqueeze(2) -2* X1 @ X2.permute(0,1,3,2)))


def kernel_exp(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1) 
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    return torch.exp(log_sf).unsqueeze(1)* torch.exp(X1 @ X2.permute(0,1,3,2))


def scale_dot(X1, X2):
    dk = X2.shape[3]
    return torch.softmax(X1 @ X2.permute(0,1,3,2)/ (math.sqrt(dk)), 3)

class Diffusion_Transformer(nn.Module):
    def __init__(
        self,
        d_model=384,
        depth=1,
        num_heads=12,
        mlp_ratio=1.0,
        dropout=0.1,
        ViT_depth=7,
        nb_cls=10
    ):
        super().__init__()
        self.ViT_depth = ViT_depth
        self.patch = 8
        self.patch_size = 4
        
        self.patch_embedding = Patch_embedding(patch_size=self.patch_size, in_channels=3, hdim=d_model, max_len=64, drop_rate=dropout)
        self.mlp = FC(hdim=d_model, drop_rate=dropout)
        self.class_head = ClassficationHead_vit(hdim=d_model, num_class=nb_cls)
        self.ln = nn.LayerNorm(d_model)
        self.share_params = DiT(hidden_size=d_model, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.mean_model = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.var_model = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(d_model)
        self.solution_head_1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.solution_head_2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, nb_cls)
        )

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

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

    def forward_step(self, x, t):
        x = self.share_params(x, t)
        
        mean_x_t = self.mean_model(x) + x
        std = self.var_model(x)
            
        return mean_x_t, std, mean_x_t + std * torch.randn_like(mean_x_t)

    def forward(self, x, train=False):
        if not train:
            x_ln, x = self.patch_embedding.forward(x)
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
                x = self.forward_step(x, t_tensor)[-1]
            x_ln = self.ln(x)
            x = self.mlp.forward(x_ln) + x
            x = self.class_head.forward(x).squeeze(1)
            return x
        else:
            assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, \
                f"Expected input list length {self.ViT_depth + 1}, got {len(x)}"
            
            means = []
            stds = []
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x[t].device).expand(x[t].shape[0])
                # print(x[t].shape, t_tensor.shape)
                mean, std, mean_plus_std = self.forward_step(x[t], t_tensor)
                means.append(mean)
                stds.append(std)
            return means, stds

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
    def __init__(self, args, d_model=384, hdim1=64, hdim2=64, hdim3=64, hdim4=64, dropout=0, clip=0.01, ViT_depth=7):
        super().__init__()
        self.args = args
        self.d_model = d_model
        self.hdim1 = hdim1
        self.hdim2 = hdim2
        self.hdim3 = hdim3
        self.hdim4 = hdim4
        self.dropout = dropout
        self.clip = clip
        self.ViT_depth = ViT_depth
        self.patch = 8
        self.patch_size = 4
        
        self.emb = nn.Linear(48, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model))
        # Main MLP - processes concatenated input and time embedding
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, hdim1),  # d_model for x, d_model for time
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hdim1, hdim2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hdim2, hdim3),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hdim3, 2*d_model),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )
        self.share_params = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hdim1),  # d_model for x, d_model for time
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim1, hdim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim2, hdim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hdim3, 2*d_model),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.mean_model = nn.Sequential(
            nn.Linear(hdim3, d_model),  
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hdim4, d_model),  
            # nn.ReLU(),
            # nn.Dropout(dropout),
        )
        
        self.var_model = nn.Sequential(
            nn.Linear(hdim3, d_model),  
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hdim4, d_model),  
            # nn.ReLU(),
            # nn.Dropout(dropout),
        )
        
        self.ln = nn.LayerNorm(d_model)
        self.solution_head_1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.solution_head_2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 10)
        )
        
    #   self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.orthogonal_(module.weight)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

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

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

    def forward_step(self, x, t):
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape
        
        # Create sinusoidal time embedding and expand to match input dimensions
        t_emb = self.get_timestep_embedding(t)  # [batch_size, d_model]
        t_emb = t_emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
        
        # Now both x and t_emb have shape [batch_size, seq_len, d_model]
        x_t = x + t_emb
        
        ### 2 separate models for mean and var
        # if self.args.attn_type == 'softmax':
        #     std = 0
        # else:
        #     std = self.sigma(x_t)
        # mean_x_t = self.mlp(x_t) + x
        
        ### An unified MLP for mean and var
        # output = self.mlp(x_t)  # [batch_size, seq_len, 2 * d_model]

        # # Split the output into mean and std
        # mean, std = torch.split(output, self.d_model, dim=-1)

        # if self.args.attn_type == 'softmax':
        #     std = 0
            
        # # Add residual connection to mean
        # mean_x_t = mean + x
        
        ### Share and private branches for mean and var
        latent = self.share_params(x_t)
        
        mean_x_t = self.mean_model(latent) + x
        # if self.args.attn_type == 'softmax':
        #     # std = torch.zeros_like(mean_x_t)
        #     std = self.var_model(latent)
        # # else:
        std = self.var_model(latent)
        # std = torch.clip(self.var_model(latent), min=-self.clip, max=self.clip)
            
        return mean_x_t, std, mean_x_t + std * torch.randn_like(mean_x_t)

    def forward(self, x, train=False):
        if not train:
            x = self._to_words(x)
            x = self.emb(x)
            x = x + self.pos_emb
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
                x = self.forward_step(x, t_tensor)[-1]
            x = self.solution_head_1(self.ln(x)) + x
            return self.solution_head_2(x.mean(1))
        else:
            assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, \
                f"Expected input list length {self.ViT_depth + 1}, got {len(x)}"
            
            means = []
            stds = []
            for t in range(self.ViT_depth):
                t_tensor = torch.tensor([t], device=x[t].device).expand(x[t].shape[0])
                mean, std, mean_plus_std = self.forward_step(x[t], t_tensor)
                # if self.args.attn_type == 'softmax':
                #     means.append(mean_plus_std)
                # else:
                #     if t < (self.args.depth - self.args.ksvd_layers):
                #         means.append(mean_plus_std)
                #     else:
                #         means.append(mean)
                means.append(mean)
                stds.append(std)
            return means, stds

class Diffusion_RNN(nn.Module):
    def __init__(self, args, d_model=384, rnn_hidden=384, rnn_num_layers=1, dropout=0.1, 
                 ViT_depth=7, low_dim=10):
        super().__init__()
        self.args = args
        self.d_model = d_model
        self.rnn_hidden = rnn_hidden
        self.num_layers = rnn_num_layers
        self.dropout = dropout
        self.ViT_depth = ViT_depth
        self.low_dim = low_dim
        self.seq_len = 64
        # For image-to-token conversion
        self.patch = 8
        self.patch_size = 4
        self.emb = nn.Linear(48, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model))
        
        # LSTM backbone (processing tokens, with time token concatenated)
        if args.backbone == 'lstm':
            self.rnn = nn.LSTM(input_size=rnn_hidden, hidden_size=rnn_hidden,   
                            num_layers=rnn_num_layers, dropout=dropout)
        elif args.backbone == 'gru':
            self.rnn = nn.GRU(input_size=rnn_hidden, hidden_size=rnn_hidden, 
                            num_layers=rnn_num_layers, dropout=dropout)
        self.proj_in = nn.Linear(d_model, rnn_hidden // self.seq_len)
        # self.proj_out = nn.Sequential(
        #     nn.Linear(rnn_hidden // self.seq_len, d_model),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model, d_model),
        #     nn.Dropout(dropout)
        # )
        # Separate branches for mean and variance predictions
        self.mean_model = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        self.var_model = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        
        self.ln = nn.LayerNorm(d_model)
        self.solution_head_1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.solution_head_2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 10)
        )
    
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

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

    def forward_step(self, x, h_c):
        # x shape: [batch_size, seq_len, d_model]
        # t = torch.tensor([t], device=x.device).expand(x.shape[0])
        # t_emb = self.get_timestep_embedding(t)  # [batch_size, d_model]
        # t_emb = t_emb.unsqueeze(1).expand(x.shape[0], x.shape[1], self.d_model)
        # x = x + t_emb
        latent, h_c = self.rnn(self.proj_in(x).flatten(-2, -1).unsqueeze(0), h_c) # shape of latent [1, batch_size, rnn_hidden]
        latent = latent.view(latent.shape[1], self.seq_len, self.rnn_hidden // self.seq_len)
        latent = latent @ self.proj_in.weight.unsqueeze(0)
        mean = self.mean_model(latent)
        std = self.var_model(latent)
        x_t = mean + std * torch.randn_like(mean)
        return mean, std, x_t, h_c

    def forward(self, x, train=False):
        if not train:
            B = x.shape[0]
            x = self._to_words(x)
            x = self.emb(x)
            x = x + self.pos_emb
            h_c = None
            for t in range(self.ViT_depth):
                x, h_c = self.forward_step(x, h_c)[-2:]
            x = self.solution_head_1(self.ln(x)) + x
            return self.solution_head_2(x.mean(1))
        else:
            assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, \
                f"Expected input list length {self.ViT_depth + 1}, got {len(x)}"
            x = torch.stack(x, dim=0)
            # t = torch.tensor(list(range(self.ViT_depth + 1)), device=x.device)
            # t = self.get_timestep_embedding(t) # shape: [ViT_depth + 1, d_model]
            # x = x + t.unsqueeze(1).unsqueeze(1).expand(self.ViT_depth + 1, x.shape[1], x.shape[2], self.d_model)
            x = self.proj_in(x).flatten(-2, -1)
            x = x[:-1] # shape: [ViT_depth, B, seq_len * low_dim]
            
            out, _ = self.rnn(x)   # shape of out: [ViT_depth, B, rnn_hidden]
            out = out.view(out.shape[0], out.shape[1], self.seq_len, self.rnn_hidden // self.seq_len)
            out = out @ self.proj_in.weight.unsqueeze(0)
            means = self.mean_model(out)
            stds = self.var_model(out)
            return means, stds

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