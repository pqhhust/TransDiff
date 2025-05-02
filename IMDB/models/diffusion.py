import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_imdb import PositionalEncoding
from models.layers import TransformerEncoder
import math
from models.DiT import DiT

class Diffusion_Transformer(nn.Module):
    def __init__(
        self,
        args,
        vocab_size,
        d_model=384,
        depth=1,
        num_heads=12,
        mlp_ratio=1.0,
        dropout=0.1,
        ViT_depth=7,
        nb_cls=10
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ViT_depth = ViT_depth
        self.max_len = args.max_len
        self.emb_dim = args.emb_dim
        self.nb_cls = nb_cls
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_encoder = PositionalEncoding(self.emb_dim, self.dropout, self.max_len)
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
            x = self.embedding.forward(x)
            x = self.pos_encoder(x)
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
                means.append(mean)
                stds.append(std)
            return means, stds

# class Embeddings(torch.nn.Module):
#     def __init__(self, vocab_size, max_len, emb_size, h_size, drop_rate):
#         super(Embeddings,self).__init__()
#         self.token_embeds=nn.Embedding(vocab_size,emb_size,padding_idx=0)
#         self.pos_embeds=nn.Embedding(max_len,emb_size+1024)
#         self.layer_norm=nn.LayerNorm(h_size)
#         self.project=nn.Linear(emb_size+1024,h_size)
#         self.dropout = nn.Dropout(drop_rate)
#         self.emb_size=emb_size
#         self.h_size = h_size
#         options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
#         weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#         num_rep=1
#         self.elmo=Elmo(options_file,weight_file,num_rep,dropout=0.)

#     def forward(self,input_data,pos,data=None):
#         pos=self.pos_embeds(pos)
#         character_ids=batch_to_ids(data).cuda()
#         rep=self.elmo(character_ids)['elmo_representations'][0]
#         rep2=self.token_embeds(input_data)
#         rep=torch.cat([rep,rep2],dim=-1)
#         output=rep+pos 
#         shape_o = output.shape
#         output = output.reshape(-1,self.emb_size+1024)
#         res=self.project(output)
#         res = self.dropout(res)
#         output=res.reshape((shape_o[0],shape_o[1],self.h_size))
#         return output

# class Diffusion_MLP(nn.Module):
#     def __init__(self, args, vocab_size, d_model=384, hdim1=64, hdim2=64, hdim3=64, hdim4=64, dropout=0, clip=0.01, ViT_depth=7, nb_cls=10):
#         super().__init__()
#         self.args = args
#         self.d_model = d_model
#         self.hdim1 = hdim1
#         self.hdim2 = hdim2
#         self.hdim3 = hdim3
#         self.hdim4 = hdim4
#         self.dropout = dropout
#         self.clip = clip
#         self.ViT_depth = ViT_depth
#         self.max_len = args.max_len
#         self.emb_dim = args.emb_dim
#         self.nb_cls = nb_cls
#         self.vocab_size = vocab_size

#         self.embedding = Embeddings(vocab_size=self.vocab_size, max_len=self.max_len, emb_size=self.emb_dim, \
#                                     h_size=self.d_model, drop_rate=self.dropout)
#         # Main MLP - processes concatenated input and time embedding
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(d_model, hdim1),  # d_model for x, d_model for time
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hdim1, hdim2),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hdim2, hdim3),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hdim3, 2*d_model),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout)
#         # )
#         self.share_params = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, hdim1),  # d_model for x, d_model for time
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hdim1, hdim2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hdim2, hdim3),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             # nn.Linear(hdim3, 2*d_model),
#             # nn.ReLU(),
#             # nn.Dropout(dropout)
#         )
#         self.mean_model = nn.Sequential(
#             nn.Linear(hdim3, d_model),  
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             # nn.Linear(hdim4, d_model),  
#             # nn.ReLU(),
#             # nn.Dropout(dropout),
#         )
        
#         self.var_model = nn.Sequential(
#             nn.Linear(hdim3, d_model),  
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             # nn.Linear(hdim4, d_model),  
#             # nn.ReLU(),
#             # nn.Dropout(dropout),
#         )
        
#         self.ln = nn.LayerNorm(d_model)
#         self.solution_head_1 = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )

#         self.solution_head_2 = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, nb_cls)
#         )

#         # self.sigma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
#         # self.sigma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
#         # self.sigma = nn.Sequential(
#         #     nn.Linear(d_model, hdim1),  # d_model for x, d_model for time
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hdim1, hdim2),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hdim2, hdim3),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hdim3, d_model),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout)
#         # )
         
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

#     def _to_words(self, x):
#         """
#         (b, c, h, w) -> (b, n, f)
#         """
#         out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
#         out = out.reshape(x.size(0), self.patch**2 ,-1)
#         return out

#     def forward_step(self, x, t):
#         # Get batch size and sequence length
#         batch_size, seq_len, _ = x.shape
        
#         # Create sinusoidal time embedding and expand to match input dimensions
#         t_emb = self.get_timestep_embedding(t)  # [batch_size, d_model]
#         t_emb = t_emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
        
#         # Now both x and t_emb have shape [batch_size, seq_len, d_model]
#         x_t = x + t_emb
        
#         ### 2 separate models for mean and var
#         # if self.args.attn_type == 'softmax':
#         #     std = 0
#         # else:
#         #     std = self.sigma(x_t)
#         # mean_x_t = self.mlp(x_t) + x
        
#         ### An unified MLP for mean and var
#         # output = self.mlp(x_t)  # [batch_size, seq_len, 2 * d_model]

#         # # Split the output into mean and std
#         # mean, std = torch.split(output, self.d_model, dim=-1)

#         # if self.args.attn_type == 'softmax':
#         #     std = 0
            
#         # # Add residual connection to mean
#         # mean_x_t = mean + x
        
#         ### Share and private branches for mean and var
#         latent = self.share_params(x_t)
        
#         mean_x_t = self.mean_model(latent) + x
#         if self.args.attn_type == 'softmax':
#             std = 0
#         else:
#             std = self.var_model(latent)
            
#         return mean_x_t, std, mean_x_t + std * torch.randn_like(mean_x_t)

#     def forward(self, x, positional=None, data=None, train=False):
#         if not train:
#             x = self.embedding.forward(x, positional, data)
#             for t in range(self.ViT_depth):
#                 t_tensor = torch.tensor([t], device=x.device).expand(x.shape[0])
#                 x = self.forward_step(x, t_tensor)[-1]
#             x = self.solution_head_1(self.ln(x)) + x
#             return self.solution_head_2(x.mean(1))
#         else:
#             assert isinstance(x, list) and len(x) - 1 == self.ViT_depth, \
#                 f"Expected input list length {self.ViT_depth + 1}, got {len(x)}"
            
#             means = []
#             stds = []
#             for t in range(self.ViT_depth):
#                 t_tensor = torch.tensor([t], device=x[t].device).expand(x[t].shape[0])
#                 mean, std, _ = self.forward_step(x[t], t_tensor)
#                 means.append(mean)
#                 stds.append(std)
#             return means, stds