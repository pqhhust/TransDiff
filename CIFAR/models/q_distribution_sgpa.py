import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.random as npr
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


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
        super(ClassficationHead_vit, self).__init__()
        self.hdim = hdim
        self.num_class = num_class
        self.fc = nn.Linear(hdim, num_class)
        self.seqpool = nn.Linear(hdim, 1)
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x): 
        # Pooling strategy as in https://arxiv.org/abs/2104.05704 
        res = self.seqpool(x).permute(0,1,3,2) 
        res = torch.softmax(res, -1) 
        res = res @ x 
        res = torch.mean(res, 2) 
        res = self.ln(res)
        res = self.fc(res) 
        return res


class SGPAttention_QDistribution(nn.Module):
    """SGPA Attention modified for q_distribution with covariance computation"""
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate):
        super(SGPAttention_QDistribution, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.hdim = hdim
        self.vdim = self.hdim // self.num_heads
        self.dq = self.vdim
        self.keys_len = keys_len
        self.drop_rate = drop_rate
        self.K_k_beta_k_beta = None
        
        if kernel_type == 'exponential':
            self.log_sf = nn.Parameter(-4. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32)) 
            self.log_ls = nn.Parameter(4. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        elif kernel_type == 'ard':
            self.log_sf = nn.Parameter(0. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32))
            self.log_ls = nn.Parameter(0. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        self.fc_qkv = nn.Linear(self.hdim, 2* self.num_heads* self.vdim, bias=False)
        
        # SGPA-specific parameters
        self.v = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.vdim), dtype=torch.float32))
        self.s_sqrt_ltri = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len, self.keys_len), dtype=torch.float32))
        self.log_s_sqrt_diag = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len), dtype=torch.float32))
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
      
    def get_q_k_v_ssqrt(self, x, cur_k):
        q, v_gamma = self.fc_qkv(x).view(x.shape[0], x.shape[1], self.num_heads, 2* self.vdim).permute(0,2,1,3).chunk(chunks=2, dim=-1)
        k_gamma = q  # For SGPA, we typically set k = q
        
        W_qk = self.fc_qkv.weight[:self.hdim]
        k_beta = W_qk.view(self.num_heads, 1, 1, self.vdim, self.hdim) @ cur_k.unsqueeze(-1) 
        k_beta = k_beta.squeeze(-1).permute(1,0,2,3) 
        v_beta = self.v.permute(1,0,2,3)
        log_ssqrt = self.log_s_sqrt_diag.permute(1,0,2,3) 
        
        return q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt  
        
    def forward(self, x, cur_k):
        q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt = self.get_q_k_v_ssqrt(x, cur_k)
            
        if self.kernel_type == 'exponential':
            K_qq, K_qk_beta = kernel_exp(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                self.log_ls, self.log_sf).tensor_split([self.max_len,],-1)
            K_k_beta_k_gamma = K_qk_beta.permute(0,1,3,2)

            if self.K_k_beta_k_beta != None:
                K_k_beta_k_beta = self.K_k_beta_k_beta
            else:
                K_k_beta_k_beta = kernel_exp(k_beta, k_beta, self.log_ls, self.log_sf)
                self.K_k_beta_k_beta = K_k_beta_k_beta
        elif self.kernel_type == 'ard':
            K_qq, K_qk_beta = kernel_ard(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                self.log_ls, self.log_sf).tensor_split([self.max_len,],-1)
            K_k_beta_k_gamma = K_qk_beta.permute(0,1,3,2)

            if self.K_k_beta_k_beta != None:
                K_k_beta_k_beta = self.K_k_beta_k_beta
            else:
                K_k_beta_k_beta = kernel_ard(k_beta, k_beta, self.log_ls, self.log_sf)
                self.K_k_beta_k_beta = K_k_beta_k_beta
        
        K_qk_gamma = K_qq
        
        # Compute SGPA covariance similar to KEP-SVGP approach
        s_sqrt = torch.exp(log_ssqrt) 
        s_sqrt_diag = torch.diag_embed(s_sqrt) 
        s_sqrt_local = s_sqrt_diag + torch.tril(self.s_sqrt_ltri.permute(1,0,2,3,4), diagonal=-1)

        jitter = self.jitter
        while True:
            try:
                chol_K_kk = torch.linalg.cholesky(K_k_beta_k_beta + jitter* torch.eye(K_k_beta_k_beta.shape[2], device=self.device))
                break
            except Exception:
                jitter = jitter * 10

        v1 = torch.linalg.solve_triangular(chol_K_kk, K_k_beta_k_gamma, upper=False)
        v2 = torch.linalg.solve_triangular(chol_K_kk, K_k_beta_k_gamma @ v_gamma, upper=False)
        v3 = v1.unsqueeze(2).permute(0,1,2,4,3) @ s_sqrt_local

        # Compute mean
        mean1 = K_qk_gamma @ v_gamma
        mean = mean1 - v1.permute(0,1,3,2) @ v2 + K_qk_beta @ v_beta 
        
        # Compute covariance approximation similar to KEP-SVGP
        # Diagonal approximation of the full covariance
        chol_covar1 = torch.diagonal(K_qq.unsqueeze(2), dim1=3, dim2=4).permute(0,1,3,2).unsqueeze(2)
        chol_covar2 = v3.pow(2).sum(-1).permute(0,1,3,2).unsqueeze(2) - \
            v1.unsqueeze(2).permute(0,1,2,4,3).pow(2).sum(-1).permute(0,1,3,2).unsqueeze(2)
        chol_covar = (chol_covar1 + chol_covar2).pow(0.5)
        
        # Generate samples for output
        samples = mean.unsqueeze(2) + chol_covar * torch.randn((mean.shape[0], mean.shape[1], self.sample_size, mean.shape[2], mean.shape[3]), device=self.device)   
        samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
        attn_out = self.W_O(samples)
        
        # Average over samples if multiple samples
        if attn_out.dim() == 4:
            attn_out = attn_out.mean(dim=1)
        
        # Compute mean and covariance outputs using KEP-SVGP strategy
        mean_out = torch.flatten(mean.permute(0,2,1,3),-2,-1)
        mean_out = self.W_O(mean_out)
        
        # Covariance approximation using vector of ones (KEP-SVGP strategy)
        # Use the same v3 structure but with ones instead of random sampling
        covariance_approx = chol_covar * torch.ones((mean.shape[0], mean.shape[1], 1, mean.shape[2], mean.shape[3]), device=self.device)
        covariance_out = torch.flatten(covariance_approx.squeeze(2).permute(0,2,1,3),-2,-1)
        covariance_out = self.W_O(covariance_out)

        # Compute KL divergence
        kl = -0.5* self.keys_len* self.vdim * self.num_heads 
        kl += 0.5* torch.mean(torch.sum(s_sqrt_local.pow(2), (-1,-2,-3,-4)))            
        kl += 0.5* torch.mean(torch.sum(v_beta.permute(0,1,3,2).unsqueeze(3) @ K_k_beta_k_beta.unsqueeze(2) @ v_beta.permute(0,1,3,2).unsqueeze(4), (1,2))) 
        second_term = v2.permute(0,1,3,2).unsqueeze(3) @ v2.permute(0,1,3,2).unsqueeze(4)
        temp = v_gamma.permute(0,1,3,2).unsqueeze(3) @ mean1.permute(0,1,3,2).unsqueeze(4) - second_term
        kl += 0.5* torch.mean(torch.sum(temp, (1,2)))
        kl -= torch.mean(torch.sum(log_ssqrt, (-1, -2, -3))) 
        
        return attn_out, None, None, kl, mean_out, covariance_out


class TransformerEncoder_SGPA(nn.Module):
    def __init__(self, args, device, feats, mlp_hidden=128, head=8, dropout=0., embed_len=64, 
                 kernel_type='ard', sample_size=1, jitter=1e-6, keys_len=16):
        super(TransformerEncoder_SGPA, self).__init__()
        self.args = args
        self.device = device
        self.la1 = nn.LayerNorm(feats)
        self.msa = SGPAttention_QDistribution(
            device=device, 
            num_heads=head, 
            max_len=embed_len, 
            hdim=feats, 
            kernel_type=kernel_type, 
            sample_size=sample_size, 
            jitter=jitter, 
            keys_len=keys_len, 
            drop_rate=dropout
        )
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, sgpa_key):
        out = self.la1(x)
        out, scores, Lambda_inv, kl, mean, cov = self.msa(out, sgpa_key)
        
        out = out + x
        x_t_trans = out
        out = self.mlp(self.la2(out)) + out
        mean = mean + x
        
        return out, scores, Lambda_inv, kl, x_t_trans, mean, cov


class ViT_SGPA_QDistribution(nn.Module):
    """SGPA ViT modified to work as q_distribution model"""
    def __init__(self, args, device, num_classes=10, img_size=32, channels=3, 
                 patch=4, dropout=0., num_layers=7, hidden=384, mlp_hidden=384, head=8, 
                 kernel_type='ard', sample_size=1, jitter=1e-6, keys_len=16):
        super(ViT_SGPA_QDistribution, self).__init__()
        self.args = args
        self.device = device
        self.patch = patch
        self.patch_size = img_size // self.patch
        num_tokens = self.patch ** 2
        self.num_layers = num_layers
        self.hdim = hidden
        self.num_heads = head

        # Patch embedding (equivalent to emb in original q_distribution)
        self.emb = nn.Linear((img_size//patch)**2*channels, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        
        # SGPA transformer layers
        self.enc = nn.ModuleList([
            TransformerEncoder_SGPA(
                args=args,
                device=device,
                feats=hidden, 
                mlp_hidden=mlp_hidden, 
                head=head, 
                dropout=dropout, 
                embed_len=num_tokens,
                kernel_type=kernel_type,
                sample_size=sample_size,
                jitter=jitter,
                keys_len=keys_len
            ) for _ in range(num_layers)
        ])
        
        # SGPA keys for each layer
        self.keys = nn.ParameterList([
            nn.Parameter(torch.tensor(npr.randn(head, 1, keys_len, hidden), dtype=torch.float32)) 
            for _ in range(num_layers)
        ])
        
        # Classification head (equivalent to fc in original q_distribution)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

    def _to_words(self, x):
        """Convert image to patches (equivalent to original q_distribution)"""
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out

    def forward(self, x):
        x_t = []
        score_list = []
        Lambda_inv_list = []
        kl_list = []
        means = []
        covariances = []
        
        # Patch embedding
        out = self._to_words(x)
        out = self.emb(out)
        out = out + self.pos_emb
        x_t.append(out)
        
        # Pass through SGPA transformer layers
        for i, enc in enumerate(self.enc):
            sgpa_key = self.keys[i]
            out, scores, Lambda_inv, kl, x_t_trans, mean, cov = enc(out, sgpa_key)
            
            score_list.append(scores if scores is not None else [])
            Lambda_inv_list.append(Lambda_inv if Lambda_inv is not None else [])
            kl_list.append(kl)
            x_t.append(x_t_trans)
            means.append(mean)
            covariances.append(cov)
        
        # Global average pooling and classification
        out = out.mean(1)
        out = self.fc(out)

        return out, x_t, means, covariances


def vit_sgpa_q_distribution(args, device, num_classes, **kwargs):
    """Factory function to create SGPA q_distribution model"""
    return ViT_SGPA_QDistribution(
        args=args,
        device=device,
        num_classes=num_classes,
        img_size=32,
        patch=8,
        dropout=0.1,
        num_layers=args.depth,
        hidden=args.hdim,
        head=args.num_heads,
        mlp_hidden=args.hdim,
        kernel_type='ard',
        sample_size=1,
        jitter=1e-6,
        keys_len=16
    )