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


class SGP_LAYER(nn.Module):
    """SGPA Attention modified for q_distribution with covariance computation"""
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate, flag_sgp=True, inference_mode=True):
        super(SGP_LAYER, self).__init__()
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
        
        return attn_out, kl, mean_out, covariance_out


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


class ViT(torch.nn.Module):
    def __init__(self, device, depth, patch_size, in_channels, max_len, num_class, hdim, num_heads, sample_size, jitter, drop_rate, keys_len, kernel_type, flag_sgp, inference_mode=False):
        super(ViT, self).__init__()
        self.hdim = hdim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.max_len = max_len
        self.num_class = num_class
        self.sample_size=sample_size
        self.depth = depth
        self.jitter = jitter
        self.flag_sgp = flag_sgp
        if not self.flag_sgp:
            self.sample_size = 1
        self.keys_len = keys_len
        self.kernel_type = kernel_type
        self.drop_rate = drop_rate
        self.inference_mode = inference_mode

        self.patch_embedding = Patch_embedding(patch_size=patch_size, in_channels=in_channels, hdim=hdim, max_len=max_len, drop_rate=drop_rate)
        
        self.class_head = ClassficationHead_vit(hdim=hdim, num_class=num_class)

        self.device = device

        self.ln = nn.LayerNorm(hdim)

        self.keys = nn.ParameterList([nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.hdim), dtype=torch.float32)) for i in range(self.depth)])

        self.sgp_layer_list = nn.ModuleList([SGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate, \
            keys_len=self.keys_len, sample_size=self.sample_size, jitter=jitter, flag_sgp=flag_sgp, inference_mode=self.inference_mode)])
        self.mlp_layer_list = nn.ModuleList([FC(hdim=hdim, drop_rate=self.drop_rate)])

        for i in range(1, depth):
            self.sgp_layer_list.append(SGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim,\
                kernel_type=self.kernel_type, drop_rate=self.drop_rate, keys_len=self.keys_len, sample_size=1, jitter=jitter, flag_sgp=flag_sgp, inference_mode=self.inference_mode))
            self.mlp_layer_list.append(FC(hdim=hdim, drop_rate=self.drop_rate))

    def forward(self, X):
        x_t = []
        means = []
        covariances = []
        patch_emb_ln, patch_emb = self.patch_embedding.forward(X) 
        x_t.append(patch_emb)
        z, total_kl, mean, covariance = self.sgp_layer_list[0].forward(patch_emb_ln, self.keys[0])
        
        z_prime = patch_emb + z
        # print(z_prime.shape, z.shape, patch_emb.shape)
        mean = patch_emb + mean
        x_t.append(z_prime)
        means.append(mean)
        covariances.append(covariance)
        z_ln = self.ln(z_prime)
        
        z = self.mlp_layer_list[0].forward(z_ln) + z_prime 

        cur_k = None
        if self.flag_sgp:
            cur_k = self.mlp_layer_list[0].forward(self.keys[1]) + self.keys[1] 
        for i in range(1, self.depth):
            z_prev = z.reshape(-1, z.shape[-2], z.shape[-1]) 
            z_ln = self.ln(z_prev) 
            if self.flag_sgp:
                cur_k = self.ln(cur_k) 
            z, kl, mean, covariance = self.sgp_layer_list[i].forward(z_ln, cur_k)
            if self.flag_sgp and not self.inference_mode:
                total_kl += kl
            # print(z.shape, z_prev.shape)
            z_prime = z_prev + z
            mean = z_prev + mean
            # print(f'Layer {i}, z_prime shape: {z_prime.shape}')
            # print(f'Layer {i}, mean shape: {mean.shape}')
            # print(f'Layer {i}, z_prev shape: {z_prev.shape}')
            x_t.append(z_prime)
            means.append(mean)
            covariances.append(covariance)
            z_ln = self.ln(z_prime)  
            z = self.mlp_layer_list[i].forward(z_ln) + z_prime  
            if self.flag_sgp and i < self.depth-1:
                cur_k = self.mlp_layer_list[i].forward(self.keys[i+1]) + self.keys[i+1] 
            
        # logits = self.class_head.forward(z).squeeze(1) 
        return None, x_t, means, covariances
    def loss(self, X, y, anneal_kl=1.):
        logits, total_kl = self.forward(X)
        ce_loss = nn.CrossEntropyLoss()
        y = torch.unsqueeze(y,1)
        y = torch.tile(torch.unsqueeze(y, 1), (1, self.sample_size, 1)).view(-1, y.shape[1])
        neg_ElogPyGf = ce_loss(logits.view(-1, self.num_class), y.view(-1))
        if self.flag_sgp and total_kl.item() > 0:
            loss = neg_ElogPyGf + anneal_kl* total_kl
        else:
            loss = neg_ElogPyGf
        return loss


def vit_sgpa_q_distribution(args, device, num_classes, **kwargs):
    """Factory function to create SGPA q_distribution model"""
    return ViT(
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
