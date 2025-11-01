import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.random as npr
import math


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout= 0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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


class ClassficationHead(torch.nn.Module):
    def __init__(self, hdim, num_class, drop_rate=0.):
        super(ClassficationHead, self).__init__()
        self.hdim = hdim
        self.num_class = num_class
        self.fc = nn.Sequential(nn.Linear(hdim, num_class), nn.Dropout(drop_rate))
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x):
        res = x
        res = torch.mean(res, 2)
        res = self.ln(res)
        res = self.fc(res)
        return res


class SGPAttention_QDistribution_IMDB(nn.Module):
    """SGPA Attention modified for IMDB q_distribution with covariance computation"""
    def __init__(self, device, num_heads, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate):
        super(SGPAttention_QDistribution_IMDB, self).__init__()
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
        max_len = x.shape[1]  # Sequence length for IMDB
            
        if self.kernel_type == 'exponential':
            K_qq, K_qk_beta = kernel_exp(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                self.log_ls, self.log_sf).tensor_split([max_len,],-1)
            K_k_beta_k_gamma = K_qk_beta.permute(0,1,3,2)

            if self.K_k_beta_k_beta != None:
                K_k_beta_k_beta = self.K_k_beta_k_beta
            else:
                K_k_beta_k_beta = kernel_exp(k_beta, k_beta, self.log_ls, self.log_sf)
                self.K_k_beta_k_beta = K_k_beta_k_beta
        elif self.kernel_type == 'ard':
            K_qq, K_qk_beta = kernel_ard(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                self.log_ls, self.log_sf).tensor_split([max_len,],-1)
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


class TransformerEncoder_SGPA_IMDB(nn.Module):
    def __init__(self, args, device, feats, mlp_hidden=128, head=8, dropout=0., 
                 kernel_type='ard', sample_size=1, jitter=1e-6, keys_len=16):
        super(TransformerEncoder_SGPA_IMDB, self).__init__()
        self.args = args
        self.device = device
        self.la1 = nn.LayerNorm(feats)
        self.msa = SGPAttention_QDistribution_IMDB(
            device=device, 
            num_heads=head, 
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


class Transformer_SGPA_QDistribution_IMDB(nn.Module):
    """SGPA Transformer modified to work as q_distribution model for IMDB"""
    def __init__(self, args, device, vocab_size, num_class=2, max_len=512, 
                 dropout=0., num_layers=6, hidden=384, mlp_hidden=384, head=8, 
                 kernel_type='ard', sample_size=1, jitter=1e-6, keys_len=16):
        super(Transformer_SGPA_QDistribution_IMDB, self).__init__()
        self.args = args
        self.device = device
        self.num_layers = num_layers
        self.hdim = hidden
        self.num_heads = head

        # Embedding layer (equivalent to embedding in original q_distribution)
        self.embedding = nn.Embedding(vocab_size, hidden)
        
        # Positional encoding (equivalent to pos_encoder in original q_distribution)
        self.pos_encoder = PositionalEncoding(hidden, dropout, max_len)
        
        # SGPA transformer layers
        self.enc = nn.ModuleList([
            TransformerEncoder_SGPA_IMDB(
                args=args,
                device=device,
                feats=hidden, 
                mlp_hidden=mlp_hidden, 
                head=head, 
                dropout=dropout, 
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
        self.fc = ClassficationHead(hidden, num_class, dropout)

    def forward(self, input_ids):
        x_t = []
        score_list = []
        Lambda_inv_list = []
        kl_list = []
        means = []
        covariances = []
        
        # Embedding and positional encoding
        out = self.embedding(input_ids).transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        out = self.pos_encoder(out)
        out = out.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
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
        
        # Classification
        out = self.fc(out)

        return out, x_t, means, covariances


def transformer_sgpa_q_distribution_imdb(args, device, vocab_size, **kwargs):
    """Factory function to create SGPA q_distribution model for IMDB"""
    return Transformer_SGPA_QDistribution_IMDB(
        args=args,
        device=device,
        vocab_size=vocab_size,
        num_class=2,
        max_len=512,
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