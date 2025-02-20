import torch
import torch.nn as nn
import torch.nn.functional as F

class KEP_SVGPAttention(nn.Module):
    def __init__(self, dim, num_heads=8, embed_len=64, low_rank=10, rank_multi=10, concate=False, \
                qk_bias=False, attn_drop=0., proj_drop=0.):
        super(KEP_SVGPAttention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ## projection weights we, wr in kep_svgp attention
        self.low_rank = low_rank
        self.rank_multi = rank_multi
        self.embed_len = embed_len
        self.we = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_heads, min(self.embed_len, self.low_rank * self.rank_multi), self.low_rank)))
        self.wr = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.num_heads, min(self.embed_len, self.low_rank * self.rank_multi), self.low_rank)))
        self.log_lambda_sqrt_inv_diag = nn.Parameter(nn.init.uniform_(torch.Tensor(self.num_heads, self.low_rank)))

        ## sparse GP
        self.m_u = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.num_heads, self.low_rank, self.low_rank)))
        self.s_sqrt_low_triangle = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.num_heads, self.low_rank, self.low_rank, self.low_rank)))
        self.log_ssqrt = nn.Parameter(nn.init.normal_(torch.Tensor(1, self.num_heads, self.low_rank, self.low_rank)))
        self.final_weight = nn.Linear(self.low_rank, self.head_dim)

        self.concate = concate
        if self.concate:
            self.embed_len_weight = nn.Linear(self.embed_len * 2, self.embed_len)

    def gen_weights(self, x):
        ## evenly sample
        if self.embed_len > self.low_rank * self.rank_multi:
            indices = torch.linspace(0, x.shape[1]-1, self.low_rank * self.rank_multi, dtype=int)
            x = x.transpose(-2,-1).reshape(x.size(0), self.num_heads, self.head_dim, x.size(1))
            x = x[:, :, :, indices].transpose(1, 2)
        else:
            x = x.transpose(-2,-1).reshape(x.size(0), self.num_heads, self.head_dim, x.size(1))
            x = x.transpose(1, 2)
        we = torch.einsum('bahd,hde->bahe', x, self.we.type_as(x)).transpose(1,2)
        wr = torch.einsum('bahd,hde->bahe', x, self.wr.type_as(x)).transpose(1,2)
        return we, wr 

    def feature_map(self, x):
        ## normalization should be on dim=-1
        return F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0) # (batch_size, num_heads, seq_len, head_dim)

        we, wr = self.gen_weights(x)
        q = self.feature_map(q) 
        k = self.feature_map(k) 
        escore = torch.einsum('...nd,...de->...ne', q, we) # (batch_size, num_heads, seq_len, low_rank)
        rscore = torch.einsum('...nd,...de->...ne', k, wr) # (batch_size, num_heads, seq_len, low_rank)
        if self.concate:
            score = torch.cat((escore, rscore), dim=2) # (batch_size, num_heads, 2 * seq_len, low_rank)

        ## compute mean and covariance for the SGP
        # mean
        lambda_sqrt_inv_diag = torch.diag_embed(torch.exp(self.log_lambda_sqrt_inv_diag)) # (num_heads, low_rank, low_rank)
        if self.concate:
            v1 = score @ (lambda_sqrt_inv_diag.unsqueeze(0) ** 2) # (batch_size, num_heads, 2 * seq_len, low_rank), E_X\times\Lambda^{-1}, R_X\times\Lambda^{-1}
        else:
            v1 = (escore + rscore) @ (lambda_sqrt_inv_diag.unsqueeze(0) ** 2) # (batch_size, num_heads, seq_len, low_rank)
        mean = v1 @ self.m_u # (batch_size, num_heads, seq_len, low_rank)
        # covariance 
        s_sqrt = torch.exp(self.log_ssqrt) # (1, num_heads, low_rank, low_rank)
        s_sqrt_diag = torch.diag_embed(s_sqrt) # (1, num_heads, low_rank, low_rank, low_rank)
        s_sqrt_local = s_sqrt_diag + torch.tril(self.s_sqrt_low_triangle, diagonal=-1) # (1, num_heads, low_rank, low_rank, low_rank) 
        # choleskey factor of the covariance matrix
        # the last dimension should be the [d] dimension
        v2 = v1.unsqueeze(2) @ s_sqrt_local # (batch_size, num_heads, low_rank, 2*seq_len / seq_len, low_rank([d] dimension))

        ## samples from the approximate posterior
        if self.concate:
            samples = mean + (v2 @ torch.randn(B, self.num_heads, self.low_rank, self.low_rank, 1).to(x.device)).squeeze().permute(0, 1, 3, 2)
            covariance = (v2 @ torch.ones(B, self.num_heads, self.low_rank, self.low_rank, 1).to(x.device)).squeeze().permute(0, 1, 3, 2)
        else:
            samples = mean + (v2.permute(0,1,3,2,4) @ torch.randn(B, self.num_heads, N, mean.shape[3], 1).to(x.device)).squeeze()
        attn_out = self.final_weight(samples)
        mean = self.final_weight(mean)
        covariance = self.final_weight(covariance)
        if self.concate:
            attn_out = self.embed_len_weight(attn_out.permute(0,1,3,2)).permute(0,1,3,2)
            mean = self.embed_len_weight(mean.permute(0,1,3,2)).permute(0,1,3,2)
            covariance = self.embed_len_weight(covariance.permute(0,1,3,2)).permute(0,1,3,2)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        mean = mean.transpose(1, 2).reshape(B, N, C)
        covariance = covariance.transpose(1, 2).reshape(B, N, C)
        # attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        mean = self.proj_drop(mean)
        covariance = self.proj_drop(covariance)

        # covariance = v2 @ v2.transpose(-2, -1) # (batch_size, num_heads, low_rank, 2 * seq_len, 2 * seq_len)
        # if self.concate:
        #     covariance = self.embed_len_weight.weight @ covariance @ self.embed_len_weight.weight.transpose(-2, -1) # (batch_size, num_heads, low_rank, seq_len, seq_len)
        # covariance = self.final_weight.weight.view(1, 1, 1, 1, self.head_dim, self.low_rank) * covariance.permute(0, 1, 3, 4, 2).view(B, self.num_heads, N, N, 1, self.low_rank) @ self.final_weight.weight.view(1, 1, 1, 1, self.head_dim, self.low_rank).permute(0, 1, 2, 3, 5, 4)
        # # (batch_size, num_heads, seq_len, seq_len, head_dim, head_dim), cross-covariance matrix between two tokens
        # covariance = covariance.permute(0, 1, 2, 4, 3, 5).reshape(B, self.num_heads, N * self.head_dim, N * self.head_dim)
        # covariance = torch.diag(covariance, dim1=-2, dim2=-1).reshape(B, self.num_heads, N, self.head_dim)

        ## compute the KL divergence 
        # Tr(\Lambda^{-2}S_{uu}) term 
        # where Tr(AA^\top) = ||A||_F^2
        v3 = (lambda_sqrt_inv_diag[None,None,...] ** 2) @ s_sqrt_local.permute(0,4,1,2,3)
        kl = 0.5 * torch.sum(v3.pow(2)) 
        # m_u^\top\Lambda^{-2}m_u term:
        mu_d = self.m_u.permute(0,1,3,2).unsqueeze(-1)
        kl += 0.5 * (mu_d.permute(0,1,2,4,3) @ (lambda_sqrt_inv_diag.unsqueeze(0).unsqueeze(2) ** 4) @ mu_d).sum()
        # log(|\Lambda^2|/|S_uu|) term:
        kl -= torch.sum(self.log_ssqrt)
        kl -= 0.5 * 4 * torch.sum(self.log_lambda_sqrt_inv_diag) * self.low_rank
        # s term, which is a constant
        kl -= 0.5 * self.low_rank * self.low_rank * self.num_heads

        return attn_out, [escore, rscore, self.we, self.wr], lambda_sqrt_inv_diag, kl, mean, covariance

class TransformerEncoder(nn.Module):
    def __init__(self, args, attn_type, feats, mlp_hidden=128, head=8, dropout=0., embed_len=64, \
                low_rank=10, rank_multi=10, attn_drop=0.):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.attn_type = attn_type
        self.la1 = nn.LayerNorm(feats)
        if self.attn_type == "softmax":
            self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        elif self.attn_type == "kep_svgp":
            self.msa = KEP_SVGPAttention(feats, head, embed_len=embed_len, low_rank=low_rank, rank_multi=rank_multi, \
                                            concate=args.concate, proj_drop=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if self.attn_type == "softmax":
            # noise_std = self.args.adversarial_noise # adjust as needed
            # Process original input
            la_x = self.la1(x)
            out_original = self.msa(la_x)
            mean_original = out_original
            # cov_original = torch.zeros_like(out_original)
            x_t_trans = out_original + x  # residual for original input
            out_final = self.mlp(self.la2(x_t_trans)) + x_t_trans

            # # Generate adversarial_samples adversarial inputs and compute aggregated mean/variance
            # means_list = [mean_original]
            # for _ in range(self.args.adversarial_samples):
            #     adv_x = x + torch.randn_like(x) * noise_std
            #     la_adv = self.la1(adv_x)
            #     out_adv = self.msa(la_adv)
            #     means_list.append(out_adv)
            # means_stack = torch.stack(means_list, dim=0)  # shape: (5, B, seq_len, d_model)
            # aggregated_mean = torch.mean(means_stack, dim=0) + x
            # aggregated_std = torch.std(means_stack, dim=0, unbiased=False) 

            # return out_final, x_t_trans, aggregated_mean, aggregated_std
            return out_final, x_t_trans, x_t_trans, torch.zeros_like(x_t_trans)

        else:
            la_x = self.la1(x)
            out, scores, Lambda_inv, kl, mean, cov = self.msa(la_x)
            out = out + x
            x_t_trans = out
            out = self.mlp(self.la2(out)) + out
            return out, scores, Lambda_inv, kl, x_t_trans, mean, cov


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats, head, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

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
        means = []
        covariances = []
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        x_t.append(out)
        for enc in self.enc:
            if enc.attn_type == "softmax":
                out, x_t_trans, mean, cov = enc(out)
                x_t.append(x_t_trans)
                means.append(mean)
                covariances.append(cov)
            elif enc.attn_type == "kep_svgp":
                out, scores, Lambda_inv, kl, x_t_trans, mean, cov = enc(out)
                score_list.append(scores)
                Lambda_inv_list.append(Lambda_inv)
                kl_list.append(kl)
                x_t.append(x_t_trans)
                means.append(mean)
                covariances.append(cov)
        
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)

        return out, x_t, means, covariances

def vit_cifar(args, attn_type, num_classes, ksvd_layers, low_rank, rank_multi):
    return ViT(args=args, attn_type=attn_type, ksvd_layers=ksvd_layers, num_classes=num_classes, low_rank=low_rank, rank_multi=rank_multi, \
                img_size=32, patch=8, dropout=0.1, num_layers=args.depth, hidden=args.hdim, head=args.num_heads, mlp_hidden=args.hdim, is_cls_token=False)