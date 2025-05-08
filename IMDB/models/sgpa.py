import torch
import numpy as np
import numpy.random as npr
import torch.nn as nn   
# from allennlp.modules.elmo import batch_to_ids, Elmo
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

class SGP_LAYER(nn.Module):
    def __init__(self, device, num_heads, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate, flag_sgp, inference_mode):
        super(SGP_LAYER, self).__init__()
        self.num_heads = num_heads
        self.hdim = hdim
        self.vdim = self.hdim // self.num_heads
        self.dq = self.vdim
        self.flag_sgp = flag_sgp
        self.keys_len = keys_len
        self.drop_rate = drop_rate
        self.K_k_beta_k_beta = None
        self.inference_mode = inference_mode
        self.cache_inverse1 = None
        self.cache_inverse2 = None
        
        if kernel_type == 'exponential':
            self.log_sf = nn.Parameter(-4. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32)) 
            self.log_ls = nn.Parameter(4. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        elif kernel_type == 'ard':
            self.log_sf = nn.Parameter(0. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32))
            self.log_ls = nn.Parameter(0. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) 
        else:
            raise ValueError("The argument 'kernel_type' should be either 'exponential' or 'ard'.")
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        self.fc_qkv = nn.Linear(self.hdim, 2* self.num_heads* self.vdim, bias=False)
        
        if self.flag_sgp:
            self.v = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.vdim), dtype=torch.float32))
            self.s_sqrt_ltri = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len, self.keys_len), dtype=torch.float32))
            self.log_s_sqrt_diag = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len), dtype=torch.float32))
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
      
    def get_q_k_v_ssqrt(self, x, cur_k):
        
        q, v_gamma = self.fc_qkv(x).view(x.shape[0], x.shape[1], self.num_heads, 2* self.vdim).permute(0,2,1,3).chunk(chunks=2, dim=-1)
        k_gamma = q
        if self.flag_sgp:
            W_qk = self.fc_qkv.weight[:self.hdim]
            k_beta = W_qk.view(self.num_heads, 1, 1, self.vdim, self.hdim) @ cur_k.unsqueeze(-1) 
            k_beta = k_beta.squeeze(-1).permute(1,0,2,3) 
            v_beta = self.v.permute(1,0,2,3)
            log_ssqrt = self.log_s_sqrt_diag.permute(1,0,2,3) 
            return q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt  
        else:
            return q, k_gamma, v_gamma
        
    def forward(self, x, cur_k):
        # We set W_q = W_k to maintain a valid symmetric deep kernel, so q = k_gamma below when kernel_type='exponential' or 'ard'.
        # We can use different projection matrices if necessary.
        if self.flag_sgp:
            q, k_gamma, k_beta, v_gamma, v_beta, log_ssqrt = self.get_q_k_v_ssqrt(x, cur_k)
        else:
            q, k_gamma, v_gamma = self.get_q_k_v_ssqrt(x, cur_k)
            
        if self.kernel_type == 'exponential':
            if not self.flag_sgp:
                K_qq = kernel_exp(q, q, self.log_ls, self.log_sf)  # [bs, num_heads, max_len, max_len]
            else:
                K_qq, K_qk_beta = kernel_exp(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                    self.log_ls, self.log_sf).tensor_split([x.shape[1],],-1) # [bs, num_heads, max_len, max_len + keys_len]
                K_k_beta_k_gamma = K_qk_beta.permute(0,1,3,2)

                if self.K_k_beta_k_beta != None:
                    K_k_beta_k_beta = self.K_k_beta_k_beta
                else:
                    K_k_beta_k_beta = kernel_exp(k_beta, k_beta, self.log_ls, self.log_sf)
                    if self.inference_mode:
                        self.K_k_beta_k_beta = K_k_beta_k_beta
            K_qk_gamma = K_qq
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq
        elif self.kernel_type == 'ard':
            if not self.flag_sgp:
                K_qq = kernel_ard(q, q, self.log_ls, self.log_sf)  
            else:
                K_qq, K_qk_beta = kernel_ard(q, torch.cat([q, k_beta.tile(q.shape[0],1,1,1)], 2), \
                    self.log_ls, self.log_sf).tensor_split([x.shape[1],],-1) 
                K_k_beta_k_gamma = K_qk_beta.permute(0,1,3,2)

                if self.K_k_beta_k_beta != None:
                    K_k_beta_k_beta = self.K_k_beta_k_beta
                else:
                    K_k_beta_k_beta = kernel_ard(k_beta, k_beta, self.log_ls, self.log_sf)
                    if self.inference_mode:
                        self.K_k_beta_k_beta = K_k_beta_k_beta
            K_qk_gamma = K_qq
            if self.flag_sgp:    
                K_k_gamma_k_gamma = K_qq
        else:
            raise ValueError("The argument 'kernel_type' should be either 'exponential' or 'ard'.")
        
        # mask1 = mask.unsqueeze(-1).view(mask.shape[0],-1, mask.shape[1]).unsqueeze(1)
        # mask2=mask.unsqueeze(1).view(mask.shape[0],mask.shape[1],-1).unsqueeze(1)
        # v_gamma = v_gamma * mask2
        if not self.flag_sgp: 
            mean = K_qk_gamma @ v_gamma
            samples = mean.unsqueeze(2) 
            samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
            samples = self.W_O(samples) 
            return samples, None
        else:
            s_sqrt = torch.exp(log_ssqrt) 
            s_sqrt_diag = torch.diag_embed(s_sqrt) 
            s_sqrt_local = s_sqrt_diag + torch.tril(self.s_sqrt_ltri.permute(1,0,2,3,4), diagonal=-1) 

            if self.inference_mode and self.cache_inverse1 == None:
                K_kk_inverse = torch.linalg.inv(K_k_beta_k_beta + self.jitter* torch.eye(K_k_beta_k_beta.shape[2], device=self.device))
                self.cache_inverse1 = K_kk_inverse
                K_kk_inverse = K_kk_inverse.unsqueeze(2)
                chol_K_kk = torch.linalg.cholesky(K_k_beta_k_beta + self.jitter* torch.eye(K_k_beta_k_beta.shape[2], device=self.device)).unsqueeze(2)
                self.cache_inverse2 = K_kk_inverse @ chol_K_kk @ s_sqrt_local @ s_sqrt_local.permute(0,1,2,4,3) @ chol_K_kk.permute(0,1,2,4,3) @ K_kk_inverse - K_kk_inverse
           
            # Notice here we make diagonal approximation of the full covariance to accelerate sampling. 
            # Empirically, it doesn't seem to hurt the performance.
            chol_covar1 = torch.diagonal(K_qq.unsqueeze(2) , dim1=3, dim2=4).permute(0,1,3,2).unsqueeze(2)
            if self.inference_mode:
                # During inference, using cached inverse instead of solving linear systems to speed up.
                mean1 = K_qk_gamma @ v_gamma
                mean = mean1 - K_qk_beta @ (self.cache_inverse1 @ (K_k_beta_k_gamma @ v_gamma)) + K_qk_beta @ v_beta
                chol_covar = (chol_covar1 + ((K_qk_beta.unsqueeze(2) @ self.cache_inverse2) * K_qk_beta.unsqueeze(2)).sum(-1).permute(0,1,3,2).unsqueeze(2)).pow(0.5)
            else:
                jitter = self.jitter
                while True:
                    try:
                        chol_K_kk = torch.linalg.cholesky(K_k_beta_k_beta + jitter* torch.eye(K_k_beta_k_beta.shape[2], device=self.device))
                        break
                    except Exception:
                        jitter = jitter * 10

                v1 = torch.linalg.solve_triangular( chol_K_kk, K_k_beta_k_gamma, upper=False)
                v2 = torch.linalg.solve_triangular(chol_K_kk, K_k_beta_k_gamma @ v_gamma, upper=False)
                v3 = v1.unsqueeze(2).permute(0,1,2,4,3) @ s_sqrt_local
                mean1 = K_qk_gamma @ v_gamma
                mean = mean1 - v1.permute(0,1,3,2) @ v2 + K_qk_beta @ v_beta

                chol_covar2 = v3.pow(2).sum(-1).permute(0,1,3,2).unsqueeze(2) - \
                    v1.unsqueeze(2).permute(0,1,2,4,3).pow(2).sum(-1).permute(0,1,3,2).unsqueeze(2)
                chol_covar = (chol_covar1 + chol_covar2).pow(0.5)
                
                
            samples = mean.unsqueeze(2) + chol_covar * torch.randn((mean.shape[0], mean.shape[1], self.sample_size, mean.shape[2], mean.shape[3]), device=self.device)   
            samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
            samples = self.W_O(samples) 

            if self.inference_mode:
                return samples, None
            else:
                kl = -0.5* self.keys_len* self.vdim * self.num_heads 
                kl += 0.5* torch.mean(torch.sum(s_sqrt_local.pow(2), (-1,-2,-3,-4)))            
                kl += 0.5* torch.mean(torch.sum(v_beta.permute(0,1,3,2).unsqueeze(3) @ K_k_beta_k_beta.unsqueeze(2) @ v_beta.permute(0,1,3,2).unsqueeze(4), (1,2))) 
                second_term = v2.permute(0,1,3,2).unsqueeze(3) @ v2.permute(0,1,3,2).unsqueeze(4)
                temp = v_gamma.permute(0,1,3,2).unsqueeze(3) @ mean1.permute(0,1,3,2).unsqueeze(4) - second_term
                kl += 0.5* torch.mean(torch.sum(temp, (1,2)))
                kl -= torch.mean(torch.sum(log_ssqrt, (-1, -2, -3))) 
                return samples, kl
                
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

class Embeddings(torch.nn.Module):
    def __init__(self,vocab_size,max_len,emb_size,h_size, drop_rate):
        super(Embeddings,self).__init__()
        
        self.token_embeds=nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.pos_embeds=PositionalEncoding(emb_size, drop_rate, max_len)
        self.layer_norm=nn.LayerNorm(h_size)
            
        self.project=nn.Linear(emb_size,h_size)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self,input_data):
        rep=self.token_embeds(input_data)
        output=self.pos_embeds(rep)
      
        output=self.project(output)
        output = self.dropout(output)
        
        return self.layer_norm(output), output
class Transformer(torch.nn.Module):
    def __init__(self, device, vocab_size, depth, max_len, num_class, embdim, hdim, num_heads, sample_size, jitter, drop_rate, keys_len, kernel_type, flag_sgp, inference_mode=False):
        super(Transformer, self).__init__()
        self.hdim = hdim
        self.max_len = max_len
        self.num_class = num_class
        self.sample_size=sample_size
        self.depth = depth
        self.jitter = jitter
        self.keys_len = keys_len
        self.kernel_type = kernel_type
        self.drop_rate = drop_rate
        self.embdim = embdim
        self.vocab_size = vocab_size
        self.flag_sgp=flag_sgp

        self.embedding = Embeddings(vocab_size=vocab_size,max_len=max_len,emb_size=embdim,h_size=hdim,drop_rate=drop_rate)
        
        self.class_head = ClassficationHead(hdim=hdim, num_class=num_class, drop_rate=drop_rate)

        self.device = device

        self.ln = nn.LayerNorm(hdim)

        self.keys = nn.ParameterList([nn.Parameter(torch.tensor(npr.randn(num_heads, 1, self.keys_len, self.hdim), dtype=torch.float32)) for i in range(self.depth)])

        self.sgp_layer_list = nn.ModuleList([SGP_LAYER(device=device, num_heads=num_heads, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate,\
                 keys_len=self.keys_len, sample_size=self.sample_size, jitter=jitter, flag_sgp=self.flag_sgp, inference_mode=inference_mode)])
        self.mlp_layer_list = nn.ModuleList([FC(hdim=hdim, drop_rate=self.drop_rate)])

        for i in range(1, depth):
            self.sgp_layer_list.append(SGP_LAYER(device=device, num_heads=num_heads, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate,\
                 keys_len=self.keys_len, sample_size=1, jitter=jitter, flag_sgp=self.flag_sgp, inference_mode=inference_mode))
            self.mlp_layer_list.append(FC(hdim=hdim, drop_rate=self.drop_rate))

    def forward(self, input_data):
        emb_ln, emb = self.embedding.forward(input_data)         
        z, total_kl = self.sgp_layer_list[0].forward(emb_ln, self.keys[0]) 
        z_prime = emb.unsqueeze(1) + z
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
            z, kl = self.sgp_layer_list[i].forward(z_ln, cur_k) 
            if total_kl:
                total_kl += kl
            z_prime = z_prev.unsqueeze(1) + z
            z_ln = self.ln(z_prime)  
            z = self.mlp_layer_list[i].forward(z_ln) + z_prime
            if self.flag_sgp and i < self.depth-1:
                cur_k = self.mlp_layer_list[i].forward(self.keys[i+1]) + self.keys[i+1] 
        logits = self.class_head.forward(z).squeeze(1)
        return logits, total_kl 
    
    def loss(self, input_data,answers, anneal_kl=1.):
        logits, total_kl = self.forward(input_data) 
        ce_loss = nn.CrossEntropyLoss()
        answers = torch.unsqueeze(answers,1) 
        answers = torch.tile(torch.unsqueeze(answers, 1), (1, self.sample_size, 1)).view(-1, answers.shape[1]) 
        neg_ElogPyGf = ce_loss(logits.view(-1, self.num_class), answers.view(-1))
        if total_kl and total_kl.item() > 0:
            loss = neg_ElogPyGf + anneal_kl* total_kl
        else:
            loss = neg_ElogPyGf
        return loss