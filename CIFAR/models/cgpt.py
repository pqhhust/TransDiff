import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.distributions import multivariate_normal
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
        ###### TODO: CHECK LINEAR PROJ IN CGPT SYM CASE
        # import pdb; pdb.set_trace()
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


class ClassficationHead(torch.nn.Module):
    def __init__(self, hdim, num_class, drop_rate=0.):
        super(ClassficationHead, self).__init__()
        self.hdim = hdim
        self.num_class = num_class
        self.fc = nn.Sequential(nn.Linear(hdim, num_class), nn.Dropout(drop_rate))
        self.seqpool = nn.Linear(hdim, 1, bias=False)
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x, input_mask):
        input_mask = input_mask.unsqueeze(-1).unsqueeze(1)
        res = x* input_mask
        res = torch.mean(res, 2)
        res = self.ln(res)
        res = self.fc(res)
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
        res = self.seqpool(x).permute(0,1,3,2) 
        res = torch.softmax(res, -1) 
        res = res @ x 
        res = torch.mean(res, 2) 
        res = self.ln(res)
        res = self.fc(res) 
        return res


def kernel_ard(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1)
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    X1 = X1.permute(0,1,3,2).unsqueeze(4) 
    X2 = X2.unsqueeze(3) 
    return  torch.exp(log_sf).unsqueeze(1) * \
        torch.exp(-0.5* torch.sum((X1-X2.permute(0,1,4,3,2)).pow(2), 2)) 


def kernel_exp(X1, X2, log_ls, log_sf):
    X1 = X1 * torch.exp(-log_ls).unsqueeze(1) 
    X2 = X2 * torch.exp(-log_ls).unsqueeze(1)
    return torch.exp(log_sf).unsqueeze(1)* torch.exp(X1 @ X2.permute(0,1,3,2))

# standardized kernel
def kernel_std(X1, X2):
    X1 = X1.permute(0,1,3,2).unsqueeze(4) 
    X2 = X2.unsqueeze(3) 
    return torch.exp(-0.5* torch.sum((X1-X2.permute(0,1,4,3,2)).pow(2), 2))

def kernel_exp_std(X1, X2):
    return torch.exp(X1 @ X2.permute(0,1,3,2)) 


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)

class CGP_LAYER(nn.Module):
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate, flag_cgp):
        super(CGP_LAYER, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.hdim = hdim
        self.vdim = self.hdim // self.num_heads
        self.dq = self.vdim
        self.flag_cgp = flag_cgp
        self.keys_len = keys_len
        self.drop_rate = drop_rate
        
        if kernel_type == 'exponential':
            self.log_sf_exp = nn.Parameter(-4. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32)) # sf=scaling factor
            self.log_ls_exp = nn.Parameter(4. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) # ls=length scale
        elif kernel_type == 'ard' or kernel_type=='std':
            self.log_sf_ard = nn.Parameter(0. + 0.* torch.tensor(npr.randn(self.num_heads,1), dtype=torch.float32))   # sf= scaling factor
            self.log_ls_ard = nn.Parameter(0. + 1.* torch.tensor(npr.randn(self.num_heads,self.dq), dtype=torch.float32)) # ls=length scale
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        # self.fc_qk = nn.Linear(self.hdim, self.hdim, bias=False)
        if self.kernel_type == 'scale_dot':
            self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)
            self.fc_q = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_v = nn.Linear(self.hdim, self.hdim, bias=False) 
        
        self.v = nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.vdim), dtype=torch.float32))
        self.s_sqrt_ltri = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len, self.keys_len), dtype=torch.float32))
        self.log_s_sqrt_diag = nn.Parameter( torch.tensor(npr.randn(self.num_heads, 1, self.vdim, self.keys_len), dtype=torch.float32))

        # For CGP
        self.sigma_q = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.sigma_k = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.fc_q = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_k = nn.Linear(self.hdim, self.hdim, bias=False)
        self.fc_x0_2 = nn.Linear(self.hdim, self.hdim,bias=False)
        
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))
        self.scale = 1 / (hdim ** 0.5)
    
    
    def get_q_k_GP(self, x):
        if self.flag_cgp:
            q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
            k = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) # Asym
            x0 = self.fc_x0_2(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        else: # kernel attention case
            q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
            k = q.clone()
            x0 = None
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        return q, k, v, x0


    def get_q_k_SDP(self, x):
        q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        k = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3) 
        return q, k, v
        
    def forward(self, x):
        q, k, v, x0 = self.get_q_k_GP(x)
            
        if self.flag_cgp:
            jitter = self.jitter
            if self.kernel_type == 'std':
                # Asym
                K_kk = (self.sigma_k**2) * kernel_std(k, k)
                K_qq = (self.sigma_q**2) * kernel_std(q, q)

                K_0 = kernel_std(x0, x0)
                K_qk = kernel_std(q, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ kernel_std(x0, k)
                
                f_K = (K_kk + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ v
                
                while True:
                    try:
                        chol_K_0 = torch.linalg.cholesky(K_0 + jitter* torch.eye(K_0.shape[2]).to(self.device)) 
                        break
                    except Exception:
                        jitter = jitter * 10
                # import pdb; pdb.set_trace()
                z0_samples = torch.zeros_like(x0) + (chol_K_0 @ torch.randn_like(x0).to(self.device))   

                while True:
                    try:
                        chol_K_kk = torch.linalg.cholesky(K_kk + jitter* torch.eye(K_kk.shape[2]).to(self.device)) 
                        break
                    except Exception:
                        jitter = jitter * 10
                
                # Full GP mean and covar
                mean = K_qk @ v
                E_z0z0 = K_0.unsqueeze(2)
                v0 = torch.triangular_solve(kernel_std(k, x0).permute(0,1,3,2), chol_K_kk, upper=False).solution
                E_z0z0 = E_z0z0 - v0.unsqueeze(2).permute(0,1,2,4,3) @ v0.unsqueeze(2) 
                # import pdb; pdb.set_trace()
                E_z0z0 = E_z0z0 + (kernel_std(x0, k) @ torch.linalg.inv(K_kk + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ f_K @ f_K.permute(0,1,3,2) @ \
                    torch.linalg.inv(K_kk + jitter * torch.eye(K_kk.shape[2]).to(self.device)) @ kernel_std(k, x0)).unsqueeze(2)
                

                covar = K_qq.unsqueeze(2) 
                v1 = torch.triangular_solve(kernel_std(q, x0).permute(0,1,3,2), chol_K_0, upper=False).solution
                covar -= v1.unsqueeze(2).permute(0,1,2,4,3) @ v1.unsqueeze(2) 
                covar += (kernel_std(q, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ E_z0z0.squeeze() @ \
                    torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ kernel_std(x0, q)).unsqueeze(2)
                covar -= mean.unsqueeze(2) @ mean.unsqueeze(2).permute(0,1,2,4,3)

                # Cholesky of covar
                while True:
                    try:
                        chol_covar = torch.linalg.cholesky(covar + jitter * torch.eye(covar.shape[3]).to(self.device))  
                        break
                    except Exception:
                        jitter = jitter * 10
                chol_covar = chol_covar.unsqueeze(2) 
                samples = mean.permute(0,1,3,2).unsqueeze(2) + (chol_covar @ \
                torch.randn(mean.shape[0], mean.shape[1], self.sample_size, mean.shape[3], mean.shape[2], 1).to(self.device)).squeeze(-1)   
                
                # mean only, no covar
                #samples = mean.permute(0,1,3,2).unsqueeze(2)

                samples = samples.permute(0,1,2,4,3) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1)
                samples = self.W_O(samples)

                ############################### log joint q ###############################
                mean_P_zq_z0 = kernel_std(q, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ z0_samples
                covar_P_zq_z0 = K_qq.unsqueeze(2) 
                vq = torch.triangular_solve(kernel_std(q, x0).permute(0,1,3,2), chol_K_0, upper=False).solution
                covar_P_zq_z0 = covar_P_zq_z0 - vq.unsqueeze(2).permute(0,1,2,4,3) @ vq.unsqueeze(2)
                
                while True:
                    try:
                        chol_covar_P_zq_z0 = torch.linalg.cholesky(covar_P_zq_z0 + jitter * torch.eye(covar_P_zq_z0.shape[3]).to(self.device))  
                        break
                    except Exception:
                        jitter = jitter * 10
                chol_covar_P_zq_z0 = chol_covar_P_zq_z0.unsqueeze(2)

                Lq = torch.triangular_solve((mean-mean_P_zq_z0), chol_covar_P_zq_z0.squeeze(), upper=False).solution 
                q_term = Lq.permute(0,1,3,2) @ Lq 

                log_joint_q = torch.mean(torch.sum(q_term, (-1,-2,-3))) + 2 * torch.abs(torch.mean(torch.sum(torch.log(torch.diagonal(chol_covar_P_zq_z0, dim1=-2, dim2=-1)), dim=-1)))

                ############################### log joint k ###############################
                mean_P_zk_z0 = kernel_std(k, x0) @ torch.linalg.inv(K_0 + jitter* torch.eye(K_kk.shape[2]).to(self.device)) @ z0_samples
                covar_P_zk_z0 = K_kk.unsqueeze(2) 
                vk = torch.triangular_solve(kernel_std(k, x0).permute(0,1,3,2), chol_K_0, upper=False).solution
                covar_P_zk_z0 = covar_P_zk_z0 - vk.unsqueeze(2).permute(0,1,2,4,3) @ vk.unsqueeze(2)
                
                while True:
                    try:
                        chol_covar_P_zk_z0 = torch.linalg.cholesky(covar_P_zk_z0 + jitter * torch.eye(covar_P_zk_z0.shape[3]).to(self.device))  
                        break
                    except Exception:
                        jitter = jitter * 10
                chol_covar_P_zk_z0 = chol_covar_P_zk_z0.unsqueeze(2)

                Lk = torch.triangular_solve((f_K-mean_P_zk_z0), chol_covar_P_zk_z0.squeeze(), upper=False).solution 
                k_term = Lk.permute(0,1,3,2) @ Lk 

                log_joint_k = torch.mean(torch.sum(k_term, (-1,-2,-3))) + 2 * torch.abs(torch.mean(torch.sum(torch.log(torch.diagonal(chol_covar_P_zk_z0, dim1=-2, dim2=-1)), dim=-1)))

                log_joint_qk = log_joint_q + log_joint_k
                return samples, log_joint_qk

            else:
                raise ValueError("kernel_type must be std")
        
        else:
            if self.kernel_type == 'ard': # Asym kernel
                q, k, v = self.get_q_k_SDP(x)
                K_qk = kernel_ard(q, k, self.log_ls_ard, self.log_sf_ard) 
                mean = K_qk @ v
                samples = mean.unsqueeze(2) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
                samples = self.W_O(samples)
                return samples, None
            
            elif self.kernel_type == 'std': # Asym kernel
                q, k, v = self.get_q_k_SDP(x)
                K_qk = kernel_std(q, k) 
                mean = K_qk @ v
                samples = mean.unsqueeze(2) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
                samples = self.W_O(samples)
                return samples, None
            
            elif self.kernel_type == "scale_dot":
                q, k, v = self.get_q_k_SDP(x)
                attn_score = (self.scale) * (torch.einsum('abid,abdj->abij', (q, k.permute(0,1,3,2))))
                attn_prob = F.softmax(attn_score, dim=1)
                out = attn_prob @ v
                samples = out.unsqueeze(2) 
                samples = torch.flatten(samples.permute(0,2,3,1,4),-2,-1) 
                samples = self.W_O(samples)
                return samples, None
            
class ViT(torch.nn.Module):
    def __init__(self, device, depth, patch_size, in_channels, max_len, num_class, hdim, num_heads, sample_size, jitter, drop_rate, keys_len, kernel_type, flag_cgp):
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
        self.keys_len = keys_len
        self.kernel_type = kernel_type
        self.drop_rate = drop_rate

        # alpha coeff
        # self.alpha_anneal = alpha_anneal

        self.patch_embedding = Patch_embedding(patch_size=patch_size, in_channels=in_channels, hdim=hdim, max_len=max_len, drop_rate=drop_rate)
        
        self.class_head = ClassficationHead_vit(hdim=hdim, num_class=num_class)

        self.device = device

        self.ln = nn.LayerNorm(hdim)

        self.keys = nn.ParameterList([nn.Parameter(torch.tensor(npr.randn(self.num_heads, 1, self.keys_len, self.hdim), dtype=torch.float32)) for i in range(self.depth)])

        self.cgp_layer_list = nn.ModuleList([CGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate, keys_len=self.keys_len, sample_size=self.sample_size, jitter=jitter, flag_cgp=flag_cgp)])
        self.mlp_layer_list = nn.ModuleList([FC(hdim=hdim, drop_rate=self.drop_rate)])

        for i in range(1, depth):
            self.cgp_layer_list.append(CGP_LAYER(device=device, num_heads=num_heads, max_len=max_len, hdim=hdim, kernel_type=self.kernel_type, drop_rate=self.drop_rate, keys_len=self.keys_len, sample_size=1, jitter=jitter, flag_cgp=flag_cgp))
            self.mlp_layer_list.append(FC(hdim=hdim, drop_rate=self.drop_rate))

    def forward(self, X):
        patch_emb_ln, patch_emb = self.patch_embedding.forward(X) 
        z, total_kl = self.cgp_layer_list[0].forward(patch_emb_ln) 
        z_prime = patch_emb.unsqueeze(1) + z 
        z_ln = self.ln(z_prime)
        
        z = self.mlp_layer_list[0].forward(z_ln) + z_prime 
        cur_k = self.mlp_layer_list[0].forward(self.keys[1]) + self.keys[1] 
        for i in range(1, self.depth):
            z_prev = z.reshape(-1, z.shape[-2], z.shape[-1]) 
            z_ln = self.ln(z_prev)  
            cur_k = self.ln(cur_k) 
            z, kl = self.cgp_layer_list[i].forward(z_ln) 
            if total_kl:
                total_kl += kl
            z_prime = z_prev.unsqueeze(1) + z  
            z_ln = self.ln(z_prime)  
            z = self.mlp_layer_list[i].forward(z_ln) + z_prime  
            
            if i < self.depth-1:
                cur_k = self.mlp_layer_list[i].forward(self.keys[i+1]) + self.keys[i+1] 
        logits = self.class_head.forward(z).squeeze(1) 
        return logits, total_kl 
    
    def loss(self, X, y, anneal_kl):
        logits, total_kl = self.forward(X)
        ce_loss = nn.CrossEntropyLoss()
        y = torch.unsqueeze(y,1)
        y = torch.tile(torch.unsqueeze(y, 1), (1, self.sample_size, 1)).view(-1, y.shape[1])
        neg_ElogPyGf = ce_loss(logits.view(-1, self.num_class), y.view(-1))
        if total_kl:
            loss = neg_ElogPyGf + anneal_kl*total_kl
        else:
            loss = neg_ElogPyGf
        return loss