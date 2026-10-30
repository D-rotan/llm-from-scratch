import torch
import torch.nn as nn
import einops

class multihead_self_attention_rope(nn.Module):
    # 加了位置参数的自我注意力
    # FLOPs：主项与普通多头注意力相同，为 8*N*D^2+4*B*T^2*D；
    # 再加 Q/K 的 RoPE 共 6*N*D，以及 SDPA 的缩放/softmax。
    # 示例：矩阵主项 1,720,320；RoPE 18,432。
    # [约束] D 必须能被 H 整除，且 D/H 为偶数。
    def __init__(self, d_model,num_head,theta,max_seq_len,device=None,dtype=None,):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        d_k = d_model // num_head
        d_v = d_model // num_head
        var = 2 / (d_model + d_model)
        std = var ** 0.5
        self.rope = Rope(theta=theta,max_seq_len=max_seq_len,d_k=d_k,device=device)
        self.QW = nn.Parameter(torch.empty(d_k * num_head,d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.QW,0,std,-3*std,3*std)
        self.KW = nn.Parameter(torch.empty(d_k * num_head,d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.KW,0,std,-3*std,3*std)
        self.VW = nn.Parameter(torch.empty(d_v * num_head,d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.VW,0,std,-3*std,3*std)
        self.OW = nn.Parameter(torch.empty(d_model,d_v * num_head,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.OW,0,std,-3*std,3*std)
    def forward(self,x:torch.Tensor,token_position):
        Q = x @ self.QW.T
        K = x @ self.KW.T
        V = x @ self.VW.T
        Q = einops.rearrange(Q,"... seq_len (num_head d_k) -> ... num_head seq_len d_k",num_head = self.num_head,d_k = self.d_model // self.num_head)
        K = einops.rearrange(K,"... seq_len (num_head d_k) -> ... num_head seq_len d_k",num_head = self.num_head,d_k = self.d_model // self.num_head)
        Q = self.rope(Q,token_position)
        K = self.rope(K,token_position)
        V = einops.rearrange(V,"... seq_len (num_head d_v) -> ... num_head seq_len d_v",num_head = self.num_head,d_v = self.d_model // self.num_head)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device))
        temp = SDPA(Q,K,V,mask)
        temp = einops.rearrange(temp,"... num_head seq_len d_v -> ... seq_len (num_head d_v)")
        result = temp @ self.OW.T
        return result


class multihead_self_attention(nn.Module):
    # 没有加位置参数的自我注意力
    # FLOPs：Q/K/V 投影 6*N*D^2，输出投影 2*N*D^2，SDPA 主项 4*B*T^2*D。
    # 合计主项 8*N*D^2+4*B*T^2*D，另加 SDPA 的缩放和 softmax。
    # 示例：主项 1,720,320；其中 SDPA 两次矩阵乘法共 147,456。
    # [约束] 当前投影和重排要求 D 能被 H 整除。
    def __init__(self, d_model,num_head,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        d_k = d_model // num_head
        d_v = d_model // num_head
        var = 2 / (d_model + d_model)
        std = var ** 0.5
        self.QW = nn.Parameter(torch.empty(d_k * num_head,d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.QW,0,std,-3*std,3*std)
        self.KW = nn.Parameter(torch.empty(d_k * num_head,d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.KW,0,std,-3*std,3*std)
        self.VW = nn.Parameter(torch.empty(d_v * num_head,d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.VW,0,std,-3*std,3*std)
        self.OW = nn.Parameter(torch.empty(d_model,d_v * num_head,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.OW,0,std,-3*std,3*std)
    def forward(self,x:torch.Tensor):
        Q = x @ self.QW.T
        K = x @ self.KW.T
        V = x @ self.VW.T
        Q = einops.rearrange(Q,"... seq_len (num_head d_k) -> ... num_head seq_len d_k",num_head = self.num_head,d_k = self.d_model // self.num_head)
        K = einops.rearrange(K,"... seq_len (num_head d_k) -> ... num_head seq_len d_k",num_head = self.num_head,d_k = self.d_model // self.num_head)
        V = einops.rearrange(V,"... seq_len (num_head d_v) -> ... num_head seq_len d_v",num_head = self.num_head,d_v = self.d_model // self.num_head)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device))
        temp = SDPA(Q,K,V,mask)
        temp = einops.rearrange(temp,"... num_head seq_len d_v -> ... seq_len (num_head d_v)")
        result = temp @ self.OW.T
        return result

    
class Rope(nn.Module):
    # FLOPs：每对通道 4 次乘法 + 2 次加/减，因此每次调用为 3*x.numel()。
    # 注意力中单独旋转 Q 或 K：3*B*H*T*K=3*N*D；两者合计 6*N*D。
    # 示例：单次 9,216，两次 18,432。缓存索引与写回不计浮点算术。
    # 缓存构造为 O(max_seq_len*K)，含 sin/cos；只在初始化计算，未计入 forward。
    # [约束] 当前偶/奇通道配对要求 d_k 为偶数，位置索引必须落在缓存范围内。
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        k = torch.arange(0,self.d_k,2,device=device,dtype=torch.float32)
        self.max_seq_len = max_seq_len
        theta_k = self.theta ** (-k / self.d_k)
        position = torch.arange(max_seq_len,device=device,dtype=torch.float32)
        angle = position[...,None] * theta_k[None,:]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1 = x[...,0::2]
        x2 = x[...,1::2]
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        new_x1 = x1 * cos - x2 * sin
        new_x2 = x1 * sin + x2 * cos
        result = torch.empty_like(x)
        result[...,0::2] = new_x1
        result[...,1::2] = new_x2
        return result
def softmax(x:torch.Tensor, i:int) -> torch.Tensor:
    # FLOPs：设 E=x.numel()、S=x.shape[i]、R=E/S（归一化行数）。
    # 减最大值 E + 求和(E-R) + 除法 E = 3*E-R 普通运算；另有 E 次 exp。
    # 求最大值约 E-R 次比较，不计浮点算术；因此不能把 softmax 当成 0 FLOPs。
    x_max = torch.max(x,dim = i,keepdim=True).values
    x = x - x_max
    exp_x = torch.exp(x)
    total = torch.sum(exp_x,dim=i,keepdim=True)
    return exp_x / total
def SDPA(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, Mask=None) -> torch.Tensor:
    # FLOPs：Q=(B,H,Tq,K)，K=(B,H,Tk,K)，V=(B,H,Tk,Kv) 时，
    # Q@K^T + P@V 主项 = 2*B*H*Tq*Tk*(K+Kv)。无 head 维时取 H=1。
    # 自注意力 Tq=Tk=T、K=Kv=D/H 时，主项为 4*B*T^2*D。
    # E=B*H*Tq*Tk，R=B*H*Tq：缩放 E + softmax(3*E-R)，另有 E 次 exp。
    # sqrt(K) 是标量开销；mask/转置另有内存开销。
    # [性能] 当前先做完整矩阵乘法再 mask，因果掩码不会把主项 FLOPs 减半；
    # scores/概率占 O(B*H*T^2) 内存，长序列时容易成为瓶颈。
    d_k = K.shape[-1]
    Kt = einops.rearrange(K,"batch_size ... seq_len d_k -> batch_size ... d_k seq_len")
    temp = (Q @ Kt) / (d_k ** 0.5)
    if Mask is not None:
        temp = temp.masked_fill(~Mask,float('-inf'))
    result = softmax(temp,-1) @ V
    return result
