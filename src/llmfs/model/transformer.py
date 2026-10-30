import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import multihead_self_attention_rope


class Transformer_block(nn.Module):
    # FLOPs 主项：8*N*D^2 + 4*B*T^2*D + 6*N*D*F。
    # 还包含：2 次 RMSNorm、Q/K RoPE、SDPA 缩放/softmax、SiLU/门控，
    # 以及两次残差相加 2*N*D；这些子模块开销见各自注释。
    # 示例：矩阵主项 4,079,616；两个残差相加共 6,144。
    def __init__(self, d_model:int,num_heads:int,d_ff:int,eps_RMS1:float,eps_RMS2:float,theta=None,max_seq_len=None):
        super().__init__()
        self.RMS1 = RMSnorm(d_model=d_model,eps=eps_RMS1)
        self.MSAR = multihead_self_attention_rope(d_model=d_model,num_head=num_heads,theta=theta,max_seq_len=max_seq_len)

        self.RMS2 = RMSnorm(d_model=d_model,eps=eps_RMS2)
        self.PWFF = PWFF(d_model=d_model,d_ff=d_ff)
    def forward(self,x:torch.Tensor):
        seq_len = x.shape[-2]
        token_position = torch.arange(seq_len, device=x.device)
        temp1 = self.RMS1(x)
        temp1 = self.MSAR(temp1,token_position)
        x = x + temp1
        
        temp2 = self.RMS2(x)
        temp2 = self.PWFF(temp2)
        x = x + temp2
        return x

    
class RMSnorm(nn.Module):
    # FLOPs：平方 N*D；求和 N*(D-1)；求均值 N；加 eps N；
    # 除以 RMS N*D；乘权重 N*D => 4*N*D+N 普通运算，另有 N 次 sqrt。
    # 示例：12,336 普通运算 + 48 次 sqrt；类型转换的内存开销另计。
    def __init__(self,  d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True) + self.eps) 
        result = x / rms * self.weight
        return result.to(in_dtype)

    
class PWFF(nn.Module):
    # FLOPs：三个矩阵乘法各 2*N*D*F，共 6*N*D*F；门控 h*b 另加 N*F。
    # SiLU 处理 N*F 个元素，另有 O(N*F) 运算（含 exp），未包含在矩阵主项内。
    # 示例：矩阵主项 2,359,296；门控 6,144；SiLU 处理 6,144 个元素。
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        var = 2 / (d_model + d_ff)
        std = var ** 0.5
        # temp = d_model * 8 / 3
        # self.d_ff = (temp / 64) * 64 if temp % 64 <= 32 else ((temp / 64) + 1) * 64
        self.W1 = nn.Parameter(torch.empty(self.d_ff,self.d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.W1,0,std,-3*std,3*std)
        self.W2 = nn.Parameter(torch.empty(self.d_model,self.d_ff,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.W2,0,std,-3*std,3*std)
        self.W3 = nn.Parameter(torch.empty(self.d_ff,self.d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.W3,0,std,-3*std,3*std)
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        a = x @ self.W1.T
        b = x @ self.W3.T
        h = F.silu(a)
        result = (h * b) @ self.W2.T
        return result
    


