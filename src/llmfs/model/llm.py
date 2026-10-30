import torch
from .embedding import EmbeddingModule
from .transformer import Transformer_block,RMSnorm
from torch import nn
# FLOPs 口径：只统计一次 forward，不含 backward、初始化及 RoPE 缓存预计算。
# B=所有 batch 维度的乘积，T=实际序列长度，N=B*T，D=d_model，F=d_ff，
# H=num_heads，K=D/H（要求整除），L=num_layers，V=vocab_size。
# 矩阵乘法按 2*m*n*k 估算（一次乘加=2 FLOPs，忽略归约末项差异）。
# 普通逐元素加/减/乘/除各计 1；exp/sqrt/silu 单列，不等同于硬件指令数。
# 索引、比较、赋值、类型转换、重排不计浮点算术，但仍有时间/内存开销。
# 汇总中的“主项”仅指矩阵乘法；各父模块已包含子模块，不能重复累加。
# 测试示例：B=4,T=12,D=64,F=128,H=4,L=3,V=10000；不是 T=context_length=16。
class Transformer_lm(nn.Module):
    # FLOPs 主项：L*(8*N*D^2 + 4*B*T^2*D + 6*N*D*F) + 2*N*D*V。
    # 另加 L 个 block 的非矩阵运算及最终 RMSNorm；embedding 是查表，输出是 logits。
    # 示例：3*4,079,616 + 61,440,000 = 73,678,848，约 73.68 MFLOPs。
    # [计算量集中] 此示例输出头占矩阵主项约 83.4%，属于大词表投影的正常开销。
    def __init__(self, vocab_size:int,context_length:int,num_layers:int,d_model:int,
                 d_ff:int,num_heads:int,rope_theta:float,eps_RMS1:float,eps_RMS2:float,eps_RMS3:float):
        super().__init__()
        self.embedding = EmbeddingModule(vocab_size,d_model)
        self.num_layers = num_layers
        self.linear = LinearModule(d_model,vocab_size)
        self.RMS3 = RMSnorm(d_model=d_model,eps=eps_RMS3)
        self.transformer = nn.ModuleList([Transformer_block(d_model=d_model,d_ff=d_ff,num_heads=num_heads,
                                        theta=rope_theta,max_seq_len=context_length,eps_RMS2=eps_RMS2,eps_RMS1=eps_RMS1) for _ in range(num_layers)])
    def forward(self,token_ids:torch.Tensor):
        x = self.embedding(token_ids)
        for layer in self.transformer:
            x = layer(x)
        x = self.RMS3(x)
        x = self.linear(x)
        return x

    
class LinearModule(nn.Module):
    # FLOPs：输入展平为 (N, in_features)，为 2*N*in_features*out_features。
    # LM 输出头示例：2*4*12*64*10000 = 61,440,000 FLOPs。
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        var = 2/(in_features + out_features)
        std = var ** 0.5
        self.W = nn.Parameter(torch.empty(self.out_features,
                                          self.in_features,
                                          device=device,
                                          dtype=dtype))
        torch.nn.init.trunc_normal_(self.W,0,std,-3 * std,3 * std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
