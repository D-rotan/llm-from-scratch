import torch
import torch.nn as nn
class EmbeddingModule(nn.Module):
    # FLOPs：查表约 0 浮点算术；读取/输出 N*embedding_dim 个元素，内存开销非零。
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.empty(self.num_embeddings,
                                          self.embedding_dim,
                                          device=device,
                                          dtype=dtype))
        torch.nn.init.trunc_normal_(self.W,0,1,-3,3)
    def forward(self, token_ids:torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]