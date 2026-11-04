import numpy as np
import torch
import math
import random
from ..model.llm import Transformer_lm
from .optimizer import AdamW
from .checkpoint import save_checkpoint,load_checkpoint
from ..configs.schema import ModelTrainerConfig
from pathlib import Path

def Training_loop(config: ModelTrainerConfig):
    data = np.load(config.src,mmap_mode="r")
    loss_sum = 0
    model = Transformer_lm(config.vocab_size,config.context_length,config.num_layers,
                           config.d_model,config.d_ff,config.num_heads,config.rope_theta,
                           eps_RMS1=config.eps_RMS1,eps_RMS2=config.eps_RMS2,
                           eps_RMS3=config.eps_RMS3)
    model.to(config.device)

    loss_fn = cross_entropy

    optimizer = AdamW(model.parameters(),betas=config.betas,weight_decay=config.weight_decay,
                     eps=config.eps_AdamW,lr=config.lr)

    model.train()
    checkpoint_dir = Path("data/checkpoints") / config.src.stem
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for iter in range(config.max_iters):
        X,Y = data_loading(data,batch_size=config.batch_size,
                           context_length=config.context_length,device=config.device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits,Y)
        loss.backward()
        if(config.on_gradient_clipping):
            gradient_clipping(model.parameters(),max_l2_norm=config.max_l2_norm,eps=config.eps_gradient,device=config.device)
        if(config.on_learning_rate_schedule):
            learning_rate = learning_rate_schedule(iter,lr_min=config.lr_min,lr_max=config.lr_max,
                                                   T_w=config.warmup_iters,T_c=config.cosine_cycle_iters)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        optimizer.step()
        loss_sum += loss.detach()
        if (iter+1) % config.checkpoint_iters == 0:
            avg_loss = loss_sum.item() / config.checkpoint_iters
            print(f"iter={iter+1},loss={avg_loss:.4f}")
            loss_sum = 0.0
            save_checkpoint(model=model,optimizer=optimizer,iteration=iter,
                            out=checkpoint_dir / f"checkpoint{iter+1}")
    save_path = config.out.parent / f"{config.out.stem}_{config.src.stem}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model.state_dict(),f=save_path)
    


def cross_entropy(logits:torch.Tensor,targets):
    max_logits = torch.max(logits,dim=-1,keepdim=True).values
    logits = logits - max_logits
    targets_logits = torch.gather(logits,dim=-1,index=targets.unsqueeze(-1))
    exp_logits = torch.exp(logits)
    temp = torch.sum(exp_logits,dim=-1,keepdim=True)
    result = torch.log(temp) - targets_logits
    return result.mean()


def learning_rate_schedule(t,lr_max,lr_min,T_w,T_c):
    if t < T_w:
        lr = t / T_w * lr_max
    elif T_w <= t and t <= T_c:
        lr = lr_min + 1/2 * (1 + math.cos((t-T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)
    elif T_c < t:
        lr = lr_min
    return lr


def gradient_clipping(params,max_l2_norm,eps,device):
    params = list(params)
    total_norm_sq = torch.zeros((), device=device)
    for param in params:
        if param.grad is not None:
            total_norm_sq += torch.sum(param.grad ** 2)
    total_norm = total_norm_sq.sqrt()
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale)
    return total_norm


def data_loading(data,batch_size,context_length,device="cpu"):
    length = len(data)
    if length <= context_length:
        raise ValueError("训练数据必须长于context_length，并且至少包含一个目标 Token")
    X_array = []
    Y_array = []
    i = random.randint(0,length - context_length - 1)
    for _ in range(batch_size):
        X_array.append(data[i:i+context_length])
        Y_array.append(data[i+1:i+context_length+1])
        i = random.randint(0,length - context_length - 1)
    X = torch.as_tensor(np.stack(X_array),dtype=torch.long)
    Y = torch.as_tensor(np.stack(Y_array),dtype=torch.long)
    return X.to(device),Y.to(device)
