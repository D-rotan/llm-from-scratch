import torch
from typing import Optional
from collections.abc import Callable, Iterable
class AdamW(torch.optim.Optimizer):
    def __init__(self,params,betas,weight_decay,eps,lr):
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
        defaults = {"lr":lr,
                    "weight_decay":weight_decay,
                    "betas":betas,
                    "eps":eps}
        super().__init__(params=params,defaults=defaults)
    def step(self,closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            betas = group["betas"]
            beta1 = betas[0]
            beta2 = betas[1]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m = state["m"]
                v = state["v"]
                t = state.get("step",1)
                lr_t = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                grad = p.grad.data
                p.data -= weight_decay * p.data * lr
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                state["m"] = m
                state["v"] = v
                p.data -= lr_t * (m / (v ** 0.5 + eps))
                state["step"] = t + 1
        return loss