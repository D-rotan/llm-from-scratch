import torch


def save_checkpoint(model: torch.nn.Module,optimizer: torch.optim.Optimizer,iteration: int,out: str):
    checkpoint = {
        "model_data":model.state_dict(),
        "optimizer_data":optimizer.state_dict(),
        "iteration_data":iteration
    }
    torch.save(checkpoint,out)


def load_checkpoint(src: str,model: torch.nn.Module,optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src,weights_only=True)
    model.load_state_dict(checkpoint["model_data"])
    optimizer.load_state_dict(checkpoint["optimizer_data"])
    return checkpoint["iteration_data"]