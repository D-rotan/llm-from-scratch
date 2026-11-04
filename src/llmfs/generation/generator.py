import torch
from ..model.llm import Transformer_lm
from ..tokenizer.tokenizer import Tokenizer

from ..configs.schema import GenerationConfig

def generate(config:GenerationConfig):
    model = Transformer_lm(vocab_size=config.vocab_size,context_length=config.context_length,
                           num_layers=config.num_layers,num_heads=config.num_heads,
                           d_model=config.d_model,d_ff=config.d_ff,rope_theta=config.rope_theta,
                           eps_RMS1=config.eps_RMS1,eps_RMS2=config.eps_RMS2,eps_RMS3=config.eps_RMS3).to(config.device)
    
    model.eval()
    with torch.no_grad():
        model_weights = torch.load(config.params_path,map_location=config.device,weights_only=True)
        if "model_data" in model_weights:
            model_weights = model_weights["model_data"]
        model.load_state_dict(model_weights)
        tokenizer = Tokenizer(config=config.tokenizer)
        ids = list(tokenizer.encode(config.prompt))
        token_ids = torch.tensor([ids],dtype=torch.long,device=config.device)
        result = decoder(token_ids=token_ids,max_new_tokens=config.max_new_tokens,
                context_length=config.context_length,model=model,
                temperature=config.temperature,top_p=config.top_p)
    return tokenizer.decode(result)
        

#目前只支持单batch推理
def decoder(token_ids:torch.Tensor,max_new_tokens:int,context_length:int,
            model:Transformer_lm,temperature:float,top_p:float):
    # size = token_ids.shape[-1]
    for _ in range(max_new_tokens):
        context = token_ids[:, -context_length:]
        next_token = sample_next_token(logits=model(context),T=temperature,p=top_p)
        token_ids = torch.cat((token_ids,next_token),dim=-1)

    # return token_ids[0, size:].tolist()
    return token_ids[0,:].tolist()


def temperature_scaling(logits:torch.Tensor,T:float):
    #logits(batch_size,seq_len,vocab_size)
    logits_max = torch.max(logits,dim=-1,keepdim=True).values
    if T <= 0:
        raise ValueError("Temperture too small")
    logits = (logits-logits_max) / T
    exp_logits = torch.exp(logits)
    result = exp_logits / torch.sum(exp_logits,dim=-1,keepdim=True)
    return result
def top_p(q:torch.Tensor,p:float):
    #q is probability distribution
    #logits(batch_size,seq_len,vocab_size)
    if not 0 < p <= 1:
        raise ValueError("top_p must be in the range (0, 1]")
    values,indices = torch.sort(q,dim=-1,descending=True)
    cumulative = torch.cumsum(values,dim=-1)
    mask_index = cumulative - values < p
    mask = torch.zeros_like(q,dtype=torch.bool)
    mask.scatter_(-1,indices,mask_index)
    q = torch.masked_fill(q,~mask,0)
    result = q / torch.sum(q,dim=-1,keepdim=True)    
    return result
def sample_next_token(logits:torch.Tensor,T:float,p:float):
    probs = top_p(temperature_scaling(logits=logits,T=T),p=p)
    probs = probs[:,-1,:]
    final_token = torch.multinomial(probs,num_samples=1)
    return final_token
