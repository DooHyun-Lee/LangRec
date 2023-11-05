import torch 
import torch.nn.functional as F

pad_sequence = lambda seq, to_len, val, device, dim: torch.cat((seq, torch.full((*seq.shape[:dim], to_len-seq.shape[dim], *seq.shape[(dim+1):]), val).to(device)), dim=dim)
map_all_kvs = lambda f, kvs: tuple([tuple(map(f, items)) for items in kvs])

def update_kvs(kvs, updated_kvs, lens_chosen, idx):
    for i, layer in enumerate(kvs):
        for x, item in enumerate(layer):
            item[lens_chosen, :, idx, :] = updated_kvs[i][x][:, :, idx, :]
    return kvs

def always_terminate(s):
    return True

def top_k_logits(logits, k):
    # logits = (batch, time, dim)
    _, bottom_k_idx = torch.topk(-logits, logits.shape[2]-k, dim=2)
    return torch.scatter(logits, dim=2, index=bottom_k_idx, value=float('-inf'))

def top_p_logits(logits, p):
    # logits = (batch, time, dim)
    sorted_logits, _ = torch.sort(logits, dim=2, descending=True)
    num_to_take = torch.sum(torch.cumsum(F.softmax(sorted_logits, dim=2), dim=2) <= p, dim=2).unsqueeze(2)
    mask = logits < torch.gather(sorted_logits, dim=2, index=torch.clamp(num_to_take, max=logits.shape[2]-1))
    return logits.masked_fill(mask, float('-inf'))

def process_logits(logits, temp=1.0, top_k=None, top_p=None):
    logits /= temp
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None:
        logits = top_p_logits(logits, top_p)
    return logits