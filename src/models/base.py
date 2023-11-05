import torch 
import torch.nn as nn 
from abc import ABC, abstractmethod
import numpy as np

def map_pytree(f, item):
    if isinstance(item, dict):
        return {k: map_pytree(f, v) for k, v in item.items()}
    elif isinstance(item, list) or isinstance(item, set) or isinstance(item, tuple):
        return [map_pytree(f, v) for v in item]
    elif isinstance(item, np.ndarray) or isinstance(item, torch.Tensor):
        return f(item)
    else:
        return item

def to(item, device):
    return map_pytree(lambda x: torch.tensor(x).to(device), item)

class BaseModel(ABC, nn.Module):
    def __init__(self, dataset, device):
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.maxlen = self.dataset.max_len

    def prepare_inputs(self, items):
        if isinstance(items, dict):
            return items
        return to(self.dataset.collate(items, self.device), self.device)

    @abstractmethod
    def get_loss(self, items, **kwargs):
        pass

class BaseTransformer(BaseModel):
    def __init__(self, pretrained_model, dataset, device):
        super().__init__(dataset, device)
        self.model = pretrained_model
        self.model.resize_token_embeddings(self.dataset.tokenizer.num_tokens())

class Policy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def act(self, items) -> str:
        pass

    def train(self):
        pass

    def eval(self):
        pass