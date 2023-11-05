from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch 
from torch.utils.data import Dataset
from environment import Language_Observation, RecObservation
from tokenizer import Tokenizer, RecTokenizer
from base import RecommendData
import numpy as np
from tqdm import tqdm

def load_RecListDataset(path, max_len=1024):
    rd = RecommendData(path)
    return RecommendListDataset(rd, max_len)

@dataclass
class DataPoint:
    raw_str : str
    tokens : List[int]
    state_idxs : List[int]
    action_idxs : List[int]
    terminals : List[int]

    def to_tensors(self, device, max_length:Optional[int]):
        tok = torch.tensor(self.tokens).to(device)
        s = torch.tensor(self.state_idxs).long().to(device)
        a = torch.tensor(self.action_idxs).long().to(device)
        term = torch.tensor(self.terminals).to(device)
        if max_length is not None:
            tok = tok[:max_length]
            s = s[:(s < max_length).sum()]
            a = a[:max(min((a < (max_length-1)).sum().item(), s.shape[0]-1), 0)]
            term = term[:s.shape[0]]
        return tok, s, a, term

    @classmethod
    def from_obs(cls, obs:Language_Observation, tokenizer:Tokenizer):
        # should update reward in this function
        sequence, terminal = obs.to_sequence()
        raw_str = tokenizer.id_to_token(tokenizer.bos_token_id)
        for i, s in enumerate(sequence):
            # i : even -> state (review)
            # i : odd -> action (attribute)
            raw_str += s
            if (i % 2) ==0:
                raw_str += tokenizer.id_to_token(tokenizer.eos_token_id)
            else:
                raw_str += tokenizer.id_to_token(tokenizer.eoa_token_id)
        if terminal:
            raw_str += tokenizer.id_to_token(tokenizer.eod_token_id)
        tokens = tokenizer.encode(raw_str)[0] # encode returns encoded result, attention mask
        state_idxs = []
        action_idxs = []
        curr_idx = 0
        for i, t in enumerate(tokens):
            # state_idxs ... but why? 
            if t == tokenizer.eos_token_id:
                curr_idx = i
            elif t == tokenizer.eoa_token_id:
                # TODO : maybe don't include first recommending action?  
                action_idxs.extend(list(range(curr_idx, i))) # record from <\s> ~ before <\a>
                state_idxs.extend(list(range(curr_idx, i)))
                curr_idx = i
        state_idxs.append(len(tokens)-1)
        terminals = ([0] * (len(state_idxs)-1)) + [int(terminal)]
        
        # enhancement not to iterate all over tokens(too much time.. )
        # not big enhancement 
        '''
        token_array = np.array(tokens)
        eos_indices = np.where(token_array == tokenizer.eos_token_id)
        eoa_indices = np.where(token_array == tokenizer.eoa_token_id)
        ranges = list(zip(eos_indices[0], eoa_indices[0]))
        state_idxs = []
        action_idxs = []
        for (start, end) in ranges:
            action_idxs.extend(list(range(start, end)))
            state_idxs.extend(list(range(start, end)))
            
        state_idxs.append(len(tokens)-1)
        terminals = ([0] * (len(state_idxs)-1)) + [int(terminal)]
        '''
        return cls(raw_str, tokens, state_idxs, action_idxs, terminals)

class RL_Dataset(ABC):
    def __init__(self, 
                 tokenizer : Tokenizer, 
                 max_len : Optional[int]) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def collate(self, items : List[DataPoint], device):
        tokens, state_idxs, action_idxs, terminals = zip(*map(lambda x:x.to_tensors(device, self.max_len), items))
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = (tokens != self.tokenizer.pad_token_id).float() # 1 if not padding token
        state_idxs = torch.nn.utils.rnn.pad_sequence(state_idxs, batch_first=True, padding_value=0)
        action_idxs = torch.nn.utils.rnn.pad_sequence(action_idxs, batch_first=True, padding_value=0)
        terminals = torch.nn.utils.rnn.pad_sequence(terminals, batch_first=True, padding_value=1)
        return {'tokens':tokens, 'attn_mask':attn_mask, 'state_idxs':state_idxs, 'action_idxs':action_idxs, 'terminals':terminals}

class List_RL_Dataset(RL_Dataset):
    @abstractmethod
    def get_item(self, idx) -> DataPoint:
        pass

    @abstractmethod
    def size(self) -> int:
        pass 


class RecommendListDataset(List_RL_Dataset):
    def __init__(self, data: RecommendData,
                 max_len : Optional[int]) -> None:
        tokenizer = RecTokenizer()
        super().__init__(tokenizer, max_len)
        self.data = data
        self.datapoints = []
        for item in tqdm(self.data): # item : Scene 
            obs = RecObservation(item, item.events[-1]) # calls and makes events into sequence
            self.datapoints.append(DataPoint.from_obs(obs, self.tokenizer)) # convert sequence string into token, make state, action idx, attention masks

    def get_item(self, idx):
        return self.datapoints[idx]

    def size(self):
        return len(self.datapoints)

class GeneralDataset(Dataset):
    def __init__(self, rl_dataset:List_RL_Dataset, 
                 device):
        self.rl_dataset = rl_dataset
        self.device = device

    def __len__(self):
        return self.rl_dataset.size()   

    def __getitem__(self, i):
        return self.rl_dataset.get_item(i)

    def collate(self, items):
        return self.rl_dataset.collate(items, self.device)

    def collate_simple(self, items):
        return items

