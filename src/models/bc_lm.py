from .gpt2_optional_final_ln import GPT2LMHeadModel
from transformers.modeling_utils import PreTrainedModel
#from data.rl_data import DataPoint, RL_Dataset
#from models.base import BaseTransformer, Evaluator, InputType
from .base import BaseTransformer, Policy
from .model_utils import pad_sequence, map_all_kvs, always_terminate, update_kvs, process_logits
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import json

def load_bclm(args, dataset):
    obj = GPT2LMHeadModel
    gpt2 = obj.from_pretrained(args.gpt2_type)
    model = BC_LM(model=gpt2, dataset=dataset)
    model = model.to(args.device)
    return model

def load_evaluator(args, model):
    evaluator = BC_Evaluator(args.metadata_path, model)
    return evaluator

class BC_LM(BaseTransformer):
    def __init__(self, 
                 model, 
                 dataset, 
                 device = "cuda", 
                 transition_weight: float=0.0, 
                ):
        assert isinstance(model, GPT2LMHeadModel)
        super().__init__(model, dataset, device)
        self.h_dim  = self.model.config.n_embd
        self.transition_weight = transition_weight


    def forward(self, tokens, attn_mask, **kwargs):
        # tokens – b,t
        # attn_mask – b,t
        # prefix_embs – b,t',d
        # prefix_attn_mask - b, t'
        input_embeddings = self.model.transformer.wte(tokens)
        model_outputs = self.model(inputs_embeds=input_embeddings, 
                                   attention_mask=attn_mask, 
                                   **kwargs)
        return model_outputs

    def get_weights(self, 
                    tokens: torch.Tensor, 
                    action_idxs: torch.Tensor):
        weights = torch.full(tokens.shape, self.transition_weight).to(self.device)
        if action_idxs.shape[1] == 0:
            n = torch.zeros((tokens.shape[0],)).long().to(self.device)
        else:
            n = torch.argmax(action_idxs, dim=1)+1
        for i in range(tokens.shape[0]):
            weights[i] = torch.scatter(weights[i], dim=0, index=action_idxs[i, :n[i]], src=torch.full((n[i].item(),), 1.0).to(self.device))
        return weights
    
    def awac_loss(self, tokens, attn_mask, logits, w):
        w = w.detach()
        losses = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction='none')
        losses = losses.reshape(tokens.shape[0], tokens.shape[1]-1)
        return (losses * w[:, :-1] * attn_mask[:, 1:]).sum() / attn_mask[:, 1:].sum()

    def get_loss(self, items):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        a_idx = prepared_inputs['action_idxs']
        model_outputs = self(tokens, attn_mask, 
                             output_attentions=True)
        logs = {}
        '''
        transformer_logs = get_transformer_logs(model_outputs.attentions, 
                                                self.model, 
                                                    attn_mask)
        '''
        n = attn_mask.sum().item()
        weights = self.get_weights(tokens, a_idx)
        token_loss = self.awac_loss(tokens, attn_mask, model_outputs.logits, weights)
        logs['loss'] = (token_loss.item(), n)
        #logs['transformer'] = transformer_logs
        return token_loss, logs, []
    
'''    
    def score(self, model_args, model_kwargs, 
              temp: float=1.0, 
              top_k: Optional[int]=None, 
              top_p: Optional[float]=None):
        model_outputs = self(*model_args, **model_kwargs)
        logits = process_logits(model_outputs.logits, temp=temp, top_k=top_k, top_p=top_p)
        return torch.log(F.softmax(logits, dim=-1)), model_outputs
    
    def get_scores(self, 
                   items: InputType, 
                   temp: float=1.0, 
                   top_k: Optional[int]=None, 
                   top_p: Optional[float]=None) -> torch.Tensor:
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        return self.score((tokens, attn_mask,), {}, 
                          temp=temp, top_k=top_k, top_p=top_p)[0]

    def initial_score(self, 
                      items: InputType, 
                      temp: float=1.0, 
                      top_k: Optional[int]=None, 
                      top_p: Optional[float]=None) -> Tuple[torch.Tensor, Any]:
        prepared_inputs = self.prepare_inputs(items)
        tokens = prepared_inputs['tokens']
        scores, model_outputs = self.score((tokens, None,), {'use_cache': True}, 
                                           temp=temp, top_k=top_k, top_p=top_p)
        return scores[:, -1, :], model_outputs.past_key_values
    
    def next_score(self, 
                   tokens: torch.Tensor, 
                   state: Any, 
                   temp: float=1.0, 
                   top_k: Optional[int]=None, 
                   top_p: Optional[float]=None) -> Tuple[torch.Tensor, Any]:
        scores, model_outputs = self.score((tokens.unsqueeze(1), None,), 
                                            {'use_cache': True, 
                                             'past_key_values': state}, 
                                           temp=temp, top_k=top_k, top_p=top_p)
        return scores.squeeze(1), model_outputs.past_key_values

'''

class BC_Policy(Policy):
    def __init__(self, bc_lm: BC_LM, **generation_kwargs) -> None:
        super().__init__()
        self.bc_lm = bc_lm
        self.generation_kwargs = generation_kwargs
    
    def sample_raw(self, 
                   tokens: torch.Tensor, attn_mask: torch.Tensor, 
                   termination_condition: Callable[[np.ndarray], bool], 
                   num_generations=1, max_generation_len=None, 
                   temp=1.0, top_k=None, top_p=None, 
                   prefix_embs: Optional[torch.Tensor]=None, 
                   prefix_attn_mask: Optional[torch.Tensor]=None, 
                   remove_prefix_position_embs: bool=False):
        tokenizer = self.bc_lm.dataset.tokenizer
        max_length = self.bc_lm.dataset.max_len
        if max_length is None:
            max_length = self.bc_lm.model.config.n_positions
        max_length = min(max_length, self.bc_lm.model.config.n_positions)
        device = self.bc_lm.device
        bsize = tokens.shape[0]
        n = bsize * num_generations
        if max_generation_len is None:
            max_generation_len = max_length+1
        input_strs = [tokenizer.decode(tokens[i, :][:attn_mask[i, :].sum().long()].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        '''
        model_outputs = self.bc_lm(tokens, attn_mask, prefix_embs=prefix_embs, 
                                   prefix_attn_mask=prefix_attn_mask, 
                                   remove_prefix_position_embs=remove_prefix_position_embs, 
                                   use_cache=True)
        '''
        model_outputs = self.bc_lm(tokens, attn_mask, use_cache = True)
        dialogue_kvs = model_outputs.past_key_values
        dialogue_lens = attn_mask.sum(dim=1)
        tokens = pad_sequence(torch.repeat_interleave(tokens, num_generations, dim=0), max_length, tokenizer.pad_token_id, device, 1)
        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)
        dialogue_kvs = map_all_kvs(lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), max_length, 0.0, device, 2), dialogue_kvs)
        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(device)
        t = torch.min(dialogue_lens).int()
        while termination_mask.sum() > 0 and (t+prefix_t) < max_length:
            curr_token = tokens[:, t-1].unsqueeze(1)
            curr_dialogue_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], dialogue_kvs)
            transformer_outputs = self.bc_lm(curr_token, None, past_key_values=curr_dialogue_kvs, use_cache=True)
            logits = transformer_outputs.logits
            logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)
            logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(t < dialogue_lens, 1e7)
            logits = process_logits(transformer_outputs.logits, temp=temp, top_k=top_k, top_p=top_p)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits[:, 0])
            new_tokens = cat_dist.sample()
            log_probs += cat_dist.log_prob(new_tokens)
            tokens[:, t] = new_tokens
            dialogue_kvs = update_kvs(dialogue_kvs, transformer_outputs.past_key_values, torch.arange(0, n).to(device), (t+prefix_t)-1)
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= (1 - int(termination_condition(tokenizer.decode(tokens[idx, :].tolist(), 
                                                                                             clean_up_tokenization_spaces=False))))
            t += 1
            termination_mask *= ((t-dialogue_lens) < max_generation_len).int()
    
        output_strs = [tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(num_generations):
                processed_str = output_strs[i*num_generations+x][len(input_strs[i]):].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.pad_token_id))].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.eoa_token_id))].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return list(zip(input_strs, processed_outputs)), log_probs.reshape(-1, num_generations)

    def act(self, items) -> str:
        prepared_inputs = self.bc_lm.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        generations, probs = self.sample_raw(tokens, attn_mask, always_terminate)
        # generations : list of (input, output) with len : batch
        # generations[]
        #sorted_outputs = list(zip(*sorted(zip(generations[0][1], probs[0]), key=lambda x: -x[1])))[0]
        #return sorted_outputs[0]
        outputs = [gen[1] for gen in generations]
        return outputs
    
    def train(self):
        self.bc_lm.train()
    
    def eval(self):
        self.bc_lm.eval()

class BC_Evaluator():
    def __init__(self, path, model:BC_LM,  **generation_kwargs):
        # dictionary with key : asin // val : attributes
        self.metadata = json.load(open(path))
        self.generation_kwargs = generation_kwargs
        self.model = model
        
    def evaluate(self, items):
        policy = BC_Policy(self.model, **self.generation_kwargs)
        return policy.act(items)