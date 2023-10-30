from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
from transformers import GPT2Tokenizer

class Tokenizer(ABC):
    def __init__(self, pad_token_id, eos_token_id, eoa_token_id, bos_token_id, boa_token_id, eod_token_id):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id # end of attribute sentene 
        self.eoa_token_id = eoa_token_id # end of review sentence
        self.bos_token_id = bos_token_id # start of attribute sentence
        self.boa_token_id = boa_token_id # start of review sentence
        self.eod_token_id = eod_token_id

    @abstractmethod
    def encode(self, str_: Union[str, List[str]], **kwargs) -> Tuple[Union[List[int], List[List[int]]], Union[List[int], List[List[int]]]]:
        pass

    @abstractmethod
    def decode(self, tokens: Union[List[int], List[List[int]]], **kwargs) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def num_tokens(self) -> int:
        pass

    @abstractmethod
    def id_to_token(self, id_: int) -> str:
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def get_vocab(self) -> Any:
        pass

class RecTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['</a>', '<a>', '<stop>', '</eod>'], 
                                           'bos_token': '<s>', 
                                           'sep_token': '</s>', 
                                           'pad_token': '<|pad|>'})
        super.__init__(self.tokenizer.convert_tokens_to_ids('<|pad|>'), 
                         self.tokenizer.convert_tokens_to_ids('</s>'), 
                         self.tokenizer.convert_tokens_to_ids('</a>'), 
                         self.tokenizer.convert_tokens_to_ids('<s>'), 
                         self.tokenizer.convert_tokens_to_ids('<a>'), 
                         self.tokenizer.convert_tokens_to_ids('</eod>'))
        self.stop_token = self.tokenizer.convert_tokens_to_ids('<stop>')

    def encode(self, str_, **kwargs):
        items = self.tokenizer(
            str_, 
            add_special_tokens = False,
            padding = True,
            **kwargs
        )
        return items['input_ids'], items['attention_mask']

    def decode(self, tokens, **kwargs):
        if len(tokens) == 0:
            return ''
        if not isinstance(tokens[0], list):
            return self.tokenizer.decode(tokens, **kwargs)
        elif isinstance(tokens[0], list):
            return [self.decode(item) for item in tokens] # if input comes in batch
        else:
            raise ValueError('tokens must be a list of ints or a list of lists of ints')

    def num_tokens(self):
        return len(self.tokenizer)

    def id_to_token(self, id_):
        return self.tokenizer.convert_ids_to_tokens(id_)

    def token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_vocab(self):
        return list(self.tokenizer.get_vocab().keys())