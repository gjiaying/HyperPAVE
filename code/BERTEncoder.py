from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import torch.nn as nn
import sys

class BERTEncoder(nn.Module):
    '''
    BERT Encoder
    '''

    def __init__(self, max_length): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def forward(self, text):
        x = self.bert(text)       
        return x[1]
    
    def tokenize(self, raw_tokens):
        # token -> index
        
        raw_tokens = '[CLS] ' + raw_tokens
        raw_tokens = raw_tokens.split(' ')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(raw_tokens)
        
        # padding, pad zeros when length < maximum length
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
        tokens_tensor = torch.tensor([indexed_tokens])

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1
        mask = torch.tensor([mask])

        return tokens_tensor, mask
    
    def other_tokenize(self, raw_tokens):
        # token -> index
        token_dict = self.tokenizer(raw_tokens)
        token_ids = token_dict.input_ids
        
        # padding, pad zeros when length < maximum length
        while len(token_ids) < self.max_length:
            token_ids.append(0)
        token_ids = token_ids[:self.max_length]
        token_ids = torch.tensor([token_ids])

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(token_ids)] = 1
        mask = torch.tensor([mask])
        return token_ids, mask