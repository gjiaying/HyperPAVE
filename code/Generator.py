from transformers import GPT2LMHeadModel, GPT2Config, pipeline
import torch
import numpy as np
import torch.nn as nn
import sys

class Generator(nn.Module):
    def __init__(self, max_length): 
        nn.Module.__init__(self)
        self.generator = pipeline('text-generation', model='gpt2', pad_token_id = 50256)
        self.max_length = max_length
        

    def forward(self, inputs):    
        inputs = 'What is ' + inputs + '?'
        generated_text = self.generator(inputs, max_length=self.max_length, num_return_sequences=1)[0]['generated_text']
        return generated_text