import torch.nn as nn
import torch.nn.functional as F
import pdb
from transformers import AutoTokenizer, AutoModel
import torch
from itertools import count
from typing import Union, Callable




# the model class extends the nn.Module
class ClassifierCodeT5(nn.Module):
    def __init__(self,encoder):
        super().__init__()

        self.encoder = encoder

        config = encoder.config
        self.linear = nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)
        self.dropout= nn.Dropout(0.1)


    def raw_emb(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is not None:
            x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state # size: [bsz, len, dim]
        else:
            x = self.encoder(input_ids=input_ids).last_hidden_state

        return x[:,0,:]

    def forward(self, input_ids=None, attention_mask=None, labels=None, smart =False, target = None):

        if smart: # only compute logits
            x = self.linear(target)
            x = F.tanh(x)
            #x = F.normalize(x, dim=1)
            return x
        else:
            raw_emb = self.raw_emb(input_ids=input_ids, attention_mask= attention_mask)
            x = self.linear(raw_emb)
            x = F.tanh(x)
            #x = F.normalize(x, dim=1)

        return x, raw_emb


    def save(self, source):
        saved = self.encoder.save_pretrained(str(source))

        return saved

