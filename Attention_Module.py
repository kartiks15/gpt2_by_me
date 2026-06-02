import torch 
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_len, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))
    

    def forward(self, x):
        b, 
