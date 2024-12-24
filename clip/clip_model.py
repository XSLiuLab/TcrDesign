import torch
from torch import nn
from ..model import BERT

class CLIP_cdr3Pair(nn.Module):
    def __init__(self, aCDR3_encoder: BERT, bCDR3_encoder: BERT, embedding_size):
        super().__init__()
        
        self.aCDR3_encoder = aCDR3_encoder
        self.bCDR3_encoder = bCDR3_encoder
        self.embedding_size = embedding_size
        
        self.temperature_parameter = nn.Parameter(torch.randn(1)) # random sample from normal distribution
        self.linearA = nn.Linear(aCDR3_encoder.hidden, embedding_size, bias=False)
        self.linearB = nn.Linear(bCDR3_encoder.hidden, embedding_size, bias=False)
    
    def forward(self, aCDR3, bCDR3):
        aCDR3_feature = self.aCDR3_encoder(aCDR3)
        aCDR3_feature = self.L2_normalization(self.linearA(aCDR3_feature.mean(dim=1)))
        # aCDR3_feature = aCDR3_feature.mean(dim=1)
        
        bCDR3_feature = self.bCDR3_encoder(bCDR3)
        bCDR3_feature = self.L2_normalization(self.linearB(bCDR3_feature.mean(dim=1)))
        # bCDR3_feature = bCDR3_feature.mean(dim=1)
        
        logits = aCDR3_feature @ bCDR3_feature.t() * torch.exp(self.temperature_parameter)
        
        return logits
        
    def L2_normalization(self, x):
        return x / torch.norm(x, dim=-1, keepdim=True)
