import torch
import torch.nn as nn

from ..model import BERT
from ..model.attention import MultiHeadedAttention


class BertForSequenceClassification(nn.Module):
    """
    BERT model for classification.
    :param bert: BERT model which should be trained
    :param vocab_size: total vocab size for masked_lm
    """

    def __init__(self, alphaEncoder: BERT, betaEncoder: BERT, pMHCEncoder: BERT, 
                 pMHC_max, TCR_max, VJ_hidden, VJ_vocab_size):
        super(BertForSequenceClassification, self).__init__()
        self.alphaEncoder = alphaEncoder
        self.betaEncoder = betaEncoder
        self.pMHCEncoder = pMHCEncoder
        self.pMHC_max = pMHC_max
        self.TCR_max = TCR_max
        self.VJ_hidden = VJ_hidden
        self.VJ_vocab_size = VJ_vocab_size
        
        self.pmhc_linear = nn.Linear(self.pMHCEncoder.hidden, 1)
        self._pmhc_linear = nn.Linear(self.pMHC_max, 1)
        self.atcr_linear = nn.Linear(self.alphaEncoder.hidden, 1)
        self._atcr_linear = nn.Linear(self.TCR_max, 1)
        self.btcr_linear = nn.Linear(self.betaEncoder.hidden, 1)
        self._btcr_linear = nn.Linear(self.TCR_max, 1)
        
        self.VJembedding = nn.Embedding(self.VJ_vocab_size, self.VJ_hidden)
        
        self.mlp = nn.Sequential(
            nn.Linear((self.pMHCEncoder.hidden + self.alphaEncoder.hidden + self.betaEncoder.hidden + \
                self.pMHC_max + self.TCR_max + self.TCR_max + self.VJ_hidden*4) * 1, 400),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(100, 1)    
        )
    
    def forward(self, aCDR3, bCDR3, pMHC, VJ, pMHC_segment_label):
        aCDR3_feature = self.alphaEncoder(aCDR3)
        bCDR3_feature = self.betaEncoder(bCDR3)
        pmhc_feature = self.pMHCEncoder(pMHC, pMHC_segment_label)
        VJ_feature = self.VJembedding(VJ)
        
        atcr = self.atcr_linear(aCDR3_feature).squeeze(-1)
        atrc_ = self._atcr_linear(aCDR3_feature.transpose(1, 2)).squeeze(-1)
        btcr = self.btcr_linear(bCDR3_feature).squeeze(-1)
        btrc_ = self._btcr_linear(bCDR3_feature.transpose(1, 2)).squeeze(-1)
        pmhc = self.pmhc_linear(pmhc_feature).squeeze(-1)
        pmhc_ = self._pmhc_linear(pmhc_feature.transpose(1, 2)).squeeze(-1)
        
        VJ_ = VJ_feature.view(-1, self.VJ_hidden * 4)
        
        out = self.mlp(torch.cat([atcr, atrc_, btcr, btrc_, pmhc, pmhc_, VJ_], dim=-1))
        
        return out  # 输出概率值
 
 
class BertForSequenceClassification_withAttn(nn.Module):
    """
    BERT model for classification.
    :param bert: BERT model which should be trained
    :param vocab_size: total vocab size for masked_lm
    """

    def __init__(self, alphaEncoder: BERT, betaEncoder: BERT, pMHCEncoder: BERT, 
                 pMHC_max, TCR_max, VJ_hidden, VJ_vocab_size, attn_heads, hidden):
        super(BertForSequenceClassification_withAttn, self).__init__()
        self.alphaEncoder = alphaEncoder
        self.betaEncoder = betaEncoder
        self.pMHCEncoder = pMHCEncoder
        self.pMHC_max = pMHC_max
        self.TCR_max = TCR_max
        self.VJ_hidden = VJ_hidden
        self.VJ_vocab_size = VJ_vocab_size
        
        self.pmhc_linear = nn.Linear(self.pMHCEncoder.hidden, 1)
        self._pmhc_linear = nn.Linear(self.pMHC_max, 1)
        self.atcr_linear = nn.Linear(self.alphaEncoder.hidden, 1)
        self._atcr_linear = nn.Linear(self.TCR_max, 1)
        self.btcr_linear = nn.Linear(self.betaEncoder.hidden, 1)
        self._btcr_linear = nn.Linear(self.TCR_max, 1)
        
        self.VJembedding = nn.Embedding(self.VJ_vocab_size, self.VJ_hidden)
        
        self.mlp = nn.Sequential(
            nn.Linear((self.pMHCEncoder.hidden + self.alphaEncoder.hidden + self.betaEncoder.hidden + \
                self.pMHC_max + self.TCR_max + self.TCR_max + self.VJ_hidden*4) * 1, 400),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(100, 1)    
        )
        
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=0)
    
    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size,len_q = seq_q.size()
        batch_size,len_k = seq_k.size()
        pad_attn_mask = seq_k.data.ne(0).unsqueeze(1) # pad token为False, pad token index: 0
        return pad_attn_mask.expand(batch_size,len_q,len_k)
    
    def forward(self, aCDR3, bCDR3, pMHC, VJ, pMHC_segment_label):
        aCDR3_feature = self.alphaEncoder(aCDR3)
        bCDR3_feature = self.betaEncoder(bCDR3)
        pmhc_feature = self.pMHCEncoder(pMHC, pMHC_segment_label)
        VJ_feature = self.VJembedding(VJ)
        
        # attn 操作 Q:aCDR3 or bCDR3; KV:pMHC
        a_mask = self.get_attn_pad_mask(aCDR3, pMHC).unsqueeze(1)
        b_mask = self.get_attn_pad_mask(bCDR3, pMHC).unsqueeze(1)
        aCDR3_feature = self.attention.forward(aCDR3_feature, pmhc_feature, pmhc_feature, mask=a_mask) + aCDR3_feature
        bCDR3_feature = self.attention.forward(bCDR3_feature, pmhc_feature, pmhc_feature, mask=b_mask) + bCDR3_feature
        
        atcr = self.atcr_linear(aCDR3_feature).squeeze(-1)
        atrc_ = self._atcr_linear(aCDR3_feature.transpose(1, 2)).squeeze(-1)
        btcr = self.btcr_linear(bCDR3_feature).squeeze(-1)
        btrc_ = self._btcr_linear(bCDR3_feature.transpose(1, 2)).squeeze(-1)
        pmhc = self.pmhc_linear(pmhc_feature).squeeze(-1)
        pmhc_ = self._pmhc_linear(pmhc_feature.transpose(1, 2)).squeeze(-1)
        
        VJ_ = VJ_feature.view(-1, self.VJ_hidden * 4)
        
        out = self.mlp(torch.cat([atcr, atrc_, btcr, btrc_, pmhc, pmhc_, VJ_], dim=-1))
        
        return out  # 输出概率值


class BertForSequenceClassification_withAttn_returnEmbedding(nn.Module):
    """
    BERT model for classification.
    :param bert: BERT model which should be trained
    :param vocab_size: total vocab size for masked_lm
    """

    def __init__(self, alphaEncoder: BERT, betaEncoder: BERT, pMHCEncoder: BERT, 
                 pMHC_max, TCR_max, VJ_hidden, VJ_vocab_size, attn_heads, hidden):
        super(BertForSequenceClassification_withAttn_returnEmbedding, self).__init__()
        self.alphaEncoder = alphaEncoder
        self.betaEncoder = betaEncoder
        self.pMHCEncoder = pMHCEncoder
        self.pMHC_max = pMHC_max
        self.TCR_max = TCR_max
        self.VJ_hidden = VJ_hidden
        self.VJ_vocab_size = VJ_vocab_size
        
        self.pmhc_linear = nn.Linear(self.pMHCEncoder.hidden, 1)
        self._pmhc_linear = nn.Linear(self.pMHC_max, 1)
        self.atcr_linear = nn.Linear(self.alphaEncoder.hidden, 1)
        self._atcr_linear = nn.Linear(self.TCR_max, 1)
        self.btcr_linear = nn.Linear(self.betaEncoder.hidden, 1)
        self._btcr_linear = nn.Linear(self.TCR_max, 1)
        
        self.VJembedding = nn.Embedding(self.VJ_vocab_size, self.VJ_hidden)
        
        self.mlp = nn.Sequential(
            nn.Linear((self.pMHCEncoder.hidden + self.alphaEncoder.hidden + self.betaEncoder.hidden + \
                self.pMHC_max + self.TCR_max + self.TCR_max + self.VJ_hidden*4) * 1, 400),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(100, 1)    
        )
        
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=0)
    
    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size,len_q = seq_q.size()
        batch_size,len_k = seq_k.size()
        pad_attn_mask = seq_k.data.ne(0).unsqueeze(1) # pad token为False, pad token index: 0
        return pad_attn_mask.expand(batch_size,len_q,len_k)
    
    def forward(self, aCDR3, bCDR3, pMHC, VJ, pMHC_segment_label):
        aCDR3_feature = self.alphaEncoder(aCDR3)
        bCDR3_feature = self.betaEncoder(bCDR3)
        pmhc_feature = self.pMHCEncoder(pMHC, pMHC_segment_label)
        VJ_feature = self.VJembedding(VJ)
        
        # attn 操作 Q:aCDR3 or bCDR3; KV:pMHC
        a_mask = self.get_attn_pad_mask(aCDR3, pMHC).unsqueeze(1)
        b_mask = self.get_attn_pad_mask(bCDR3, pMHC).unsqueeze(1)
        aCDR3_feature = self.attention.forward(aCDR3_feature, pmhc_feature, pmhc_feature, mask=a_mask) + aCDR3_feature
        bCDR3_feature = self.attention.forward(bCDR3_feature, pmhc_feature, pmhc_feature, mask=b_mask) + bCDR3_feature
        
        atcr = self.atcr_linear(aCDR3_feature).squeeze(-1)
        atrc_ = self._atcr_linear(aCDR3_feature.transpose(1, 2)).squeeze(-1)
        btcr = self.btcr_linear(bCDR3_feature).squeeze(-1)
        btrc_ = self._btcr_linear(bCDR3_feature.transpose(1, 2)).squeeze(-1)
        pmhc = self.pmhc_linear(pmhc_feature).squeeze(-1)
        pmhc_ = self._pmhc_linear(pmhc_feature.transpose(1, 2)).squeeze(-1)
        
        VJ_ = VJ_feature.view(-1, self.VJ_hidden * 4)
        
        out = self.mlp(torch.cat([atcr, atrc_, btcr, btrc_, pmhc, pmhc_, VJ_], dim=-1))
        
        return out, aCDR3_feature, bCDR3_feature, pmhc_feature
