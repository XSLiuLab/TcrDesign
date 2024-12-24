import torch.nn as nn
import torch.nn.functional as F

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model (optional) + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size, has_next=False):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.has_next = has_next
        # 两个预训练任务可以选择性训练
        if has_next:
            self.next_sentence = NextSentencePrediction(self.bert.hidden)  
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label=None):
        x = self.bert(x, segment_label)
        output = (self.next_sentence(x), self.mask_lm(x)) if self.has_next else self.mask_lm(x)
        return output # 返回的是softmax过后的概率值

class BERTLM_pMHC(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model (optional) + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.binding_task = mhcBindingPrediction(self.bert.hidden)  
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label=None):
        x = self.bert(x, segment_label)
        output = self.binding_task(x), self.mask_lm(x)
        return output

class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))
    

class mhcBindingPrediction(nn.Module):
    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear1 = nn.Linear(hidden, 64)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.selu(self.linear1(x.mean(dim=1)))
        x = self.dropout(x)
        # x = self.sigmoid(self.linear2(x))
        x = self.linear2(x)
        return x


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
