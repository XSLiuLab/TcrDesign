from torch.utils.data import Dataset
import pandas as pd
import torch
import tqdm

class CLIP_dataset(Dataset):
    def __init__(self, data_path, vocab, max_length=22):
        super().__init__()
        self.data_path = data_path
        self.vocab = vocab
        self.max_length = max_length
        
        self.data = pd.read_csv(data_path)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        CDR3a_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.iloc[index]['CDR3a']] + [self.vocab.eos_index]
        CDR3b_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.iloc[index]['CDR3b']] + [self.vocab.eos_index]
        
        CDR3a_ids = CDR3a_ids[:self.max_length]
        CDR3b_ids = CDR3b_ids[:self.max_length]
        padding_a = [self.vocab.pad_index for _ in range(self.max_length - len(CDR3a_ids))]
        padding_b = [self.vocab.pad_index for _ in range(self.max_length - len(CDR3b_ids))]
        CDR3a_ids.extend(padding_a), CDR3b_ids.extend(padding_b)
        
        output = {
            'CDR3a': torch.tensor(CDR3a_ids, dtype=torch.long),
            'CDR3b': torch.tensor(CDR3b_ids, dtype=torch.long)
        }
        
        return output
