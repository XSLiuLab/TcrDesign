from torch.utils.data import Dataset
import pandas as pd
import torch

# 生成模型的数据处理类 
# 动态决定序列的最大长度
class GenerateRNNdataset(Dataset):
    def __init__(self, dat_path, vocab, has_aCDR3=False, antigen_max_len=23):
        self.dat_path = dat_path
        self.has_aCDR3 = has_aCDR3
        self.antigen_max_len = antigen_max_len
        self.vocab = vocab
        
        df = pd.read_csv(self.dat_path)
        if has_aCDR3:
            df["total_length"] = df.alphaCDR3.str.len() + df.antigen.str.len()
        else:
            df["total_length"] = df.betaCDR3.str.len() + df.antigen.str.len()
            
        self.data = df.sort_values("total_length", ascending=True, ignore_index=True)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        # 获取数据并转化成id
        epitope_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.antigen[index]] + [self.vocab.eos_index]
        epitope_ids = epitope_ids[:self.antigen_max_len]
        padding = [self.vocab.pad_index for _ in range(self.antigen_max_len - len(epitope_ids))]
        epitope_ids.extend(padding)  # 满足our BERT模型的输入喜好，长度一致

        if self.has_aCDR3:
            cdr3_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.alphaCDR3[index]]
        else:
            cdr3_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.betaCDR3[index]]
        
        rnn_encoder_input = epitope_ids
        rnn_decoder_input = cdr3_ids
        rnn_decoder_target = rnn_decoder_input[1:] + [self.vocab.eos_index] 
        rnn_decoder_input_len = len(rnn_decoder_input)
        
        final_dict = {"rnn_encoder_input": rnn_encoder_input, 
                      "rnn_decoder_input": rnn_decoder_input, "rnn_decoder_input_len": rnn_decoder_input_len, 
                      "rnn_decoder_target": rnn_decoder_target}

        return final_dict


    def padding_batch(self, batch):
        rnn_decoder_input_lens = [d["rnn_decoder_input_len"] for d in batch] # batch的每个元素均为__getitem__获得
        decoder_maxlen = max(rnn_decoder_input_lens)
        
        # 补齐
        for d in batch: # 进行填充
            d["rnn_decoder_input"].extend([self.vocab.pad_index] * (decoder_maxlen - d["rnn_decoder_input_len"]))
            d["rnn_decoder_target"].extend([self.vocab.pad_index] * (decoder_maxlen - d["rnn_decoder_input_len"]))
        
        rnn_encoder_input = torch.tensor([d["rnn_encoder_input"] for d in batch], dtype=torch.long) # 类型转换
        rnn_decoder_input = torch.tensor([d["rnn_decoder_input"] for d in batch], dtype=torch.long)
        rnn_decoder_target = torch.tensor([d["rnn_decoder_target"] for d in batch], dtype=torch.long)
        
        return rnn_encoder_input, rnn_decoder_input, rnn_decoder_target


class GenerateRNNdataset_both(Dataset):
    def __init__(self, dat_path, vocab, antigen_max_len=23, cdr3_max_len=22):
        self.dat_path = dat_path
        self.antigen_max_len = antigen_max_len
        self.cdr3_max_len = cdr3_max_len
        self.vocab = vocab
        
        self.data = pd.read_csv(self.dat_path)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        # 获取数据并转化成id
        epitope_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.antigen[index]] + [self.vocab.eos_index]
        epitope_ids = epitope_ids[:self.antigen_max_len]
        padding = [self.vocab.pad_index for _ in range(self.antigen_max_len - len(epitope_ids))]
        epitope_ids.extend(padding)  # 满足our BERT模型的输入喜好，长度一致
        
        Bcdr3_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.betaCDR3[index]] + [self.vocab.eos_index]
        Bcdr3_ids = Bcdr3_ids[:self.cdr3_max_len]
        padding = [self.vocab.pad_index for _ in range(self.cdr3_max_len - len(Bcdr3_ids))]
        Bcdr3_ids.extend(padding)  # 满足our BERT模型的输入喜好，长度一致

        Acdr3_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.alphaCDR3[index]]
        
        rnn_encoder_input = epitope_ids + Bcdr3_ids
        rnn_decoder_input = Acdr3_ids
        rnn_decoder_target = rnn_decoder_input[1:] + [self.vocab.eos_index] 
        rnn_decoder_input_len = len(rnn_decoder_input)
        
        final_dict = {"rnn_encoder_input": rnn_encoder_input, "rnn_decoder_input": rnn_decoder_input, 
                      "rnn_decoder_input_len": rnn_decoder_input_len, 
                      "rnn_decoder_target": rnn_decoder_target}

        return final_dict
    
    def padding_batch(self, batch):
        rnn_decoder_input_lens = [d["rnn_decoder_input_len"] for d in batch] # batch的每个元素均为__getitem__获得
        decoder_maxlen = max(rnn_decoder_input_lens)
        
        # 补齐
        for d in batch: # 进行填充
            d["rnn_decoder_input"].extend([self.vocab.pad_index] * (decoder_maxlen - d["rnn_decoder_input_len"]))
            d["rnn_decoder_target"].extend([self.vocab.pad_index] * (decoder_maxlen - d["rnn_decoder_input_len"]))
        
        rnn_encoder_input = torch.tensor([d["rnn_encoder_input"] for d in batch], dtype=torch.long) # 类型转换
        rnn_decoder_input = torch.tensor([d["rnn_decoder_input"] for d in batch], dtype=torch.long)
        rnn_decoder_target = torch.tensor([d["rnn_decoder_target"] for d in batch], dtype=torch.long)
        
        return rnn_encoder_input, rnn_decoder_input, rnn_decoder_target


class GenerateRNNdataset_VJ(Dataset):
    def __init__(self, dat_path, vocab, gene_path, cdr3_max_len=23, is_alphaCDR3=False):
        self.dat_path = dat_path
        self.cdr3_max_len = cdr3_max_len
        self.vocab = vocab
        self.gene_path = gene_path
        self.is_alphaCDR3 = is_alphaCDR3
        
        df = pd.read_csv(self.dat_path)
        self.data = df
        self.data_genes = pd.read_csv(self.gene_path)["genes"].to_list()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        # 获取数据并转化成id
        if self.is_alphaCDR3:
            cdr3_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.alphaCDR3[index]] + [self.vocab.eos_index]
        else:
            cdr3_ids = [self.vocab.sos_index] + [self.vocab.stoi[aa] for aa in self.data.betaCDR3[index]] + [self.vocab.eos_index]
        cdr3_ids = cdr3_ids[:self.cdr3_max_len]
        padding = [self.vocab.pad_index for _ in range(self.cdr3_max_len - len(cdr3_ids))]
        cdr3_ids.extend(padding)  # 满足our BERT模型的输入喜好，长度一致
        
        if self.is_alphaCDR3:
            vj_ids = [self.vocab.sos_index] + [self.data_genes.index(self.data.alphaV[index])+5] + [self.data_genes.index(self.data.alphaJ[index])+5]
        else:
            vj_ids = [self.vocab.sos_index] + [self.data_genes.index(self.data.betaV[index])+5] + [self.data_genes.index(self.data.betaJ[index])+5]
        
        rnn_encoder_input = cdr3_ids
        rnn_decoder_input = vj_ids
        rnn_decoder_target = rnn_decoder_input[1:] + [self.vocab.eos_index] 
        
        final_dict = {"rnn_encoder_input": rnn_encoder_input, 
                      "rnn_decoder_input": rnn_decoder_input, "rnn_decoder_target": rnn_decoder_target}

        return [torch.tensor(value, dtype=torch.long) for _, value in final_dict.items()]
