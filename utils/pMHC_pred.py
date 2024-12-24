import os
import random
import numpy as np
import pandas as pd

import tqdm
import torch
from torch.utils.data import DataLoader

from TcrDesign.model.bert import BERT 
from TcrDesign.model.language_model import BERTLM_pMHC
from TcrDesign.dataset import BERTDataset_pMHC, WordVocab
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

import math
from functools import partial
from multiprocessing import Pool

# 定义参数值
vocab_path = "TcrDesign/data/vocab_1mer.pkl"
pre_weight_path = "TcrDesign/weights/pMHC/bert_pMHC_pretrain.pt"
natural_I_path = "TcrDesign/data/I_natural_peptide.csv"
natural_II_path = "TcrDesign/data/II_natural_peptide.csv"
tmp_path = "TcrDesign/tmp"

MHC_psedo_path = "TcrDesign/data/mhc_pseudo/mhc_all.dat"

mask_freq = 0.0
seq_len = 58
corpus_lines = None
on_memory = True
num_workers = 30

hidden = 256
layers = 6
attn_heads = 4
d_ff_fold = 4

max_processes = 32 # 最大并行进程数
chunksize = 5000 # 任务处理非常迅速，那么增加 chunksize 可以减少进程之间通信的次数，从而减少开销。
# 相反，如果每个任务的处理时间很长，较小的 chunksize 可以帮助更均匀地分配工作负载


def rank_score_cal_single(index_row, bg_df):
    index, row = index_row
    MHC = row["mhc"]
    score = row["pred_score"]
    bg_scores = bg_df.loc[bg_df["mhc"] == MHC, "binding_score"]
    rank_score = round(sum(score >= bg_scores) / len(bg_scores), 4)
    return rank_score


def rank_score_cal(pred_df, bg_df):
    print("Start Rank score Predicting... ...")
    partial_rank_score_cal_single = partial(rank_score_cal_single, bg_df=bg_df) # 固定一个参数
    processes = min(max_processes, math.ceil(len(pred_df) / chunksize)) # 动态调整processes
    with Pool(processes=processes) as pool:
        results = pool.imap(partial_rank_score_cal_single, pred_df.iterrows(), chunksize=chunksize)
        rank_score_list = list(results)
    pred_df["rank_score"] = rank_score_list
    return pred_df


def bg_score_cal(model, bg_num, unique_MHC, vocab, batch_size, device, mhc_class):
    if mhc_class == "I":
        natural_peps = pd.read_csv(natural_I_path, header=0, names=["peptide"])
    else:
        natural_peps = pd.read_csv(natural_II_path, header=0, names=["peptide"])
    natural_peps = natural_peps.sample(n=bg_num, replace=False, ignore_index=True)
    
    mhc_list = []
    for mhc in unique_MHC:
        mhc_list.extend([mhc] * bg_num)
    
    bg_df = pd.concat([natural_peps] * len(unique_MHC), ignore_index=True)
    bg_df["mhc"] = mhc_list
    bg_df = bg_df[["mhc", "peptide"]]
    bg_df.to_csv(os.path.join(tmp_path, "bg_natural_peptide_mhc.csv"), index=False, sep="\t", header=None)
    
    print("Start Bg Natural Peptide Binding Predicting... ...")
    bg_Dataset = BERTDataset_pMHC(os.path.join(tmp_path, "bg_natural_peptide_mhc.csv"), 
                                  vocab, seq_len=seq_len, mask_freq=mask_freq, corpus_lines=corpus_lines, on_memory=on_memory, include_label=False)
    bg_data_loader = DataLoader(bg_Dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    _, bg_pred_score = bind_score_cal(model, bg_data_loader, device, include_label=False, thre=None)
    bg_df["binding_score"] = bg_pred_score
    
    # delete the tmp file
    os.remove(os.path.join(tmp_path, "bg_natural_peptide_mhc.csv"))
    
    return bg_df


def bind_score_cal(model, val_data_loader, device, include_label, thre):
    pred_scores = []
    true_scores = []
    for data in tqdm.tqdm(val_data_loader):
        data = {key: value.to(device) for key, value in data.items()}
        binding_score_output, mask_lm_output = model.forward(data["bert_input"], data["segment_label"])
        mask_pred = mask_lm_output.argmax(dim=-1)
        pred_scores.extend(torch.sigmoid(binding_score_output.squeeze()).cpu().detach().tolist())
        if include_label:
            true_scores.extend(data["binding_score"].cpu().detach().tolist())
    if include_label:
        true_label_array = np.array(true_scores)
        pred_score_array = np.array(pred_scores)
        print(" AUC: ", roc_auc_score(true_label_array, pred_score_array), '\n',
              "Acc: ", accuracy_score(true_label_array, (pred_score_array > thre).astype(int)), '\n',
              "Precision: ", precision_score(true_label_array, (pred_score_array > thre).astype(int)), '\n',
              "Recall: ", recall_score(true_label_array, (pred_score_array > thre).astype(int)), '\n',
              "F1: ", f1_score(true_label_array, (pred_score_array > thre).astype(int)))
    return true_scores, pred_scores


def pMHC_binding_predict(data_path, batch_size = 256, with_cuda = True, include_label=True, thre=0.9,
                         is_cal_rank=False, seed=3407, bg_num=1000, mhc_class="I"):
    if bg_num > 10000:
        print("The max number of Background Natural Peps is 10000, set bg_num to 10000!")
        bg_num = 10000
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    
    print("Loading Vocab", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))
    
    print("Loading Data", data_path)
    val_dataset = BERTDataset_pMHC(data_path, vocab, seq_len=seq_len, mask_freq=mask_freq, corpus_lines=corpus_lines, on_memory=on_memory, include_label=include_label)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=True) # has_next=True为了seg embedding
    model = BERTLM_pMHC(bert, len(vocab))
    model.load_state_dict(torch.load(pre_weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Start Binding Predicting... ...")
    if include_label:
        binding_df = pd.read_csv(data_path, sep="\t", header=None, names=["mhc", "peptide", "label"])
    else:
        binding_df = pd.read_csv(data_path, sep="\t", header=None, names=["mhc", "peptide"])
    _, pred_scores = bind_score_cal(model, val_data_loader, device, include_label, thre)
    binding_df['pred_score'] = pred_scores
    
    if is_cal_rank:
        unique_MHC = binding_df['mhc'].unique()
        bg_df = bg_score_cal(model, bg_num, unique_MHC, vocab, batch_size, device, mhc_class)
        binding_df = rank_score_cal(binding_df, bg_df)
    
    print("\nCalculation Down")
    
    return binding_df


def pMHC_binding_predict_single(peptide, MHC, with_cuda = True, 
                                is_cal_rank=False, seed=3407, bg_num=1000, mhc_class="I"):
    if bg_num > 10000:
        print("The max number of Background Natural Peps is 10000, set bg_num to 10000!")
        bg_num = 10000
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    
    print("Loading Vocab", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))
    
    MHC_mapping_df = pd.read_csv(MHC_psedo_path, sep="\s+", names=["MHC", "MHC_psedo"])
    MHC = MHC_mapping_df.loc[MHC_mapping_df["MHC"]==MHC, "MHC_psedo"].values[0]
    
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=True) # has_next=True为了seg embedding
    model = BERTLM_pMHC(bert, len(vocab))
    model.load_state_dict(torch.load(pre_weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Start Binding Predicting... ...")
    mhc = [vocab.sos_index] + [vocab.stoi.get(token, vocab.unk_index) for token in MHC] + [vocab.eos_index]
    pep = [vocab.stoi.get(token, vocab.unk_index) for token in peptide] + [vocab.eos_index]
    segment_label = ([1 for _ in range(len(mhc))] + [2 for _ in range(len(pep))])[:seq_len]
    pmhc = (mhc + pep)[:seq_len]
    padding = [vocab.pad_index for _ in range(seq_len - len(pmhc))]
    pmhc.extend(padding), segment_label.extend(padding)
    
    # 转成tensor格式，加上batch维度
    pmhc = torch.tensor(pmhc, dtype=torch.long).unsqueeze(0)
    segment_label = torch.tensor(segment_label, dtype=torch.long).unsqueeze(0)
    
    binding_score_output, _ = model.forward(pmhc.to(device), segment_label.to(device))
    predict_binding_score = torch.sigmoid(binding_score_output.squeeze()).cpu().detach().tolist()
    
    predict_rank_score = 0
    if is_cal_rank:
        if mhc_class == "I":
            natural_peps = pd.read_csv(natural_I_path, header=0, names=["peptide"])
        else:
            natural_peps = pd.read_csv(natural_II_path, header=0, names=["peptide"])
        natural_peps = natural_peps.sample(n=bg_num, replace=False, ignore_index=True)
        
        natural_peps["mhc"] = MHC
        natural_peps = natural_peps[["mhc", "peptide"]]
        natural_peps.to_csv(os.path.join(tmp_path, "bg_natural_peptide_mhc.csv"), index=False, sep="\t", header=None)
        
        bg_Dataset = BERTDataset_pMHC(os.path.join(tmp_path, "bg_natural_peptide_mhc.csv"), 
                                      vocab, seq_len=seq_len, mask_freq=mask_freq, corpus_lines=corpus_lines, on_memory=on_memory, include_label=False)
        bg_data_loader = DataLoader(bg_Dataset, batch_size=256, num_workers=num_workers, shuffle=False)
        _, bg_pred_score = bind_score_cal(model, bg_data_loader, device, include_label=False, thre=None)
        # delete the tmp file
        os.remove(os.path.join(tmp_path, "bg_natural_peptide_mhc.csv"))
        
        predict_rank_score = round(float(np.sum(np.array(predict_binding_score) >= np.array(bg_pred_score))/len(bg_pred_score)), 4)
    
    return predict_binding_score, predict_rank_score
