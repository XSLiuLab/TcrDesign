import os
import sys
import shutil
import re
import tqdm
import random
import torch
import pickle
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
import pandas as pd
from TcrDesign.dataset import WordVocab
from TcrDesign.model import BERT
from TcrDesign.fine_tune import FineTuneDataset, BertForSequenceClassification_withAttn
from TcrDesign.fine_tune import BertForSequenceClassification_withAttn_returnEmbedding

import math
from functools import partial
from multiprocessing import Pool

healthy_tcrs_path = "TcrDesign/data/data_model/bert_pMHC_TCR_finetune/healthy_tcrs.csv"
all_tcrs_path = "TcrDesign/data/data_model/bert_pMHC_TCR_finetune/all_tcrs.csv"
bg_pMHC_path = "TcrDesign/data/data_model/bert_pMHC_TCR_finetune/bg_pMHC"

vocab_path = "TcrDesign/data/vocab_1mer.pkl"
gene_dat_path = "TcrDesign/data/data_model/bert_pMHC_TCR_finetune/VJ_vocab.csv"

MHC_psedo_path = "TcrDesign/data/mhc_pseudo/mhc_all.dat"

# 模型超参
tcr_max_len = 22
pmhc_max_len = 58
VJ_vocab_size = 197
VJ_hidden = 32

num_workers = 25

# BERT模型的超参
hidden = 256
layers = 6
attn_heads = 4
d_ff_fold = 4

# mask标准，如果缺失，需要多少mask字符
cdr3_mask_num = 15
antigen_mask_num = 21
MHC_mask_num = 34

max_processes = 32 # 最大并行进程数
chunksize = 5000 # 任务处理非常迅速，那么增加 chunksize 可以减少进程之间通信的次数，从而减少开销。
# 相反，如果每个任务的处理时间很长，较小的 chunksize 可以帮助更均匀地分配工作负载

# 模型权重路径
single_model_path_dir = "TcrDesign/weights/binding/single"
pan_model_path = "TcrDesign/weights/binding/pan/pMHC_TCR_finetune.ep"

# distance weight path
distance_weight_path = "TcrDesign/weights/binding/distance/distance_finetune.ep"

def rank_score_cal_single(index_row, bg_df):
    index, row = index_row
    antigen = row["antigen"]
    MHC = row["MHC"]
    score = row["binding_score"]
    bg_scores = bg_df.loc[(bg_df["antigen"] == antigen) & (bg_df["MHC"] == MHC), "binding_score"]
    rank_score = round(sum(score >= bg_scores) / len(bg_scores), 4)
    return (1 - rank_score) * 100


def rank_score_cal(pred_df, bg_df):
    print("Start Rank score Predicting... ...")
    partial_rank_score_cal_single = partial(rank_score_cal_single, bg_df=bg_df) # 固定一个参数
    processes = min(max_processes, math.ceil(len(pred_df) / chunksize)) # 动态调整processes
    with Pool(processes=processes) as pool:
        results = pool.imap(partial_rank_score_cal_single, pred_df.iterrows(), chunksize=chunksize)
        rank_score_list = list(results)
    pred_df["rank_score"] = rank_score_list
    return pred_df


def bg_score_cal(model, bg_num, unique_pMHC, vocab, batch_size, device, verbose):
    with open("TcrDesign/tmp/args.pkl", "rb") as f:
        args = pickle.load(f)
    
    new_order = ["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3"]
    healthy_tcrs_df = pd.read_csv(healthy_tcrs_path if args.healthy_bg else all_tcrs_path)
    healthy_tcrs_df = healthy_tcrs_df[new_order]
    healthy_tcrs_bg_df = healthy_tcrs_df.sample(n=bg_num, replace=False, ignore_index=True)
    
    antigen_list = []
    MHC_list = []
    for _, row in unique_pMHC.iterrows():
        antigen_list.extend([row["antigen"]] * bg_num)
        MHC_list.extend([row["MHC"]] * bg_num)
    
    bg_df = pd.concat([healthy_tcrs_bg_df] * len(unique_pMHC), ignore_index=True)
    bg_df["antigen"] = antigen_list
    bg_df["MHC"] = MHC_list
    
    print("Start Bg TCRs Binding Predicting... ...")
    bg_Dataset = FineTuneDataset(gene_dat_path=gene_dat_path, vocab=vocab, 
                                 tcr_max_len=tcr_max_len, pmhc_max_len=pmhc_max_len, include_label=False,
                                 is_pandas_df=True, pandas_df=bg_df, verbose=verbose)
    bg_data_loader = DataLoader(bg_Dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    _, bg_pred_score = bind_score_cal(model, bg_data_loader, device, include_label=False, verbose=verbose)
    bg_df["binding_score"] = bg_pred_score
    
    return bg_df


def bind_score_cal(model, data_loader, device, include_label, verbose):
    true_label = []
    pred_score = []
    for data in tqdm.tqdm(data_loader) if verbose else data_loader:
        data = {key: value.to(device) for key, value in data.items()}
        model_output = model.forward(data["aCDR3"], data["bCDR3"], data["pMHC"], data["VJ"], data["pMHC_segment_label"])
        pred_score.extend(torch.sigmoid(model_output.squeeze(dim=1)).cpu().detach().tolist())
        if include_label:
            true_label.extend(data["binding_score"].cpu().tolist())
    if include_label:
        true_label_array = np.array(true_label)
        pred_score_array = np.array(pred_score)
        fpr, tpr, _ = metrics.roc_curve(true_label_array, pred_score_array)
        precision, recall, _ = metrics.precision_recall_curve(true_label_array, pred_score_array)
        print("ROC-AUC: ", round(metrics.auc(fpr, tpr), 4))
        print("PR-AUC: ", round(metrics.auc(recall, precision), 4))
    return true_label, pred_score

def binding_predict(model_path, data_path, batch_size = 256, with_cuda = True, include_label=True,
                    is_cal_rank=False, seed=3407, bg_num=1000, verbose=True, add_order=False):
    if bg_num > 10000:
        print("The max number of Background TCRs is 10000, set bg_num to 10000!")
        bg_num = 10000
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    
    print("Loading Vocab", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))
    
    print("Loading Test Dataset", data_path)
    val_Dataset = FineTuneDataset(gene_dat_path, vocab, data_path, tcr_max_len, pmhc_max_len, include_label, verbose=verbose)
    val_data_loader = DataLoader(val_Dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False) # shuffle False保证了样本顺序
    
    print("Loading binding model for TCR-pMHC")
    Abert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    Bbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    pMHCbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=True) # has_next=True为了seg embedding
    model = BertForSequenceClassification_withAttn(Abert, Bbert, pMHCbert, pmhc_max_len, tcr_max_len, VJ_hidden, VJ_vocab_size,
                                                   attn_heads, hidden)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Start Binding Predicting... ...")
    if include_label:
        col_names = ["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3", "antigen", "MHC", "label"]
    else:
        col_names = ["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3", "antigen", "MHC"]
    if add_order:
        col_names += ["order"]
    
    pred_df = pd.read_csv(data_path, sep="\t", names=col_names)
    _, pred_score = bind_score_cal(model, val_data_loader, device, include_label, verbose)
    pred_df["binding_score"] = pred_score
    
    if is_cal_rank:
        unique_pMHC = pred_df[["antigen", "MHC"]].drop_duplicates()
        bg_df = bg_score_cal(model, bg_num, unique_pMHC, vocab, batch_size, device, verbose)
        pred_df = rank_score_cal(pred_df, bg_df)
        
    print("Calculation Down")
    
    return pred_df

# 如果缺失，采用X代替
def normalize_attn_matrix(attn_matrix):
    # Find the minimum and maximum distances
    min_dist = np.min(attn_matrix)
    max_dist = np.max(attn_matrix)
    # Perform the normalization
    normalized_matrix = (attn_matrix - min_dist) / (max_dist - min_dist)
    return normalized_matrix

def binding_predict_single(alphaV, alphaJ, alphaCDR3, betaV, betaJ, betaCDR3, antigen, MHC,
                           with_cuda = True, is_cal_rank=False, seed=3407, bg_num=1000, batch_size=256):
    if bg_num > 10000:
        print("The max number of Background TCRs is 10000, set bg_num to 10000!")
        bg_num = 10000
    
    with open("TcrDesign/tmp/args.pkl", "rb") as f:
        args = pickle.load(f)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 决定模型权重
    exites_epitope_models = os.listdir(single_model_path_dir)
    exites_epitopes = [single.split('.')[0].split('_')[0] for single in os.listdir(single_model_path_dir)]
    if antigen in exites_epitopes:
        print("Epitope Model Exists!")
        matched_model = [epitope_model for epitope_model in exites_epitope_models if re.search(antigen, epitope_model)][0]
        model_path = f'{single_model_path_dir}/{matched_model}'
    else:
        print("Epitope Model Not Exists! Use the pan model!")
        model_path = pan_model_path
    
    MHC_mapping_df = pd.read_csv(MHC_psedo_path, sep="\s+", names=["MHC", "MHC_psedo"])
    gene_mapping_df = pd.read_csv(gene_dat_path)
    
    if not MHC.startswith("X"):
        MHC = MHC_mapping_df.loc[MHC_mapping_df["MHC"]==MHC, "MHC_psedo"].values[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    
    vocab = WordVocab.load_vocab(vocab_path)
    
    print("Loading binding model for TCR-pMHC")
    Abert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    Bbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    pMHCbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=True) # has_next=True为了seg embedding
    model = BertForSequenceClassification_withAttn(Abert, Bbert, pMHCbert, pmhc_max_len, tcr_max_len, VJ_hidden, VJ_vocab_size,
                                                   attn_heads, hidden)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 准备数据
    VJ = [gene_mapping_df.loc[gene_mapping_df["gene_name"] == gene, "index"].values[0] for gene in [alphaV, alphaJ, betaV, betaJ]]
    
    aCDR3 = [vocab.mask_index]*cdr3_mask_num if alphaCDR3.startswith("X") else \
            [vocab.stoi.get(token, vocab.unk_index) for token in alphaCDR3]
    aCDR3 = [vocab.sos_index] + aCDR3 + [vocab.eos_index]
    aCDR3 = aCDR3[:tcr_max_len]
    padding = [vocab.pad_index for _ in range(tcr_max_len - len(aCDR3))]
    aCDR3.extend(padding)
    
    bCDR3 = [vocab.mask_index]*cdr3_mask_num if betaCDR3.startswith("X") else \
            [vocab.stoi.get(token, vocab.unk_index) for token in betaCDR3]
    bCDR3 = [vocab.sos_index] + bCDR3 + [vocab.eos_index]
    bCDR3 = bCDR3[:tcr_max_len]
    padding = [vocab.pad_index for _ in range(tcr_max_len - len(bCDR3))]
    bCDR3.extend(padding)
    
    pep = [vocab.mask_index]*antigen_mask_num if antigen.startswith("X") else \
        [vocab.stoi.get(token, vocab.unk_index) for token in antigen]
    mhc = [vocab.mask_index]*MHC_mask_num if MHC.startswith("X") else \
        [vocab.stoi.get(token, vocab.unk_index) for token in MHC]
    mhc = [vocab.sos_index] + mhc + [vocab.eos_index]
    pep = pep + [vocab.eos_index]
    segment_label = ([1 for _ in range(len(mhc))] + [2 for _ in range(len(pep))])[:pmhc_max_len]
    pmhc = (mhc + pep)[:pmhc_max_len]
    padding = [vocab.pad_index for _ in range(pmhc_max_len - len(pmhc))]
    pmhc.extend(padding), segment_label.extend(padding)
    
    # 转成tensor格式，加上batch维度
    VJ = torch.tensor(VJ, dtype=torch.long).unsqueeze(0)
    aCDR3 = torch.tensor(aCDR3, dtype=torch.long).unsqueeze(0)
    bCDR3 = torch.tensor(bCDR3, dtype=torch.long).unsqueeze(0)
    pmhc = torch.tensor(pmhc, dtype=torch.long).unsqueeze(0)
    segment_label = torch.tensor(segment_label, dtype=torch.long).unsqueeze(0)
    
    print("Predicting... ...")
    model_output = model.forward(aCDR3.to(device), bCDR3.to(device), pmhc.to(device), VJ.to(device), segment_label.to(device))
    predict_binding_score = torch.sigmoid(model_output.squeeze()).cpu().detach().tolist()
    
    predict_rank_score = 1
    if is_cal_rank:
        new_order = ["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3"]
        healthy_tcrs_df = pd.read_csv(healthy_tcrs_path if args.healthy_bg else all_tcrs_path)
        healthy_tcrs_df = healthy_tcrs_df[new_order]
        healthy_tcrs_bg_df = healthy_tcrs_df.sample(n=bg_num, replace=False, ignore_index=True)
        
        healthy_tcrs_bg_df["antigen"] = "X"*antigen_mask_num if antigen.startswith("X") else antigen
        healthy_tcrs_bg_df["MHC"] = "X"*MHC_mask_num if MHC.startswith("X") else MHC
        
        bg_Dataset = FineTuneDataset(gene_dat_path=gene_dat_path, vocab=vocab, 
                                     tcr_max_len=tcr_max_len, pmhc_max_len=pmhc_max_len, include_label=False,
                                     is_pandas_df=True, pandas_df=healthy_tcrs_bg_df)
        bg_data_loader = DataLoader(bg_Dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        _, bg_pred_score = bind_score_cal(model, bg_data_loader, device, include_label=False, verbose=True)
        
        predict_rank_score = round(float(np.sum(np.array(predict_binding_score) >= np.array(bg_pred_score))/len(bg_pred_score)), 4)
    
    # 计算distance矩阵
    model.load_state_dict(torch.load(distance_weight_path, map_location=device))
    model.to(device)
    model.eval()
    _ = model.forward(aCDR3.to(device), bCDR3.to(device), pmhc.to(device), VJ.to(device), segment_label.to(device))
    attn_weight = model.attention.attention.attn_weight # 收集attention权重
    attn_weight = attn_weight.squeeze().mean(0)[1:len(betaCDR3)+1, 36:36+len(antigen)].cpu().detach().numpy()
    
    return predict_binding_score, (1-predict_rank_score)*100, normalize_attn_matrix(attn_weight)


def binding_predict_pmhc_specific(data_path, batch_size = 256, with_cuda = True, include_label=True,
                                  is_cal_rank=False, seed=3407, bg_num=1000):
    exites_epitope_models = os.listdir(single_model_path_dir)
    exites_epitopes = [single.split('.')[0].split('_')[0] for single in os.listdir(single_model_path_dir)]
    
    directory = os.path.dirname(data_path)
    os.makedirs(f'{directory}/tmp', exist_ok=True)
    
    if include_label:
        dat_df = pd.read_csv(data_path, sep="\t", names=["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3", "antigen", "MHC", "label"])
    else:
        dat_df = pd.read_csv(data_path, sep="\t", names=["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3", "antigen", "MHC"])
    dat_df["order"] = list(range(len(dat_df)))
    
    epitopes = dat_df['antigen'].unique().tolist()
    need_split_epitopes = list(set(epitopes) & set(exites_epitopes)) # 获取相交元素，需要单独预测的epitope
    need_pan_epitopes = list(set(epitopes) - set(need_split_epitopes)) # 获取差集，pan预测的epitope
    
    for epitope in need_split_epitopes:
        tmp_df = dat_df[dat_df['antigen'] == epitope]
        tmp_df.to_csv(f'{directory}/tmp/{epitope}.tsv', sep="\t", index=False, header=False)
        
    pan_mask = dat_df['antigen'].apply(lambda x: x in need_pan_epitopes)
    pan_df = dat_df[pan_mask]
    pan_df.to_csv(f'{directory}/tmp/pan.tsv', sep="\t", index=False, header=False)
    
    if is_cal_rank:
        col_names = dat_df.columns.tolist() + ["binding_score", "rank_score", "predict_type"]
    else:
        col_names = dat_df.columns.tolist() + ["binding_score", "predict_type"]
    pred_df = pd.DataFrame(columns=col_names)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if len(need_split_epitopes) != 0:
        print("start predict for epitope-specific models")
    for epitope in tqdm.tqdm(need_split_epitopes):
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            
            matched_model = [epitope_model for epitope_model in exites_epitope_models if re.search(epitope, epitope_model)][0]
            model_path = f'{single_model_path_dir}/{matched_model}'
            single_data_path = f'{directory}/tmp/{epitope}.tsv'
            single_pred_df = binding_predict(model_path, single_data_path, batch_size=batch_size, with_cuda=with_cuda, include_label=include_label,
                                             is_cal_rank=is_cal_rank, seed=seed, bg_num=bg_num, verbose=False, add_order=True)
            single_pred_df["predict_type"] = "epitope-specific"
            pred_df = pd.concat([pred_df, single_pred_df], axis=0, ignore_index=True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    if len(pan_df) != 0:
        print("start predict for pan model")
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            pan_pred_df = binding_predict(pan_model_path, f'{directory}/tmp/pan.tsv', batch_size=batch_size, with_cuda=with_cuda, include_label=include_label,
                                          is_cal_rank=is_cal_rank, seed=seed, bg_num=bg_num, verbose=False, add_order=True)
            pan_pred_df["predict_type"] = "pan"
            pred_df = pd.concat([pred_df, pan_pred_df], axis=0, ignore_index=True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    pred_df = pred_df.sort_values(by=['order']).reset_index(drop=True)
    pred_df.drop(columns=['order'], inplace=True)
    
    if include_label:
        pred_df["label"] = pred_df["label"].astype(int)
        fpr, tpr, _ = metrics.roc_curve(np.array(pred_df["label"].astype(int)), np.array(pred_df["binding_score"]))
        precision, recall, _ = metrics.precision_recall_curve(np.array(pred_df["label"].astype(int)), np.array(pred_df["binding_score"]))
        print("ROC-AUC: ", round(metrics.auc(fpr, tpr), 4))
        print("PR-AUC: ", round(metrics.auc(recall, precision), 4))
    
    # 删除tmp
    delete_folder(f'{directory}/tmp')
    
    return pred_df


def delete_folder(folder_path):
    try:
        # 检查目录是否存在且为目录
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
        else:
            print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")


def embedding_cal(model_path, data_path, batch_size = 256, with_cuda = True):
    device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")
    vocab = WordVocab.load_vocab(vocab_path)
    
    val_Dataset = FineTuneDataset(gene_dat_path, vocab, data_path, tcr_max_len, pmhc_max_len, include_label=False, verbose=False)
    val_data_loader = DataLoader(val_Dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False) # shuffle False保证了样本顺序
    
    Abert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    Bbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=False)
    pMHCbert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, d_ff_fold=d_ff_fold, has_next=True) # has_next=True为了seg embedding
    model = BertForSequenceClassification_withAttn_returnEmbedding(Abert, Bbert, pMHCbert, pmhc_max_len, 
                                                                   tcr_max_len, VJ_hidden, VJ_vocab_size,
                                                                   attn_heads, hidden)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    col_names = ["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3", "antigen", "MHC"]
    col_names += ["order"] # order是为了保证样本顺序
    pred_df = pd.read_csv(data_path, sep="\t", names=col_names)
    
    aCDR3_embed_list = []
    bCDR3_embed_list = []
    pMHC_embed_list = []
    for data in val_data_loader:
        data = {key: value.to(device) for key, value in data.items()}
        _, aCDR3_feature, bCDR3_feature, pmhc_feature = model.forward(data["aCDR3"], data["bCDR3"], data["pMHC"], data["VJ"], data["pMHC_segment_label"])
        aCDR3_embed_list.append(aCDR3_feature.mean(dim=1).squeeze().cpu().detach())
        bCDR3_embed_list.append(bCDR3_feature.mean(dim=1).squeeze().cpu().detach())
        pMHC_embed_list.append(pmhc_feature.mean(dim=1).squeeze().cpu().detach())
    
    aCDR3_embed = torch.cat(aCDR3_embed_list, dim=0)
    bCDR3_embed = torch.cat(bCDR3_embed_list, dim=0)
    tcr_embed = (aCDR3_embed + bCDR3_embed) / 2
    pMHC_embed = torch.cat(pMHC_embed_list, dim=0)
    
    pred_df["aCDR3_embed"] = aCDR3_embed.tolist()
    pred_df["bCDR3_embed"] = bCDR3_embed.tolist()
    pred_df["tcr_embed"] = tcr_embed.tolist()
    pred_df["pMHC_embed"] = pMHC_embed.tolist()

    return pred_df


def embedding_cal_pmhc_specific(data_path, batch_size = 256, with_cuda = True):
    exites_epitope_models = os.listdir(single_model_path_dir)
    exites_epitopes = [single.split('.')[0].split('_')[0] for single in os.listdir(single_model_path_dir)]
    
    directory = os.path.dirname(data_path)
    os.makedirs(f'{directory}/tmp', exist_ok=True)
    
    dat_df = pd.read_csv(data_path, sep="\t", names=["alphaV", "alphaJ", "alphaCDR3", "betaV", "betaJ", "betaCDR3", "antigen", "MHC"])
    dat_df["order"] = list(range(len(dat_df)))
    
    epitopes = dat_df['antigen'].unique().tolist()
    need_split_epitopes = list(set(epitopes) & set(exites_epitopes)) # 获取相交元素，需要单独预测的epitope
    need_pan_epitopes = list(set(epitopes) - set(need_split_epitopes)) # 获取差集，pan预测的epitope
    
    for epitope in need_split_epitopes:
        tmp_df = dat_df[dat_df['antigen'] == epitope]
        tmp_df.to_csv(f'{directory}/tmp/{epitope}.tsv', sep="\t", index=False, header=False)
        
    pan_mask = dat_df['antigen'].apply(lambda x: x in need_pan_epitopes)
    pan_df = dat_df[pan_mask]
    pan_df.to_csv(f'{directory}/tmp/pan.tsv', sep="\t", index=False, header=False)
    
    col_names = dat_df.columns.tolist() + ["aCDR3_embed", "bCDR3_embed", "tcr_embed", "pMHC_embed", "predict_type"]
    pred_df = pd.DataFrame(columns=col_names)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if len(need_split_epitopes) != 0:
        print("start predict for epitope-specific models")
    for epitope in tqdm.tqdm(need_split_epitopes):
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            
            matched_model = [epitope_model for epitope_model in exites_epitope_models if re.search(epitope, epitope_model)][0]
            model_path = f'{single_model_path_dir}/{matched_model}'
            single_data_path = f'{directory}/tmp/{epitope}.tsv'
            single_pred_df = embedding_cal(model_path, single_data_path, batch_size=batch_size, with_cuda=with_cuda)
            single_pred_df["predict_type"] = "epitope-specific"
            pred_df = pd.concat([pred_df, single_pred_df], axis=0, ignore_index=True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    if len(pan_df) != 0:
        print("start predict for pan model")
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            pan_pred_df = embedding_cal(pan_model_path, f'{directory}/tmp/pan.tsv', batch_size=batch_size, with_cuda=with_cuda)
            pan_pred_df["predict_type"] = "pan"
            pred_df = pd.concat([pred_df, pan_pred_df], axis=0, ignore_index=True)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    pred_df = pred_df.sort_values(by=['order']).reset_index(drop=True)
    pred_df.drop(columns=['order'], inplace=True)
    
    # 删除tmp
    delete_folder(f'{directory}/tmp')
    
    return pred_df
