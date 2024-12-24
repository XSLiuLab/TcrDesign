import os
import sys
import math
import fileinput
import pandas as pd
from collections import OrderedDict

from TcrDesign.utils.pMHC_TCR_pred_attn import binding_predict_pmhc_specific

from TcrDesign.utils import pMHC_binding_predict_single
from TcrDesign.utils import generate_betaTCRs, generate_alphaTCRs
from TcrDesign.utils import generate_VJ

import tqdm
import subprocess as sp
from functools import partial
import Levenshtein
import editdistance
from multiprocessing import Pool

tmp_path = "TcrDesign/tmp"
MHC_psedo_path = "TcrDesign/data/mhc_pseudo/mhc_all.dat"

MHC_mask_num = 34

def contains_only_standard_amino_acids(sequence):
    standard_amino_acids = {'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'}
    return set(sequence).issubset(standard_amino_acids)


def generate_betaTCRs_and_processed(antigen, beta_cdr3_num, beta_cdr3_maxLen, with_cuda, beta_temperature):
    generate_betaCDR3s = generate_betaTCRs(antigen, beta_cdr3_num, beta_cdr3_maxLen, with_cuda, beta_temperature)
    # 剔除生成失败 & 重复的TCR
    for i, tcr in enumerate(generate_betaCDR3s):
        if not contains_only_standard_amino_acids(tcr):
            del generate_betaCDR3s[i]
    generate_betaCDR3s = list(OrderedDict.fromkeys(generate_betaCDR3s)) # 进行去重
    return generate_betaCDR3s


def generate_alphaTCRs_and_processed(antigen, beta_cdr3, alpha_cdr3_num, alpha_cdr3_maxLen, with_cuda, alpha_temperature):
    generate_alphaCDR3s = generate_alphaTCRs(antigen, beta_cdr3, alpha_cdr3_num, alpha_cdr3_maxLen, with_cuda, alpha_temperature)
    for j, tcr in enumerate(generate_alphaCDR3s):
        if not contains_only_standard_amino_acids(tcr):
            del generate_alphaCDR3s[j]
    generate_alphaCDR3s = list(OrderedDict.fromkeys(generate_alphaCDR3s)) # 进行去重
    return generate_alphaCDR3s


def calculate_distance(tuple_pair, threshold):
    str1, str2 = tuple_pair
    # distance = Levenshtein.distance(str1, str2)
    distance = editdistance.eval(str1, str2)
    if distance <= threshold:
        return (str1, str2, distance)
    else:
        return None

def finder_TCRs(TCR1, TCR2, threshold=1, processes=4, chunksize=10000):
    # 创建一个包含所有可能的字符串对的生成器
    # 注意：这一步并不会实际创建列表，它会在计算时产生元素
    pairs = ((str1, str2) for str1 in TCR1 for str2 in TCR2)
    partial_calculate_distance = partial(calculate_distance, threshold=threshold) # 固定一个参数
    # 多进程处理，加快运算速度
    with Pool(processes=processes) as pool:
        results = pool.imap(partial_calculate_distance, pairs, chunksize=chunksize)
        filtered_results = filter(None, results)
        results_list = list(filtered_results)
    return results_list


def get_whole_coding_sequence(v, j, cdr3):
    try:
        out = sp.run(['stitchr', '-v', v, '-j', j, '-cdr3', cdr3, '-sw'], stdout=sp.PIPE)
    except ValueError:
        print('Notice: prediction error, please check double', out)
    nt = ''
    aa = ''
    name = ''
    for line in out.stdout.decode('utf-8').split('\n'):
        if line.startswith('>nt'):
            name = line[5:]
        elif line.startswith('>aa'):
            continue
        elif set(line).issubset(set('ACGT')):
            nt += line
        elif set(line).issubset(set('ACDEFGHIKLMNPQRSTVWY')):
            aa += line
        else:
            pass
    return nt, aa, name


def correct_generate_error(df):
    mask1 = df.alphaJ == "TRAJ33"
    mask2 = df.alphaCDR3.str.endswith("IF")
    df.alphaCDR3.loc[mask1 & mask2] = df.alphaCDR3.loc[mask1 & mask2].str[:-1] + 'W'
    return df


def add_whole_coding_sequence(df):
    alpha_nt_list, alpha_aa_list, alpha_name_list = [], [], []
    beta_nt_list, beta_aa_list, beta_name_list = [], [], []
    for _, row in df.iterrows():
        Ant, Aaa, Aname = get_whole_coding_sequence(row['alphaV'], row['alphaJ'], row['alphaCDR3'])
        Bnt, Baa, Bname = get_whole_coding_sequence(row['betaV'], row['betaJ'], row['betaCDR3'])
        alpha_nt_list.append(Ant), alpha_aa_list.append(Aaa), alpha_name_list.append(Aname)
        beta_nt_list.append(Bnt), beta_aa_list.append(Baa), beta_name_list.append(Bname)
    df["alphaNT"] = alpha_nt_list
    df["alphaAA"] = alpha_aa_list
    df["alphaName"] = alpha_name_list
    df["betaNT"] = beta_nt_list
    df["betaAA"] = beta_aa_list
    df["betaName"] = beta_name_list
    return df


# MHC可以缺失
def generate_TCRs_for_one_antigen(antigen, MHC, with_cuda=True, mhc_class="I", out_path='./', is_cal_coding_seq=True,
                                  beta_cdr3_num=100, alpha_cdr3_num=100, beta_temperature=1, alpha_temperature=0.8,
                                  beta_cdr3_maxLen=20, alpha_cdr3_maxLen=20,
                                  beta_vj_num=5, alpha_vj_num=5, 
                                  rank_thre=100, sort=True, binding_batchSize=256, seed=3407, bg_num=1000, max_num_one_file=500000):
    pMHC_binding_score, pMHC_rank_score = pMHC_binding_predict_single(antigen, MHC, with_cuda=with_cuda,
                                                                      is_cal_rank=True, seed=seed, bg_num=bg_num, mhc_class=mhc_class)
    print(f'pMHC binding_score is {pMHC_binding_score}, rank score (%) is {round(1 - pMHC_rank_score, 4)*100}')
    if pMHC_rank_score >= 0.9:
        print("pMHC rank score is less than 10, the binding is *reliable*, please double check with netMHCpan")
    else:
        print("pMHC rank score is greater than 10, the binding is *not reliable*, please double check with netMHCpan")
    print("=="*30)
    
    pMHC = f'{antigen}_{MHC}'
    
    MHC_mapping_df = pd.read_csv(MHC_psedo_path, sep="\s+", names=["MHC", "MHC_psedo"])
    if not MHC.startswith("X"):
        MHC = MHC_mapping_df.loc[MHC_mapping_df["MHC"]==MHC, "MHC_psedo"].values[0]
    else:
        MHC = "X" * MHC_mask_num
        
    original_stdout = sys.stdout # 保存原始标准输出 
    
    print("Stage 1: Generating beta TCRs... ...")
    sys.stdout = open(os.devnull, 'w') # 重定向
    generate_betaCDR3s = generate_betaTCRs_and_processed(antigen, beta_cdr3_num, beta_cdr3_maxLen, with_cuda, beta_temperature)
    generate_beta_VJs = generate_VJ(generate_betaCDR3s, generate_num=beta_vj_num, with_cuda=with_cuda, is_alpha=False, batch_mode=True)
    generate_beta_VJs = generate_beta_VJs[::beta_vj_num] # 每个TCR生成beta_vj_num个，取第一个
    sys.stdout.close()  # 关闭空设备的文件描述符
    sys.stdout = original_stdout # 重定向
    print("=="*15)
    
    print("Stage 2: Generating alpha TCRs... ...")
    for i, beta_cdr3 in tqdm.tqdm(enumerate(generate_betaCDR3s), total=len(generate_betaCDR3s), bar_format="{l_bar}{r_bar}"):
        sys.stdout = open(os.devnull, 'w') # 重定向
        generate_alphaCDR3s = generate_alphaTCRs_and_processed(antigen, beta_cdr3, alpha_cdr3_num, alpha_cdr3_maxLen, with_cuda, alpha_temperature)
        generate_alpha_VJs = generate_VJ(generate_alphaCDR3s, generate_num=alpha_vj_num, with_cuda=with_cuda, is_alpha=True, batch_mode=True)
        generate_alpha_VJs = generate_alpha_VJs[::alpha_vj_num]
        # 保存文件，后续调用
        tmp_df = pd.DataFrame({'alphaV': [tmp[0] for tmp in generate_alpha_VJs],
                               'alphaJ': [tmp[1] for tmp in generate_alpha_VJs],
                               'alphaCDR3': generate_alphaCDR3s,
                               'betaV': generate_beta_VJs[i][0],
                               'betaJ': generate_beta_VJs[i][1],
                               'betaCDR3': beta_cdr3,
                               'antigen': antigen,
                               'MHC': MHC})
        tmp_df = correct_generate_error(tmp_df)
        tmp_df.to_csv(os.path.join(tmp_path, f'{beta_cdr3}.tsv'), index=False, sep="\t", header=None)
        sys.stdout.close()  # 关闭空设备的文件描述符
        sys.stdout = original_stdout # 重定向
    print("=="*15)
    
    print("Stage3: Binding Filtering")
    # one file contains most ? TCRs
    contain_file_num = math.floor(max_num_one_file / alpha_cdr3_num)
    large_file_num = math.ceil(len(generate_betaCDR3s) / contain_file_num)
    for i in range(large_file_num):
        filenames = [os.path.join(tmp_path, f'{tcr}.tsv') for tcr in generate_betaCDR3s[i*contain_file_num:(i+1)*contain_file_num]]
        with open(os.path.join(tmp_path, f'large_file_{i}.tsv'), 'w') as outfile:
            for line in fileinput.input(filenames):
                outfile.write(line)
    # Delete tmp files
    for beta_cdr3 in generate_betaCDR3s:
        os.remove(os.path.join(tmp_path, f'{beta_cdr3}.tsv'))
    
    pred_df_list = []
    for i in tqdm.tqdm(range(large_file_num), total=len(range(large_file_num)), bar_format="{l_bar}{r_bar}"):
        # sys.stdout = open(os.devnull, 'w') # 重定向
        tmp_data_path = os.path.join(tmp_path, f'large_file_{i}.tsv')
        tmp_pred_df = binding_predict_pmhc_specific(tmp_data_path, batch_size=binding_batchSize,
                                                    with_cuda=with_cuda, include_label=False, is_cal_rank=True, seed=seed, bg_num=bg_num)
        tmp_pred_df = tmp_pred_df.loc[tmp_pred_df.rank_score<=rank_thre, :]
        pred_df_list.append(tmp_pred_df)
        # sys.stdout.close()  # 关闭空设备的文件描述符
        # sys.stdout = original_stdout # 重定向
    pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True)
    if sort:
        pred_df = pred_df.sort_values(by='binding_score', ascending=False, ignore_index=True)
    # CC开头的删除
    pred_df = pred_df.loc[~pred_df.alphaCDR3.str.startswith("CC") & ~pred_df.betaCDR3.str.startswith("CC"), :]
    pred_df = pred_df.reset_index(drop=True)
    
    # print(f'Generate {beta_cdr3_num*alpha_cdr3_num} TCRs, having {round(len(pred_df)/(beta_cdr3_num*alpha_cdr3_num), 4)*100}% binding TCR for the provided pMHC')
    
    # Delete tmp files
    for i in range(large_file_num):
        os.remove(os.path.join(tmp_path, f'large_file_{i}.tsv'))
    
    if is_cal_coding_seq:
        pred_df = add_whole_coding_sequence(pred_df)
    
    # Save
    pred_df.to_csv(os.path.join(out_path, f'{pMHC}.csv'), index=False)
    
    return pMHC_binding_score, pMHC_rank_score


def TCR_crossReactivity_finder(antigen1, antigen2, MHC1=None, MHC2=None, mhc_class1="I", mhc_class2="I", exact=True, is_cal_coding_seq=True,
                               alpha1V=None, alpha1J=None, beta1V=None, beta1J=None,
                               alpha2V=None, alpha2J=None, beta2V=None, beta2J=None,
                               out_path="./", with_cuda=True,
                               beta_cdr3_num=10000, alpha_cdr3_num=3000, beta_temperature=1, alpha_temperature=0.8,
                               beta_cdr3_maxLen=20, alpha_cdr3_maxLen=20,
                               beta_vj_num=5, alpha_vj_num=5, 
                               binding_batchSize=256, seed=3407, bg_num=1000, rank_thre=100,
                               processes=4, chunksize=10000):
    original_stdout = sys.stdout # 保存原始标准输出 
    sys.stdout = open(os.devnull, 'w') # 重定向
    pMHC1_binding_score, pMHC1_rank_score = pMHC_binding_predict_single(antigen1, MHC1, with_cuda=with_cuda,
                                                                        is_cal_rank=True, seed=seed, bg_num=bg_num, mhc_class=mhc_class1)
    pMHC2_binding_score, pMHC2_rank_score = pMHC_binding_predict_single(antigen2, MHC2, with_cuda=with_cuda,
                                                                        is_cal_rank=True, seed=seed, bg_num=bg_num, mhc_class=mhc_class2)
    sys.stdout.close()  # 关闭空设备的文件描述符
    sys.stdout = original_stdout # 重定向
    print(f'pMHC1 binding_score is {pMHC1_binding_score}, rank score is {pMHC1_rank_score}')
    print(f'pMHC2 binding_score is {pMHC2_binding_score}, rank score is {pMHC2_rank_score}')
    print("=="*30)
    
    MHC_mapping_df = pd.read_csv(MHC_psedo_path, sep="\s+", names=["MHC", "MHC_psedo"])
    if not MHC1.startswith("X"):
        MHC1 = MHC_mapping_df.loc[MHC_mapping_df["MHC"]==MHC1, "MHC_psedo"].values[0]
    else:
        MHC1 = "X" * MHC_mask_num
    if not MHC2.startswith("X"):
        MHC2 = MHC_mapping_df.loc[MHC_mapping_df["MHC"]==MHC2, "MHC_psedo"].values[0]
    else:
        MHC2 = "X" * MHC_mask_num
    
    print("Stage 1: Generating beta TCRs for two antigens... ...")
    sys.stdout = open(os.devnull, 'w') # 重定向
    generate_betaCDR3s_1 = generate_betaTCRs_and_processed(antigen1, beta_cdr3_num, beta_cdr3_maxLen, with_cuda, beta_temperature)
    generate_betaCDR3s_2 = generate_betaTCRs_and_processed(antigen2, beta_cdr3_num, beta_cdr3_maxLen, with_cuda, beta_temperature)
    # 寻找相同的或者只差一个氨基酸的beta链
    if exact:
        results_beta_tcrs = finder_TCRs(generate_betaCDR3s_1, generate_betaCDR3s_2, threshold=0, processes=processes, chunksize=chunksize)
    else:
        results_beta_tcrs = finder_TCRs(generate_betaCDR3s_1, generate_betaCDR3s_2, threshold=1, processes=processes, chunksize=chunksize)
    same_beta_tcrs, beta_tcrs_1, beta_tcrs_2 = [], [], []
    for tcr1, tcr2, distance in results_beta_tcrs:
        if distance == 0:
            same_beta_tcrs.append(tcr1)
        else:
            beta_tcrs_1.append(tcr1)
            beta_tcrs_2.append(tcr2)
    sys.stdout.close()  # 关闭空设备的文件描述符
    sys.stdout = original_stdout # 重定向
    print("=="*15)
    
    print("Stage 2: Generating alpha TCRs for two antigens... ...")
    sys.stdout = open(os.devnull, 'w') # 重定向
    same_alpha_tcrs_list = []
    if len(beta_tcrs_1) != 0:
        for beta_tcr1, beta_tcr2 in tqdm.tqdm(zip(beta_tcrs_1, beta_tcrs_2), total=len(beta_tcrs_1), bar_format="{l_bar}{r_bar}"):
            generate_alphaCDR3s_1 = generate_alphaTCRs_and_processed(antigen1, beta_tcr1, alpha_cdr3_num, alpha_cdr3_maxLen, with_cuda, alpha_temperature)
            generate_alphaCDR3s_2 = generate_alphaTCRs_and_processed(antigen2, beta_tcr2, alpha_cdr3_num, alpha_cdr3_maxLen, with_cuda, alpha_temperature)
            same_alpha_tcrs = [alpha_tcr1 for alpha_tcr1 in generate_alphaCDR3s_1 if alpha_tcr1 in generate_alphaCDR3s_2]
            same_alpha_tcrs_list.append(same_alpha_tcrs)
    
    alpha_tcrs_list = []
    if len(same_beta_tcrs) != 0:
        for beta_tcr in tqdm.tqdm(same_beta_tcrs):
            generate_alphaCDR3s_1 = generate_alphaTCRs_and_processed(antigen1, beta_tcr, alpha_cdr3_num, alpha_cdr3_maxLen, with_cuda, alpha_temperature)
            generate_alphaCDR3s_2 = generate_alphaTCRs_and_processed(antigen2, beta_tcr, alpha_cdr3_num, alpha_cdr3_maxLen, with_cuda, alpha_temperature)
            if exact:
                results_alpha_tcrs = finder_TCRs(generate_alphaCDR3s_1, generate_alphaCDR3s_2, threshold=0, processes=processes, chunksize=chunksize)
            else:
                results_alpha_tcrs = finder_TCRs(generate_alphaCDR3s_1, generate_alphaCDR3s_2, threshold=1, processes=processes, chunksize=chunksize)
            alpha_tcrs_list.append(results_alpha_tcrs)
    sys.stdout.close()  # 关闭空设备的文件描述符
    sys.stdout = original_stdout # 重定向
    print("=="*15)
    
    # 整理生成序列
    antigen1_beta_list, antigen1_alpha_list = [], []
    antigen2_beta_list, antigen2_alpha_list = [], []
    distance_list = []
    
    for beta_tcr1, beta_tcr2, same_alpha_tcrs in zip(beta_tcrs_1, beta_tcrs_2, same_alpha_tcrs_list):
            for alpha_tcr in same_alpha_tcrs:
                antigen1_beta_list.append(beta_tcr1)
                antigen2_beta_list.append(beta_tcr2)
                antigen1_alpha_list.append(alpha_tcr)
                antigen2_alpha_list.append(alpha_tcr)
                distance_list.append(1)
    
    for beta_tcr, results_alpha_tcrs in zip(same_beta_tcrs, alpha_tcrs_list):
        for alpha_tcr1, alpha_tcr2, distance in results_alpha_tcrs:
            antigen1_beta_list.append(beta_tcr)
            antigen2_beta_list.append(beta_tcr)
            antigen1_alpha_list.append(alpha_tcr1)
            antigen2_alpha_list.append(alpha_tcr2)
            distance_list.append(distance)
        
    if not distance_list:
        print("No TCRs Found, program will be exited.")
        sys.exit(0)
    
    print("Stage 3: Generating or Using VJ genes for alpha and beta CDR3... ...")
    max_vj_num = 4000
    sys.stdout = open(os.devnull, 'w') # 重定向
    if alpha1V is None or alpha1J is None:
        results = []
        for i in range(0, len(antigen1_alpha_list), max_vj_num):
            tmp = generate_VJ(antigen1_alpha_list[i:i+max_vj_num], generate_num=alpha_vj_num, with_cuda=with_cuda, is_alpha=True, batch_mode=True)
            results += tmp[::alpha_vj_num] # 每个TCR生成vj_num个，取第一个
        alpha1V = [alpha_VJ[0] for alpha_VJ in results]
        alpha1J = [alpha_VJ[1] for alpha_VJ in results]
    else:
        alpha1V = [alpha1V] * len(antigen1_alpha_list)
        alpha1J = [alpha1J] * len(antigen1_alpha_list)
    
    if beta1V is None or beta1J is None:
        results = []
        for i in range(0, len(antigen1_beta_list), max_vj_num):
            tmp = generate_VJ(antigen1_beta_list[i:i+max_vj_num], generate_num=beta_vj_num, with_cuda=with_cuda, is_alpha=False, batch_mode=True)
            results += tmp[::beta_vj_num]
        beta1V = [beta_VJ[0] for beta_VJ in results]
        beta1J = [beta_VJ[1] for beta_VJ in results]
    else:
        beta1V = [beta1V] * len(antigen1_beta_list)
        beta1J = [beta1J] * len(antigen1_beta_list)
    
    if alpha2V is None or alpha2J is None:
        results = []
        for i in range(0, len(antigen2_alpha_list), max_vj_num):
            tmp = generate_VJ(antigen2_alpha_list[i:i+max_vj_num], generate_num=alpha_vj_num, with_cuda=with_cuda, is_alpha=True, batch_mode=True)
            results += tmp[::alpha_vj_num] # 每个TCR生成vj_num个，取第一个
        alpha2V = [alpha_VJ[0] for alpha_VJ in results]
        alpha2J = [alpha_VJ[1] for alpha_VJ in results]
    else:
        alpha2V = [alpha2V] * len(antigen2_alpha_list)
        alpha2J = [alpha2J] * len(antigen2_alpha_list)
    
    if beta2V is None or beta2J is None:
        results = []
        for i in range(0, len(antigen2_beta_list), max_vj_num):
            tmp = generate_VJ(antigen2_beta_list[i:i+max_vj_num], generate_num=beta_vj_num, with_cuda=with_cuda, is_alpha=False, batch_mode=True)
            results += tmp[::beta_vj_num]
        beta2V = [beta_VJ[0] for beta_VJ in results]
        beta2J = [beta_VJ[1] for beta_VJ in results]
    else:
        beta2V = [beta2V] * len(antigen2_beta_list)
        beta2J = [beta2J] * len(antigen2_beta_list)
    sys.stdout.close()  # 关闭空设备的文件描述符
    sys.stdout = original_stdout # 重定向
    print("=="*15)
    
    print("Stage 4: Binding Prediction & Filtering... ...")
    sys.stdout = open(os.devnull, 'w') # 重定向
    final_df_1 = pd.DataFrame({'alphaV': alpha1V, 'alphaJ': alpha1J, 
                               'alphaCDR3': antigen1_alpha_list,
                               'betaV': beta1V, 'betaJ': beta1J,
                               'betaCDR3': antigen1_beta_list,
                               'antigen': antigen1,
                               'MHC': MHC1})
    final_df_2 = pd.DataFrame({'alphaV': alpha2V, 'alphaJ': alpha2J, 
                               'alphaCDR3': antigen2_alpha_list,
                               'betaV': beta2V, 'betaJ': beta2J,
                               'betaCDR3': antigen2_beta_list,
                               'antigen': antigen2,
                               'MHC': MHC2})
    final_df_1 = correct_generate_error(final_df_1)
    final_df_2 = correct_generate_error(final_df_2)
    
    final_df_1.to_csv(os.path.join(tmp_path, f'{antigen1}.tsv'), index=False, sep="\t", header=None)
    final_df_2.to_csv(os.path.join(tmp_path, f'{antigen2}.tsv'), index=False, sep="\t", header=None)
    
    pred_df_1 = binding_predict_pmhc_specific(os.path.join(tmp_path, f'{antigen1}.tsv'), batch_size=binding_batchSize,
                                with_cuda=with_cuda, include_label=False, is_cal_rank=True, seed=seed, bg_num=bg_num)
    pred_df_2 = binding_predict_pmhc_specific(os.path.join(tmp_path, f'{antigen2}.tsv'), batch_size=binding_batchSize,
                                with_cuda=with_cuda, include_label=False, is_cal_rank=True, seed=seed, bg_num=bg_num)
    
    final_df = pd.concat([pred_df_1, pred_df_2], axis=0, ignore_index=True)
    final_df["distance"] = distance_list * 2
    final_df["pair"] = list(range(len(antigen1_alpha_list))) * 2
    
    sys.stdout.close()  # 关闭空设备的文件描述符
    sys.stdout = original_stdout # 重定向
    print("=="*15)
    
    # delete tmp files
    os.remove(os.path.join(tmp_path, f'{antigen1}.tsv'))
    os.remove(os.path.join(tmp_path, f'{antigen2}.tsv'))
    
    # filter 
    final_df = final_df[final_df["rank_score"] <= rank_thre]
    valid_pairs = final_df.pair.value_counts()[final_df.pair.value_counts()==2].index.tolist()
    final_df = final_df[final_df.pair.isin(valid_pairs)]
    final_df = final_df.reset_index(drop=True)
    
    if is_cal_coding_seq:
        final_df = add_whole_coding_sequence(final_df)
    
    # Save
    final_df.to_csv(os.path.join(out_path, 'TCR_finder_results.csv'), index=False)
    
    print("Calculation Down.")
    
    return pMHC1_binding_score, pMHC1_rank_score, pMHC2_binding_score, pMHC2_rank_score
