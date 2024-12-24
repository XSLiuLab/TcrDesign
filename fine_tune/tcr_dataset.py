import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import tqdm
import torch
import random
import math

MHC_psedo_path = "/home/data/sdb/dkx/TCR_Project/data/data_collected_processed/mhc_pseudo/mhc_all.dat"

class FineTuneDataset(Dataset):
    def __init__(self, gene_dat_path, vocab, data_path=None, tcr_max_len=22, pmhc_max_len=58, include_label=True,
                 is_pandas_df=False, pandas_df=None, verbose=True):
        super().__init__()
        self.data_path = data_path
        self.gene_data_path = gene_dat_path
        self.vocab = vocab
        self.tcr_max_len = tcr_max_len
        self.pmhc_max_len = pmhc_max_len
        self.include_label = include_label
        self.is_pandas_df = is_pandas_df
        self.pandas_df = pandas_df
        
        if not is_pandas_df:
            with open(data_path, "r", encoding="utf-8") as f:
                if verbose:
                    self.lines = [line[:-1].split("\t") for line in tqdm.tqdm(f, desc="Loading Dataset")]
                else:
                    self.lines = [line[:-1].split("\t") for line in f]
        else:
            if verbose:
                self.lines = [row.tolist() for _, row in tqdm.tqdm(pandas_df.iterrows(), desc="Loading Dataset")]
            else:
                self.lines = [row.tolist() for _, row in pandas_df.iterrows()]
            
        self.corpus_lines = len(self.lines)
        self.gene_mapping = pd.read_csv(gene_dat_path)
    
    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, index):
        aV, aJ, aCDR3 = self.lines[index][0], self.lines[index][1], self.lines[index][2]
        bV, bJ, bCDR3 = self.lines[index][3], self.lines[index][4], self.lines[index][5]
        pep, mhc= self.lines[index][6], self.lines[index][7]
        if self.include_label:
            score = float(self.lines[index][8])
        
        VJ = [self.gene_mapping.loc[self.gene_mapping["gene_name"] == gene, "index"].values[0] for gene in [aV, aJ, bV, bJ]]
        
        aCDR3 = [self.vocab.mask_index]*len(aCDR3) if aCDR3.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in aCDR3]
        aCDR3 = [self.vocab.sos_index] + aCDR3 + [self.vocab.eos_index]
        aCDR3 = aCDR3[:self.tcr_max_len]
        padding = [self.vocab.pad_index for _ in range(self.tcr_max_len - len(aCDR3))]
        aCDR3.extend(padding)
        
        bCDR3 = [self.vocab.mask_index]*len(bCDR3) if bCDR3.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in bCDR3]
        bCDR3 = [self.vocab.sos_index] + bCDR3 + [self.vocab.eos_index]
        bCDR3 = bCDR3[:self.tcr_max_len]
        padding = [self.vocab.pad_index for _ in range(self.tcr_max_len - len(bCDR3))]
        bCDR3.extend(padding)
        
        pep = [self.vocab.mask_index]*len(pep) if pep.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in pep]
        mhc = [self.vocab.mask_index]*len(mhc) if mhc.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in mhc]
        mhc = [self.vocab.sos_index] + mhc + [self.vocab.eos_index]
        pep = pep + [self.vocab.eos_index]
        segment_label = ([1 for _ in range(len(mhc))] + [2 for _ in range(len(pep))])[:self.pmhc_max_len]
        pmhc = (mhc + pep)[:self.pmhc_max_len]
        padding = [self.vocab.pad_index for _ in range(self.pmhc_max_len - len(pmhc))]
        pmhc.extend(padding), segment_label.extend(padding)
        
        if self.include_label:
            output = {"VJ": VJ,
                    "aCDR3": aCDR3,
                    "bCDR3": bCDR3,
                    "pMHC": pmhc,
                    "pMHC_segment_label": segment_label,
                    "binding_score": score}
        else:
            output = {"VJ": VJ,
                    "aCDR3": aCDR3,
                    "bCDR3": bCDR3,
                    "pMHC": pmhc,
                    "pMHC_segment_label": segment_label}

        return {key: torch.tensor(value) for key, value in output.items()}


class Distance_FineTuneDataset(Dataset):
    def __init__(self, gene_dat_path, vocab, data_path=None, tcr_max_len=22, pmhc_max_len=58, peptide_max_len=23, include_label=True):
        super().__init__()
        
        self.data_path = data_path
        self.info_df = pd.read_csv(self.data_path)
        self.MHC_mapping_df = pd.read_csv(MHC_psedo_path, sep="\s+", names=["MHC", "MHC_psedo"])
           
        self.gene_data_path = gene_dat_path
        self.vocab = vocab
        self.tcr_max_len = tcr_max_len
        self.peptide_max_len = peptide_max_len
        self.pmhc_max_len = pmhc_max_len
        self.include_label = include_label
        
        self.gene_mapping = pd.read_csv(gene_dat_path)
        
    def __len__(self):
        return len(self.info_df)
    
    def space_distance(self, x: list, y: list):
        x = list(map(float, x))
        y = list(map(float, y))
        distance = math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)
        return round(distance, 4)
    
    def normalize_distance_matrix(self, dist_matrix):
        # Find the minimum and maximum distances
        min_dist = np.min(dist_matrix)
        max_dist = np.max(dist_matrix)
        # Perform the normalization
        normalized_matrix = 1 - (dist_matrix - min_dist) / (max_dist - min_dist)
        return normalized_matrix
    
    def get_true_norm_distance_matrix(self, pdb_id):
        pdb_path = "/home/data/sdb/dkx/TCR_Project/data/data_plot/126_pdbs/" + pdb_id + "_need.pdb"
        distance_df = pd.read_csv(pdb_path)
        chains = distance_df["chain"].unique().tolist()
        cdr3_df = distance_df[distance_df["chain"]==chains[0]]
        pep_df = distance_df[distance_df["chain"]==chains[1]]

        contact_map = np.zeros((len(cdr3_df), len(pep_df)))
        for i in range(len(cdr3_df)):
            distances = [self.space_distance(list(cdr3_df.iloc[i, 7:10]), list(pep_df.iloc[j, 7:10])) for j in range(len(pep_df))]
            contact_map[i, :] = distances
        contact_map = self.normalize_distance_matrix(contact_map)
        
        new_contact_map = np.zeros((self.tcr_max_len-2, self.peptide_max_len-2))
        new_contact_map[:contact_map.shape[0], :contact_map.shape[1]] = contact_map
        
        return new_contact_map

    def __getitem__(self, idx):
        row = self.info_df.iloc[idx, :] # 取出一行
        pdb_id = row["PDB ID"]
        
        aV = row["TRAV"]
        aJ = "X"
        bV = row["TRBV"]
        bJ = "X"
        aCDR3 = row["CDR3a"]
        bCDR3 = row["CDR3b"]
        pep = row["Epitope"]
        mhc = row["MHC"].replace("*", "")
        mhc = self.MHC_mapping_df.loc[self.MHC_mapping_df["MHC"]==mhc, "MHC_psedo"].values[0]
        
        ## tranform to index
        VJ = [self.gene_mapping.loc[self.gene_mapping["gene_name"] == gene, "index"].values[0] for gene in [aV, aJ, bV, bJ]]
        
        aCDR3 = [self.vocab.mask_index]*len(aCDR3) if aCDR3.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in aCDR3]
        aCDR3 = [self.vocab.sos_index] + aCDR3 + [self.vocab.eos_index]
        aCDR3 = aCDR3[:self.tcr_max_len]
        padding = [self.vocab.pad_index for _ in range(self.tcr_max_len - len(aCDR3))]
        aCDR3.extend(padding)
        
        bCDR3 = [self.vocab.mask_index]*len(bCDR3) if bCDR3.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in bCDR3]
        bCDR3 = [self.vocab.sos_index] + bCDR3 + [self.vocab.eos_index]
        bCDR3 = bCDR3[:self.tcr_max_len]
        padding = [self.vocab.pad_index for _ in range(self.tcr_max_len - len(bCDR3))]
        bCDR3.extend(padding)
        
        pep = [self.vocab.mask_index]*len(pep) if pep.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in pep]
        mhc = [self.vocab.mask_index]*len(mhc) if mhc.startswith("X") else \
            [self.vocab.stoi.get(token, self.vocab.unk_index) for token in mhc]
        mhc = [self.vocab.sos_index] + mhc + [self.vocab.eos_index]
        pep = pep + [self.vocab.eos_index]
        segment_label = ([1 for _ in range(len(mhc))] + [2 for _ in range(len(pep))])[:self.pmhc_max_len]
        pmhc = (mhc + pep)[:self.pmhc_max_len]
        padding = [self.vocab.pad_index for _ in range(self.pmhc_max_len - len(pmhc))]
        pmhc.extend(padding), segment_label.extend(padding)
        
        if self.include_label:
            true_norm_distance_matrix = self.get_true_norm_distance_matrix(pdb_id)
            output = {"VJ": VJ,
                    "aCDR3": aCDR3,
                    "bCDR3": bCDR3,
                    "pMHC": pmhc,
                    "pMHC_segment_label": segment_label,
                    "norm_distance_matrix": true_norm_distance_matrix}
        else:
            output = {"VJ": VJ,
                    "aCDR3": aCDR3,
                    "bCDR3": bCDR3,
                    "pMHC": pmhc,
                    "pMHC_segment_label": segment_label}

        return {key: torch.tensor(value) for key, value in output.items()}
