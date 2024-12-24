import os
import re
import subprocess
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def process_row_highThrough(args):
    try:
        index, row = args
        # print(f'This is row {index}')
        aa_seqs = row.iloc[13:].values.tolist()
        column_names = row.iloc[13:].index
        cdr3, seq="",""
        cdr1, cdr2="",""
        exist_pos = []
        for pos, res in zip(column_names, aa_seqs):
            pos = int(re.findall(r'\d+', pos)[0]) # extract numbers from string
            # if pos in exist_pos:
            #     continue
            # else:
            #     exist_pos.append(pos)
            if res != "-":
                seq += res
                if pos >= 104 and pos <= 118: # 不包含C和F
                    cdr3 += res
                elif pos >= 27 and pos <= 38:
                    cdr1 += res
                elif pos >= 56 and pos <= 65:
                    cdr2 += res
        return cdr1, cdr2, cdr3, seq
    except Exception as e:
        print(f"Error processing row: {e}")
        return None, None, None, None


def summrize_cdr(TRA_results, TRB_results):
    TRA_result_dict = {
        'Acdr1': [], 'Acdr2': [], 'Acdr3': [], 'Aseq': []
    }
    TRB_result_dict = {
        'Bcdr1': [], 'Bcdr2': [], 'Bcdr3': [], 'Bseq': []
    }
    for alpha_aa_list in TRA_results:
        if alpha_aa_list is not None:
            TRA_result_dict['Acdr1'].append(alpha_aa_list[0])
            TRA_result_dict['Acdr2'].append(alpha_aa_list[1])
            TRA_result_dict['Acdr3'].append(alpha_aa_list[2])
            TRA_result_dict['Aseq'].append(alpha_aa_list[3])
        else:
            TRA_result_dict['Acdr1'].append(None)
            TRA_result_dict['Acdr2'].append(None)
            TRA_result_dict['Acdr3'].append(None)
            TRA_result_dict['Aseq'].append(None)
    for beta_aa_list in TRB_results:
        if beta_aa_list is not None:
            TRB_result_dict['Bcdr1'].append(beta_aa_list[0])
            TRB_result_dict['Bcdr2'].append(beta_aa_list[1])
            TRB_result_dict['Bcdr3'].append(beta_aa_list[2])
            TRB_result_dict['Bseq'].append(beta_aa_list[3])
        else:
            TRB_result_dict['Bcdr1'].append(None)
            TRB_result_dict['Bcdr2'].append(None)
            TRB_result_dict['Bcdr3'].append(None)
            TRB_result_dict['Bseq'].append(None)
    return TRA_result_dict, TRB_result_dict


def add_protein_sequence_highThrough(df, max_threads=48, max_workers=4,
                                     tmp="/home/data/sdb/dkx/TCR_structure/tmp"):
    os.makedirs(tmp, exist_ok=True)
    
    print("Executing thimble...")
    thimble_df = df.copy()
    thimble_df["TCR_name"] = list(range(len(df)))
    thimble_df = thimble_df[["TCR_name", "Va", "Ja", "CDR3a", "Vb", "Jb", "CDR3b"]]
    thimble_df = thimble_df.rename(columns={"Va": "TRAV", "Ja": "TRAJ", "CDR3a": "TRA_CDR3", "Vb": "TRBV", "Jb": "TRBJ", "CDR3b": "TRB_CDR3"})
    # TRAC    TRBC    TRA_leader      TRB_leader      Linker     Link_order      TRA_5_prime_seq TRA_3_prime_seq TRB_5_prime_seq TRB_3_prime_seq
    thimble_df = thimble_df.assign(TRAC=pd.NA, TRBC=pd.NA, TRA_leader=pd.NA, TRB_leader=pd.NA, Linker=pd.NA, 
                                   Link_order=pd.NA, TRA_5_prime_seq=pd.NA, TRA_3_prime_seq=pd.NA, TRB_5_prime_seq=pd.NA, TRB_3_prime_seq=pd.NA)
    thimble_df.to_csv(f"{tmp}/thimble.tsv", index=False, sep="\t")
    # run thimble
    subprocess.run(["thimble", "-i", f"{tmp}/thimble.tsv", "-o", f"{tmp}/thimble_out", "-s", "HUMAN", "-r", "ab"])
    # read thimble output & add to df
    thimble_out = pd.read_csv(f"{tmp}/thimble_out.tsv", sep="\t", dtype={"TCR_name": "str"})
    thimble_out = thimble_out[["TRA_aa", "TRB_aa"]]
    df["TRA_aa"] = thimble_out["TRA_aa"].values
    df["TRB_aa"] = thimble_out["TRB_aa"].values
    df = df.dropna().reset_index(drop=True)
    
    # wirte to fasta file
    TRA_aa_str = ""
    TRB_aa_str = ""
    for i, row in df.iterrows():
        TRA_aa_str += f">{i}\n{row['TRA_aa']}\n"
        TRB_aa_str += f">{i}\n{row['TRB_aa']}\n"
    with open(f"{tmp}/TRA.fasta", "w") as f:
        f.write(TRA_aa_str) 
    with open(f"{tmp}/TRB.fasta", "w") as f:
        f.write(TRB_aa_str)
    
    # execute anarci
    print("Executing ANARCI...")
    subprocess.run(["ANARCI", "-i", f"{tmp}/TRA.fasta", "--csv", "--outfile", f"{tmp}/TRA", "--ncpu", f"{max_threads}", 
                    "--scheme", "imgt", "--restrict", "A", "--use_species", "human"])
    subprocess.run(["ANARCI", "-i", f"{tmp}/TRB.fasta", "--csv", "--outfile", f"{tmp}/TRB", "--ncpu", f"{max_threads}",
                    "--scheme", "imgt", "--restrict", "B", "--use_species", "human"])
    # read anarci output & operate
    TRA_df = pd.read_csv(f"{tmp}/TRA_A.csv")
    TRB_df = pd.read_csv(f"{tmp}/TRB_B.csv")
    
    print("Parsing CDRs...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        TRA_results = list(tqdm(executor.map(process_row_highThrough, [(i, row) for i, row in TRA_df.iterrows()]), total=len(TRA_df)))
        TRB_results = list(tqdm(executor.map(process_row_highThrough, [(i, row) for i, row in TRB_df.iterrows()]), total=len(TRB_df)))
    # Summarize results
    TRA_result_dict, TRB_result_dict = summrize_cdr(TRA_results, TRB_results)
    TRA_df = TRA_df[["Id"]].assign(**TRA_result_dict).dropna().reset_index(drop=True)
    TRB_df = TRB_df[["Id"]].assign(**TRB_result_dict).dropna().reset_index(drop=True)
    TRA_B_df = TRA_df.merge(TRB_df, on="Id", how="inner") # merge TRA and TRB
    
    # final df
    df["Id"] = list(range(len(df)))
    df = df.merge(TRA_B_df, on="Id", how="inner")
    df = df.sort_values(by="Id", ascending=True).dropna().reset_index(drop=True).drop(columns=["Id"])
    df = df.drop(columns=["TRA_aa", "TRB_aa"]) # drop redundant columns(full length TCRa/b)
    df = df.loc[df.CDR3a==df.Acdr3, :] # ensure CDR3a and Acdr3 are the same
    df = df.loc[df.CDR3b==df.Bcdr3, :]
    
    # delete tmp file
    os.remove(f"{tmp}/thimble.tsv")
    os.remove(f"{tmp}/thimble_out.tsv")
    os.remove(f"{tmp}/TRA.fasta")
    os.remove(f"{tmp}/TRB.fasta")
    os.remove(f"{tmp}/TRA_A.csv")
    os.remove(f"{tmp}/TRB_B.csv")
    
    print(f"lose {len(thimble_df)-len(df)} TCRs due to failed parsing")
    print(f"Successfully parsed {len(df)} TCRs")
    print("Done!")
    
    return df
