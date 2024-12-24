import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
from dataset import WordVocab
import argparse

from utils import generate_TCRs_for_one_antigen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-epitope", "--epitope", type=str, required=True, help="epitope sequence for design")
    parser.add_argument("-mhc", "--mhc", type=str, required=True, help="MHC allele for design")
    parser.add_argument("-mhc_type", "--mhc_type", type=str, required=False, default="I", help="MHC type, I or II, default is I")
    parser.add_argument("-o", "--output_dir", type=str, default="./", 
                        help="path to the output directory, default is ./")
    # generate settings
    parser.add_argument("-gen_beta_num", "--gen_beta_num", type=int, default=100, help="number of beta TCRs to generate")
    parser.add_argument("-gen_alpha_num", "--gen_alpha_num", type=int, default=10, help="number of alpha TCRs to generate")
    parser.add_argument("-beta_temperature", "--beta_temperature", type=float, default=1, help="temperature for beta TCR generation")
    parser.add_argument("-alpha_temperature", "--alpha_temperature", type=float, default=0.8, help="temperature for alpha TCR generation")
    
    # binding settings
    parser.add_argument("-rank_threshold", "--rank_threshold", type=float, default=100, 
                        help="rank score threshold for binding, default is 100, which means no filtering")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=32, help="batch size for model prediction")
    parser.add_argument("-bg_num", "--bg_num", type=int, default=1000, help="number of background TCRs")
    parser.add_argument("-seed", "--seed", type=int, default=3407, help="random seed for reproducibility")
    parser.add_argument("-healthy_bg", "--healthy_bg", type=str, default="False", help="use healthy background TCRs (True) or all background TCRs (False)")
    
    # universal settings
    parser.add_argument("-cuda", "--with_cuda", type=str, default="True", help="use cuda or not")
    parser.add_argument("-coding_seq", "--coding_seq", type=str, default="False", help="output TCR nucleotide sequence")
    
    args = parser.parse_args()
    # 将字符串转换为bool
    args.healthy_bg = args.healthy_bg.lower() == "true"
    args.with_cuda = args.with_cuda.lower() == "true"
    args.coding_seq = args.coding_seq.lower() == "true"
    
    with open("TcrDesign/tmp/args.pkl", "wb") as f:
        pickle.dump(args, f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    _, __ = generate_TCRs_for_one_antigen(args.epitope, args.mhc, args.with_cuda, args.mhc_type, args.output_dir, args.coding_seq, 
                                          args.gen_beta_num, args.gen_alpha_num, args.beta_temperature, args.alpha_temperature, 
                                          20, 20, 5, 5, args.rank_threshold, True, args.batch_size, args.seed, args.bg_num)
    
    print("Running completed")
    
    os.remove("TcrDesign/tmp/args.pkl")
